from __future__ import annotations

import mowl
import torch

from common.json_extraction import extract_json
from filter_candidates.subroutines import _load_metrics, _load_definitions, filter_candidates, load_candidates, \
    LLMConfig, _build_langchain_ollama, _build_user_prompt, _write_accepted, _init_model_for_embeddings, \
    _get_system_prompt

mowl.init_jvm("6g")
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import dotenv
from util.logger_config import config
from common.settings import build_settings

logger = logging.getLogger(__name__)
config(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
SRC_ROOT = PROJECT_ROOT / "src"
PROMPTS_ROOT = PROJECT_ROOT / "prompts"


def _parse_args(settings: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter candidate EL axioms using hybrid MOWL cosine + LLM plausibility")
    p.add_argument("--candidates", default=str(GENERATED_ROOT / "candidate_el.ttl"),
                   help="Input TTL with candidate axioms (rdfs:subClassOf)")
    p.add_argument("--out", default=str(GENERATED_ROOT / "accepted_el.ttl"), help="Output TTL path for accepted axioms")
    p.add_argument("--metrics", default=str(REPORTS_ROOT / "mowl_metrics.json"),
                   help="Path to training metrics JSON (for tau, hyperparams)")
    p.add_argument("--w-cos", type=float, default=0.7,
                   help="Weight for cosine in hybrid score (final = w*cos + (1-w)*plaus)")
    p.add_argument("--tau", type=float, default=None,
                   help="Override threshold tau; defaults to selected_tau from metrics or 0.7 if missing")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of candidates to process (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM calls; use cosine only for filtering")
    # LLM options (LangChain + Ollama backend)
    env_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    p.add_argument("--ollama-host", default=env_host, help="Ollama server base URL (used via LangChain)")
    p.add_argument("--ollama-model", default=str(settings.get("model_name", "llama3:instruct")),
                   help="Ollama model name to use for plausibility scoring (via LangChain)")
    p.add_argument("--temperature", type=float, default=float(settings.get("temperature", 0.0)), help="LLM temperature")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-call timeout seconds for LLM requests (best-effort)")
    return p.parse_args()


def _cosine_scores(model, pairs: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    cos = torch.nn.functional.cosine_similarity
    ent_map = model.get_embeddings()[0]
    scores: Dict[Tuple[str, str], float] = {}
    for sub, sup in pairs:
        v1 = ent_map.get(sub)
        v2 = ent_map.get(sup)
        if v1 is None or v2 is None:
            logger.debug("Missing embedding for %s or %s; assigning cosine=0.0", sub, sup)
            scores[(sub, sup)] = 0.0
            continue
        t1 = torch.tensor(v1)
        t2 = torch.tensor(v2)
        sim = float(cos(t1, t2, dim=0).item())
        # Clip to [0, 1] for hybrid combination
        sim = max(0.0, min(1.0, sim))
        scores[(sub, sup)] = sim
    return scores


def _score_plausibility(
    pairs: Iterable[Tuple[str, str]],
    llm_cfg: LLMConfig,
    defs: Dict[str, Dict[str, str]],
    dry_run: bool = False,
) -> Dict[Tuple[str, str], float]:
    results: Dict[Tuple[str, str], float] = {}
    if dry_run:
        # Return neutral plausibility
        for p in pairs:
            results[p] = 0.5
        return results

    chat = _build_langchain_ollama(llm_cfg)
    if chat is None:
        for p in pairs:
            results[p] = 0.5
        return results

    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    except Exception as e:
        logger.warning("langchain-core messages not available (%s). Falling back to plausibility=0.5.", e)
        for p in pairs:
            results[p] = 0.5
        return results

    # Load system prompt from file (with default fallback)
    system_prompt = _get_system_prompt(PROMPTS_ROOT)

    for sub, sup in pairs:
        prompt = _build_user_prompt(sub, sup, defs)
        try:
            # Build chat messages for LangChain
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
            resp = chat.invoke(messages)
            # resp is an AIMessage; extract the content
            text = getattr(resp, "content", None)
            logger.info(text)
            if text is None:
                text = str(resp)
            obj = extract_json(text)
            val = float(obj.get("plausibility", 0.0))
            val = max(0.0, min(1.0, val))
        except Exception as e:
            logger.warning("LLM scoring failed for %s ⊑ %s (LangChain/Ollama): %s", sub, sup, e)
            val = 0.5
        results[(sub, sup)] = val
    return results


def main():
    # Read settings
    dotenv.load_dotenv()
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    args = _parse_args(settings)
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)

    # Load information from metrics report
    metrics = _load_metrics(metrics_path)
    tau = float(args.tau) if args.tau is not None else float(metrics.get("selected_tau") or 0.70)
    w_cos = float(args.w_cos)
    if not (0.0 <= w_cos <= 1.0):
        raise ValueError("--w-cos must be in [0,1]")

    # Load the candidates from candidate_el.ttl
    pairs = load_candidates(args, candidates_path)

    # Create the cosine similarities
    logger.info("Initializing ELEmbeddings model for cosine scoring…")
    model, dataset = _init_model_for_embeddings(metrics, CHECKPOINTS_ROOT, SRC_ROOT)
    cos_scores = _cosine_scores(model, pairs)

    # Create the LLM plausibility scores
    llm_cfg = LLMConfig(
        host=args.ollama_host,
        model=args.ollama_model,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    defs = _load_definitions(DATA_ROOT / "definitions_enriched.csv")
    plaus_scores = _score_plausibility(pairs, llm_cfg, defs, dry_run=bool(args.dry_run))

    # Filter candidates to accept
    accepted, hybrid_details = filter_candidates(pairs, cos_scores, plaus_scores, tau, w_cos)

    # Output accepted candidates
    n_written = _write_accepted(out_path, accepted)
    logger.info("Accepted %d/%d candidates at tau=%.2f with w_cos=%.2f. Output: %s", n_written, len(pairs), tau, w_cos,
                out_path)

    # Write a small report next to the TTL for traceability
    try:
        report = {
            "tau": tau,
            "w_cos": w_cos,
            "num_candidates": len(pairs),
            "num_accepted": n_written,
            "ollama_model": args.ollama_model,
            "ollama_host": args.ollama_host,
            "dry_run": bool(args.dry_run),
            "details": hybrid_details[:200],  # cap for brevity
        }
        with (out_path.with_suffix(out_path.suffix + ".json")).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.debug("Could not write hybrid report JSON: %s", e)


if __name__ == "__main__":
    main()
