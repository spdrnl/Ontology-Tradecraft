from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from rdflib import Graph, RDFS
from rdflib.term import URIRef

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
SRC_ROOT = PROJECT_ROOT / "src"


# Lazy imports for MOWL extension to avoid JVM init if not needed
def _lazy_mowl_imports():  # noqa: D401
    """Import dataset/model only when needed (keeps this script lightweight)."""
    from mowl_ext.datasets import OtcPathDataset  # type: ignore
    from mowl_ext.otc_model import OtcModel  # type: ignore
    return OtcPathDataset, OtcModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter candidate EL axioms using hybrid MOWL cosine + LLM plausibility")
    p.add_argument("--candidates", default=str(GENERATED_ROOT / "candidate_el.ttl"), help="Input TTL with candidate axioms (rdfs:subClassOf)")
    p.add_argument("--out", default=str(GENERATED_ROOT / "accepted_el.ttl"), help="Output TTL path for accepted axioms")
    p.add_argument("--metrics", default=str(REPORTS_ROOT / "mowl_metrics.json"), help="Path to training metrics JSON (for tau, hyperparams)")
    p.add_argument("--w-cos", type=float, default=0.7, help="Weight for cosine in hybrid score (final = w*cos + (1-w)*plaus)")
    p.add_argument("--tau", type=float, default=None, help="Override threshold tau; defaults to selected_tau from metrics or 0.7 if missing")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of candidates to process (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM calls; use cosine only for filtering")
        # LLM options (LangChain + Ollama backend)
    p.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama server base URL (used via LangChain)")
    p.add_argument("--ollama-model", default="llama3:instruct", help="Ollama model name to use for plausibility scoring (via LangChain)")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default 0.0)")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-call timeout seconds for LLM requests (best-effort)")
    return p.parse_args()


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_candidates(ttl_path: Path) -> List[Tuple[str, str]]:
    if not ttl_path.exists():
        raise FileNotFoundError(f"Candidates TTL not found: {ttl_path}")
    g = Graph()
    g.parse(ttl_path.as_posix())
    pairs: List[Tuple[str, str]] = []
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            pairs.append((str(s), str(o)))
    return pairs


def _load_definitions(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Return mapping IRI -> {label, definition} from data/definitions.csv if available."""
    import csv

    info: Dict[str, Dict[str, str]] = {}
    if not csv_path.exists():
        logger.warning("definitions.csv not found at %s; LLM context will be minimal", csv_path)
        return info
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expected columns: IRI,label,definition (case-insensitive tolerant)
        for row in reader:
            iri = row.get("IRI") or row.get("iri") or row.get("Iri")
            if not iri:
                continue
            info[iri] = {
                "label": row.get("label") or row.get("Label") or "",
                "definition": row.get("definition") or row.get("Definition") or "",
            }
    return info


def _find_checkpoint() -> Optional[Path]:
    if not CHECKPOINTS_ROOT.exists():
        return None
    # Prefer optuna.best.* then the most recent elembeddings.*.model
    optuna_best = sorted(CHECKPOINTS_ROOT.glob("elembeddings.optuna.best.*.model"), reverse=True)
    if optuna_best:
        return optuna_best[0]
    generic = sorted(CHECKPOINTS_ROOT.glob("elembeddings*.model"), reverse=True)
    if generic:
        return generic[0]
    return None


def _init_model_for_embeddings(metrics: dict):
    OtcPathDataset, OtcModel = _lazy_mowl_imports()

    train_path = Path(metrics.get("train_path", SRC_ROOT / "train.ttl"))
    valid_path = Path(metrics.get("valid_path", SRC_ROOT / "valid.ttl"))
    # We only need the entity space; eval_scope 'all' is fine
    dataset = OtcPathDataset(
        ontology_path=train_path.as_posix(),
        validation_path=valid_path.as_posix(),
        eval_scope="all",
    )

    dim = int(metrics.get("dim", 200))
    lr = float(metrics.get("lr", 1e-3))
    margin = float(metrics.get("margin", 1.0))
    reg_norm = float(metrics.get("reg_norm") or metrics.get("best_reg_norm", 1.0))

    ckpt = _find_checkpoint()
    model_filepath = ckpt.as_posix() if ckpt else (CHECKPOINTS_ROOT / "elembeddings.model").as_posix()

    model = OtcModel(
        dataset,
        embed_dim=dim,
        learning_rate=lr,
        margin=margin,
        model_filepath=model_filepath,
        batch_size=int(metrics.get("batch", 1024)),
        reg_norm=reg_norm,
        device='cuda:0'
    )

    # Try loading checkpoint weights if a file exists
    try:
        if ckpt and Path(model_filepath).exists():
            state = torch.load(model_filepath, map_location=model.device if hasattr(model, 'device') else 'cpu')
            model.module.load_state_dict(state)
            logger.info("Loaded ELEmbeddings checkpoint: %s", model_filepath)
        else:
            logger.warning("No ELEmbeddings checkpoint found. Using current (untrained) weights for cosine.")
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", model_filepath, e)

    return model, dataset


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


@dataclass
class LLMConfig:
    host: str
    model: str
    temperature: float = 0.0
    timeout: float = 60.0


def _build_langchain_ollama(llm: LLMConfig):
    """Initialize a LangChain ChatOllama model. Return None on failure (caller will fallback)."""
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception as e:
        logger.warning("langchain_ollama not available (%s). Falling back to plausibility=0.5.", e)
        return None
    try:
        # ChatOllama parameters: model, base_url, temperature, keep_alive, num_ctx, etc.
        # Timeout control is limited; we keep llm.timeout for future use/logging.
        chat = ChatOllama(model=llm.model, base_url=llm.host, temperature=llm.temperature)
        return chat
    except Exception as e:
        logger.warning("Failed to initialize LangChain ChatOllama (%s). Falling back to plausibility=0.5.", e)
        return None


SYSTEM_PROMPT = (
    "You are a precise ontology editor expert in BFO and CCO. Judge whether a candidate OWL 2 EL "
    "subclass axiom is semantically plausible in CCO style. Evaluate only is-a (subClassOf). "
    "Penalize category mistakes (process vs. continuant, role vs. type, part-of vs. is-a). "
    "Use genus–differentia and BFO upper ontology as reference. Return a compact JSON with fields: "
    "plausibility (0–1), el_ok (boolean), issues (array), rationale (short string). Do not include extra text."
)


def _build_user_prompt(sub_iri: str, sup_iri: str, defs: Dict[str, Dict[str, str]], reference_context: str = "") -> str:
    sub_info = defs.get(sub_iri, {})
    sup_info = defs.get(sup_iri, {})
    sub_label = sub_info.get("label", "") or sub_iri.rsplit('/', 1)[-1]
    sup_label = sup_info.get("label", "") or sup_iri.rsplit('/', 1)[-1]
    sub_def = sub_info.get("definition", "")
    sup_def = sup_info.get("definition", "")
    user = (
        f"Candidate: {sub_label} ⊑ {sup_label}\n"
        f"Sub IRI: {sub_iri}\n"
        f"Super IRI: {sup_iri}\n"
        f"Sub definition: {sub_def}\n"
        f"Super definition: {sup_def}\n"
        f"Reference context (optional, bullet list):\n{reference_context}\n"
        "Scoring rubric:\n"
        "- 0.90–1.00: clear subtype\n"
        "- 0.70–0.89: likely subtype\n"
        "- 0.40–0.69: uncertain\n"
        "- 0.10–0.39: probably wrong\n"
        "- 0.00–0.09: clearly wrong\n"
        "Respond ONLY with JSON: {\"plausibility\": x.xx, \"el_ok\": true/false, \"issues\": [\"...\"], \"rationale\": \"...\"}"
    )
    return user


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

    import json as _json
    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    except Exception as e:
        logger.warning("langchain-core messages not available (%s). Falling back to plausibility=0.5.", e)
        for p in pairs:
            results[p] = 0.5
        return results

    for sub, sup in pairs:
        prompt = _build_user_prompt(sub, sup, defs)
        try:
            # Build chat messages for LangChain
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            resp = chat.invoke(messages)
            # resp is an AIMessage; extract the content
            text = getattr(resp, "content", None)
            if text is None:
                text = str(resp)
            try:
                obj = _json.loads(text)
            except _json.JSONDecodeError:
                cleaned = str(text).strip().strip("`").strip()
                obj = _json.loads(cleaned)
            val = float(obj.get("plausibility", 0.0))
            val = max(0.0, min(1.0, val))
        except Exception as e:
            logger.warning("LLM scoring failed for %s ⊑ %s (LangChain/Ollama): %s", sub, sup, e)
            val = 0.5
        results[(sub, sup)] = val
    return results


def _write_accepted(out_path: Path, pairs: Iterable[Tuple[str, str]]) -> int:
    g = Graph()
    n = 0
    for sub, sup in pairs:
        g.add((URIRef(sub), RDFS.subClassOf, URIRef(sup)))
        n += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=out_path.as_posix(), format="turtle")
    return n


def main():
    args = _parse_args()
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)

    metrics = _load_metrics(metrics_path)
    tau = float(args.tau) if args.tau is not None else float(metrics.get("selected_tau") or 0.70)
    w_cos = float(args.w_cos)
    if not (0.0 <= w_cos <= 1.0):
        raise ValueError("--w-cos must be in [0,1]")

    logger.info("Reading candidates from %s", candidates_path)
    pairs_all = _read_candidates(candidates_path)
    if args.limit and args.limit > 0:
        pairs = pairs_all[: args.limit]
    else:
        pairs = pairs_all
    logger.info("Loaded %d candidate subclass axioms", len(pairs))

    logger.info("Initializing ELEmbeddings model for cosine scoring…")
    model, dataset = _init_model_for_embeddings(metrics)
    cos_scores = _cosine_scores(model, pairs)

    defs = _load_definitions(DATA_ROOT / "definitions.csv")
    llm_cfg = LLMConfig(host=args.ollama_host, model=args.ollama_model, temperature=args.temperature, timeout=args.timeout)
    plaus_scores = _score_plausibility(pairs, llm_cfg, defs, dry_run=bool(args.dry_run))

    accepted: List[Tuple[str, str]] = []
    hybrid_details: List[dict] = []
    for p in pairs:
        c = cos_scores.get(p, 0.0)
        l = plaus_scores.get(p, 0.5)
        final = w_cos * c + (1.0 - w_cos) * l
        if final >= tau:
            accepted.append(p)
        hybrid_details.append({"sub": p[0], "sup": p[1], "cosine": c, "plausibility": l, "hybrid": final})

    n_written = _write_accepted(out_path, accepted)
    logger.info("Accepted %d/%d candidates at tau=%.2f with w_cos=%.2f. Output: %s", n_written, len(pairs), tau, w_cos, out_path)

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