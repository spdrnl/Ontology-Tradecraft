from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import torch
from rdflib import Graph, RDFS, URIRef

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)


# Lazy imports for MOWL extension to avoid JVM init if not needed
def _lazy_mowl_imports():  # noqa: D401
    """Import dataset/model only when needed (keeps this script lightweight)."""
    from mowl_ext.datasets import OtcPathDataset  # type: ignore
    from mowl_ext.otc_model import OtcModel  # type: ignore
    return OtcPathDataset, OtcModel


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


def filter_candidates(pairs: list[tuple[str, str]], cos_scores: dict[tuple[str, str], float],
                      plaus_scores: dict[tuple[str, str], float], tau: float, w_cos: float) -> tuple[
    list[tuple[str, str]], list[dict]]:
    accepted: List[Tuple[str, str]] = []
    hybrid_details: List[dict] = []
    for p in pairs:
        c = cos_scores.get(p, 0.0)
        l = plaus_scores.get(p, 0.5)
        final = w_cos * c + (1.0 - w_cos) * l
        if final >= tau:
            accepted.append(p)
        hybrid_details.append({"sub": p[0], "sup": p[1], "cosine": c, "plausibility": l, "hybrid": final})
    return accepted, hybrid_details


def load_candidates(args, candidates_path: Path) -> list[tuple[str, str]]:
    logger.info("Reading candidates from %s", candidates_path)
    pairs_all = _read_candidates(candidates_path)
    if args.limit and args.limit > 0:
        pairs = pairs_all[: args.limit]
    else:
        pairs = pairs_all
    logger.info("Loaded %d candidate subclass axioms", len(pairs))
    return pairs


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


def _write_accepted(out_path: Path, pairs: Iterable[Tuple[str, str]]) -> int:
    g = Graph()
    n = 0
    for sub, sup in pairs:
        g.add((URIRef(sub), RDFS.subClassOf, URIRef(sup)))
        n += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=out_path.as_posix(), format="turtle")
    return n


def _find_checkpoint(checkpoints_root: Path) -> Optional[Path]:
    if not checkpoints_root.exists():
        return None
    # Prefer optuna.best.* then the most recent elembeddings.*.model
    optuna_best = sorted(checkpoints_root.glob("elembeddings.optuna.best.*.model"), reverse=True)
    if optuna_best:
        return optuna_best[0]
    generic = sorted(checkpoints_root.glob("elembeddings*.model"), reverse=True)
    if generic:
        return generic[0]
    return None


def _init_model_for_embeddings(metrics: dict, check_point_root, src_root):
    OtcPathDataset, OtcModel = _lazy_mowl_imports()

    train_path = Path(metrics.get("train_path", src_root / "train.ttl"))
    valid_path = Path(metrics.get("valid_path", src_root / "valid.ttl"))
    # We only need the entity space; eval_scope 'all' is fine
    dataset = OtcPathDataset(
        ontology_path=train_path.as_posix(),
        validation_path=valid_path.as_posix(),
        eval_scope="all",
    )

    # Embedding dimension must come from metrics (do not infer from checkpoint)
    if "dim" not in metrics:
        raise ValueError(
            "mowl_metrics.json must include 'dim' (embedding dimension) written by scripts/train_mowl.py"
        )
    dim = int(metrics["dim"])
    lr = float(metrics.get("lr", 1e-3))
    margin = float(metrics.get("margin", 1.0))
    reg_norm = float(metrics.get("reg_norm") or metrics.get("best_reg_norm", 1.0))

    # Resolve checkpoint strictly from metrics; no fallback discovery
    best_model_path = metrics.get("best_model_path")
    best_model_filename = metrics.get("best_model_filename")
    ckpt: Optional[Path] = None
    # Prefer absolute path if provided
    if isinstance(best_model_path, str) and best_model_path.strip():
        ckpt = Path(best_model_path)
    elif isinstance(best_model_filename, str) and best_model_filename.strip():
        ckpt = check_point_root / best_model_filename
    else:
        raise ValueError(
            "mowl_metrics.json must include 'best_model_path' or 'best_model_filename'. "
            "Please run scripts/train_mowl.py to generate metrics with checkpoint info."
        )

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint specified in metrics not found: {ckpt}. "
            f"Ensure the file exists, or regenerate metrics by retraining."
        )

    model_filepath = ckpt.as_posix()

    # Note: We intentionally avoid inferring/overriding embed_dim from checkpoint.
    # The authoritative embedding dimension is taken from mowl_metrics.json written by training.

    # Select device safely: prefer CUDA if available, otherwise CPU. Avoid forcing CUDA on systems without it.
    safe_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = OtcModel(
        dataset,
        embed_dim=dim,
        learning_rate=lr,
        margin=margin,
        model_filepath=model_filepath,
        batch_size=int(metrics.get("batch", 1024)),
        reg_norm=reg_norm,
        device=safe_device
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


def _get_system_prompt(prompts_root) -> str:
    """Read the system prompt from prompts/filter_candidates_hybrid_prompt.md.

    Falls back to DEFAULT_SYSTEM_PROMPT if the file is missing or unreadable.
    """
    prompt_path = prompts_root / "filter_candidates_hybrid_prompt.md"
    text = prompt_path.read_text(encoding="utf-8").strip()
    return text
