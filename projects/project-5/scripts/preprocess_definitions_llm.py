import logging
from pathlib import Path
import os
import time
import re
import dotenv
import pandas as pd

from logger_config import config
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rapidfuzz import fuzz, process
from typing import List, Iterable


logger = logging.getLogger(__name__)
config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

INPUT_FILE = DATA_ROOT / "definitions.csv"
BFO_CCO_TERMS = DATA_ROOT / "bfo_cco_terms.csv"
OUTPUT_FILE = DATA_ROOT / "definitions_enriched.csv"


def read_csv(path: Path) -> pd.DataFrame:
    # skipinitialspace handles the sample header "label, definition" (with a space)
    return pd.read_csv(path, header=0, skipinitialspace=True)


def write_df_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def main():
    logger.info("===================================================================")
    logger.info("Preprocessing definitions (LLM enrichment via LangChain + Ollama)")
    logger.info("===================================================================")

    # Read input
    logger.info(f"Reading definitions CSV from: {INPUT_FILE}")
    df = read_csv(INPUT_FILE)

    required = {"iri", "label", "definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Configure LLM (Ollama must be running: `ollama serve` and model pulled)
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    llm = ChatOllama(model=model_name, temperature=temperature)

    # Load optional BFO/CCO reference terms (lightweight RAG-style context)
    reference_mode = os.getenv("REFERENCE_MODE", "retrieve").lower()  # off | full | retrieve | fuzzy
    ref_top_k = int(os.getenv("REF_TOP_K", "5"))
    ref_max_chars = int(os.getenv("REF_MAX_CHARS", "4000"))
    # Fuzzy retrieval knobs
    ref_fuzzy_scorer = os.getenv("REF_FUZZY_SCORER", "token_set_ratio").lower()
    ref_fuzzy_cutoff = int(os.getenv("REF_FUZZY_CUTOFF", "70"))  # 0–100
    # Post-processing switch: enforce capitalization of overlapping glossary labels
    enforce_label_casing = os.getenv("ENFORCE_LABEL_CASING", "false").lower() in {"1", "true", "yes", "on"}

    ref_entries = []
    full_reference_context = ""
    if BFO_CCO_TERMS.exists():
        try:
            ref_df = read_csv(BFO_CCO_TERMS)
            if not {"label", "definition"}.issubset(set(ref_df.columns)):
                logger.warning(
                    "Reference file present but missing required columns 'label' and 'definition'. Columns found: %s",
                    list(ref_df.columns),
                )
            else:
                # Build a list of dicts and a precomputed full context string
                ref_entries = [
                    {
                        "label": str(row["label"]).strip(),
                        "definition": str(row["definition"]).strip(),
                    }
                    for _, row in ref_df.iterrows()
                    if pd.notna(row.get("label")) and pd.notna(row.get("definition"))
                ]
                glossary_lines = [
                    f"- {e['label']}: {e['definition']}" for e in ref_entries
                ]
                full_reference_context = "\n".join(glossary_lines)
                if len(full_reference_context) > ref_max_chars:
                    full_reference_context = full_reference_context[: ref_max_chars] + "\n…"
                logger.info(
                    "Loaded %d reference terms from %s (mode=%s)",
                    len(ref_entries),
                    BFO_CCO_TERMS,
                    reference_mode,
                )
        except Exception as e:
            logger.warning("Failed to load reference terms from %s: %s", BFO_CCO_TERMS, e)
            ref_entries = []
            full_reference_context = ""
    else:
        if reference_mode != "off":
            logger.info("Reference file not found at %s; proceeding without external references.", BFO_CCO_TERMS)
        reference_mode = "off"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a precise ontology editor and an expert in Basic Formal Ontology (BFO) and Common Core Ontologies (CCO). "
                "Improve class definitions in a clear, concise, academic style that adheres to BFO principles."
            ),
        ),
        (
            "system",
            (
                "Reference BFO/CCO glossary (authoritative context; use labels as written when relevant):\n"
                "{reference_context}"
            ),
        ),
        (
            "user",
            (
                "Improve the following definition in 1 sentence, academic style.\n"
                "Standardize to a canonical form where appropriate (e.g., 'X is a Y that Zs'). "
                "Remove ambiguity and expand abbreviations. Capitalize BFO/CCO entity labels in definitions as per {reference_context}.\n"
                "Return only the improved definition (no prefixes, quotes, or extra commentary).\n\n"
                "Term: {label}\n"
                "Existing definition: {definition}"
            ),
        ),
    ])

    chain = prompt | llm

    # --- RapidFuzz helpers (used when reference_mode == "fuzzy") ---
    SCORERS = {
        "token_set_ratio": fuzz.token_set_ratio,   # robust to extra/missing words
        "token_sort_ratio": fuzz.token_sort_ratio, # robust to word order changes
        "wratio": fuzz.WRatio,                     # a smart overall scorer
    }

    def get_scorer(name: str):
        return SCORERS.get(name, fuzz.token_set_ratio)

    # Precompute combined strings for faster matching
    ref_strings: List[str] = [f"{e['label']} — {e['definition']}" for e in ref_entries]

    def fuzzy_top_k(query: str, top_k: int, cutoff: int) -> List[str]:
        if not ref_entries:
            return []
        results = process.extract(
            query,
            ref_strings,
            scorer=get_scorer(ref_fuzzy_scorer),
            processor=str.lower,
            limit=max(top_k, 0),
            score_cutoff=cutoff,
        )
        lines: List[str] = []
        for _match, _score, idx in results:
            e = ref_entries[idx]
            lines.append(f"- {e['label']}: {e['definition']}")
        return lines

    def simple_score(query_text: str, label: str, definition: str) -> int:
        """Lightweight relevance score based on token overlap.
        Returns an integer score; higher is better.
        """
        try:
            q = set(t for t in query_text.lower().split() if t.isalpha())
            l = set(t for t in str(label).lower().split() if t.isalpha())
            d = set(t for t in str(definition).lower().split() if t.isalpha())
            # Weighted overlap: label tokens more important than definition tokens
            return 3 * len(q & l) + 1 * len(q & d)
        except Exception:
            return 0

    def build_reference_context(term_label: str, term_definition: str) -> str:
        if reference_mode == "off" or not ref_entries:
            return ""
        if reference_mode == "full":
            return full_reference_context
        # Build query from current row
        query = f"{term_label or ''} {term_definition or ''}"
        if reference_mode == "fuzzy":
            # fuzzy mode: use RapidFuzz to select top-K relevant lines
            lines = fuzzy_top_k(query, ref_top_k, ref_fuzzy_cutoff)
        else:
            # retrieve mode: pick top-K by simple overlap against query composed of label + definition
            scored = [
                (
                    simple_score(query, e["label"], e["definition"]),
                    f"- {e['label']}: {e['definition']}",
                )
                for e in ref_entries
            ]
            # keep top-k with positive score
            scored = [s for s in scored if s[0] > 0]
            scored.sort(key=lambda x: x[0], reverse=True)
            lines = [s[1] for s in scored[: max(ref_top_k, 0)]]
        context = "\n".join(lines)
        if len(context) > ref_max_chars:
            context = context[: ref_max_chars] + "\n…"
        return context

    # --- Post-processing helper to enforce label casing in outputs ---
    def apply_label_casing(text: str, labels: Iterable[str]) -> str:
        """Replace case-insensitive matches of labels with their canonical casing.
        Longer labels are applied first to preserve multi-word phrases.
        """
        # deduplicate and sort by length desc
        sorted_labels = sorted({l for l in labels if l}, key=len, reverse=True)
        out = text
        for lbl in sorted_labels:
            try:
                pat = re.compile(r"\b" + re.escape(lbl) + r"\b", flags=re.IGNORECASE)
                out = pat.sub(lbl, out)
            except re.error:
                # If a label has odd regex chars despite escaping, skip gracefully
                continue
        return out

    def enrich(label: str, definition: str) -> str:
        start = time.time()
        ref_ctx = build_reference_context(label, definition)
        resp = chain.invoke({
            "label": label or "",
            "definition": definition or "",
            "reference_context": ref_ctx,
        })
        text = (resp.content or "").strip()
        # Optional post-processing to enforce capitalization of overlapping labels
        if enforce_label_casing and ref_ctx:
            candidate_labels = [
                line.split(":", 1)[0].lstrip("- ").strip()
                for line in ref_ctx.splitlines()
                if line.startswith("-") and ":" in line
            ]
            before = text
            text = apply_label_casing(text, candidate_labels)
            if before != text:
                logger.debug("Applied label casing normalization")
        elapsed = (time.time() - start) * 1000.0
        logger.debug(f"Enriched '{label}' in {elapsed:.0f} ms (ref={reference_mode}, ctx_len={len(ref_ctx)})")
        return text

    logger.info("Enriching definitions...")
    # Replace the definition column with enriched text, preserve schema
    df["definition"] = df.apply(lambda r: enrich(r["label"], r["definition"]), axis=1)

    # Ensure column order: iri,label,definition
    df = df[["iri", "label", "definition"]]

    logger.info(f"Writing enriched CSV to: {OUTPUT_FILE}")
    write_df_to_csv(df, OUTPUT_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    main()
