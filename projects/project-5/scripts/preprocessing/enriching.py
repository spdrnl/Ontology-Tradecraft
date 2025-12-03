import logging
import time
from typing import Any, List

from preprocessing.fuzzy_matching import fuzzy_top_k, get_scorer, simple_score
from preprocessing.io import format_messages_as_text, append_to_file
from preprocessing.output_sanitizing import sanitize_llm_output, _strip_wrapping_quotes
from preprocessing.post_processing import apply_label_casing, apply_property_snakecase
from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)


def enrich(elem_iri: str, elem_label: str, elem_type: str, elem_definition: str, chains, prompts, ref_entries,
           settings: dict) -> str:
  start = time.time()

  ref_type_by_label = {}
  full_reference_context = ""

  ref_strings: List[str] = [f"{e['label']} — {e['definition']}" for e in ref_entries]

  ref_ctx = build_reference_context(settings, ref_entries, full_reference_context, ref_strings, elem_label, elem_definition)
  vars = {
    "label": elem_label or "",
    "definition": elem_definition or "",
    "reference_context": ref_ctx,
  }

  # Choose prompt/chain based on element type
  t = elem_type.strip().lower()
  chain = chains.get("class" if t not in {"class", "property"} else t)
  prompt = prompts.get("class" if t not in {"class", "property"} else t)

  # If echoing is enabled, render the actual prompt with context
  echo_prompts = settings.get("echo_prompts", False)
  if echo_prompts:
    try:
      echo_prompt_to_user(elem_iri, elem_label, prompt, ref_ctx, ref_type_by_label, settings, vars)
    except Exception as e:
      logger.warning("Failed to render echo prompt for %s: %s", elem_iri, e)

  resp = chain.invoke(vars)
  raw_text = (resp.content or "").strip()
  text = sanitize_llm_output(_strip_wrapping_quotes, raw_text, fallback=elem_definition)
  if text != raw_text:
    logger.debug("Sanitized LLM output for %s (length %d -> %d)", elem_iri, len(raw_text), len(text))
  # Optional post-processing to enforce label formatting:
  # - Capitalize class labels to canonical casing
  # - Normalize property labels to snake_case and lowercase
  enforce_label_casing = settings.get("enforce_label_casing", False)
  if enforce_label_casing and ref_ctx:
    candidate_labels = [
      line.split(":", 1)[0].lstrip("- ").strip()
      for line in ref_ctx.splitlines()
      if line.startswith("-") and ":" in line
    ]
    class_labels = [l for l in candidate_labels if ref_type_by_label.get(l) == "class"]
    property_labels = [l for l in candidate_labels if ref_type_by_label.get(l) == "property"]

    before = text
    if class_labels:
      text = apply_label_casing(text, class_labels)
    if property_labels:
      text = apply_property_snakecase(text, property_labels)
    if before != text:
      logger.debug("Applied label normalization (class casing/property snake_case)")
  elapsed = (time.time() - start) * 1000.0
  reference_mode = settings.get("reference_mode")
  logger.debug(f"Enriched '{elem_label}' in {elapsed:.0f} ms (ref={reference_mode}, ctx_len={len(ref_ctx)})")
  return text


def echo_prompt_to_user(elem_iri: str, elem_label: str, prompt, ref_ctx: str, ref_type_by_label: dict[Any, Any],
                        settings: dict, vars: dict[str, str]):
  # Also print the reference context explicitly to avoid any confusion
  # about where it appears inside the rendered prompt messages.
  reference_mode = settings.get("reference_mode")
  ref_top_k = settings.get("ref_top_k")
  ref_max_chars = settings.get("ref_max_chars")
  ref_fuzzy_scorer = settings.get("ref_fuzzy_scorer")
  ref_fuzzy_cutoff = settings.get("ref_fuzzy_cutoff")
  ref_meta = (
    f"reference_mode={reference_mode}; "
    f"top_k={ref_top_k}; max_chars={ref_max_chars}"
  )
  if reference_mode == "fuzzy":
    ref_meta += f"; scorer={ref_fuzzy_scorer}; cutoff={ref_fuzzy_cutoff}"
  # Add counts of context types, if any
  if ref_ctx:
    c_count, p_count, u_count = classify_context_counts(ref_type_by_label, ref_ctx)
    ref_meta += f"; ctx_counts=class:{c_count}|property:{p_count}|unknown:{u_count}"

  msgs = prompt.format_messages(**vars)
  rendered = format_messages_as_text(msgs)
  header = f"\n==== Prompt for IRI: {elem_iri} | label: {elem_label} ===="
  prompt_header = "---- Rendered Messages (system + user) ----"
  footer = "==== End Prompt ====\n"
  out_text = (
    f"{header}\n"
    f"{prompt_header}\n{rendered}\n{footer}"
  )
  logger.info(out_text)
  echo_prompts_file = settings.get("echo_prompts_file")
  if echo_prompts_file:
    append_to_file(echo_prompts_file, out_text)


def classify_context_counts(ref_type_by_label, ref_ctx_text: str):
  """Count class/property/unknown labels present in the effective context."""
  labels = [
    line.split(":", 1)[0].lstrip("- ").strip()
    for line in ref_ctx_text.splitlines()
    if line.startswith("-") and ":" in line
  ]
  n_class = sum(1 for l in labels if ref_type_by_label.get(l) == "class")
  n_prop = sum(1 for l in labels if ref_type_by_label.get(l) == "property")
  n_unk = sum(1 for l in labels if ref_type_by_label.get(l) not in {"class", "property"})
  return n_class, n_prop, n_unk


def build_reference_context(settings: dict, ref_entries, full_reference_context, ref_strings,
                            term_label: str, term_definition: str) -> str:
  reference_mode = settings.get("reference_mode")
  ref_fuzzy_scorer = settings.get("ref_fuzzy_scorer")
  ref_top_k = settings.get("ref_top_k")
  ref_fuzzy_cutoff = settings.get("ref_fuzzy_cutoff")
  ref_max_chars = settings.get("ref_max_chars")

  if reference_mode == "off" or not ref_entries:
    return ""
  if reference_mode == "full":
    return full_reference_context
  # Build query from current row
  query = f"{term_label or ''} {term_definition or ''}"
  if reference_mode == "fuzzy":
    # fuzzy mode: use RapidFuzz to select top-K relevant lines
    lines = fuzzy_top_k(ref_entries, ref_strings, get_scorer, ref_fuzzy_scorer, query, ref_top_k, ref_fuzzy_cutoff)
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
