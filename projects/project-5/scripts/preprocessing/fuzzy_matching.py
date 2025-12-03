from typing import List

from rapidfuzz import process, fuzz

# --- RapidFuzz helpers (used when reference_mode == "fuzzy") ---
SCORERS = {
  "token_set_ratio": fuzz.token_set_ratio,  # robust to extra/missing words
  "token_sort_ratio": fuzz.token_sort_ratio,  # robust to word order changes
  "wratio": fuzz.WRatio,  # a smart overall scorer
}

def fuzzy_top_k(ref_entries, ref_strings, get_scorer, ref_fuzzy_scorer, query: str, top_k: int,
                cutoff: int) -> List[str]:
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


def get_scorer(name: str):
  return SCORERS.get(name, fuzz.token_set_ratio)