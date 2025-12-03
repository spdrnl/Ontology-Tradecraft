import re
from typing import Iterable


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


def apply_property_snakecase(text: str, snake_labels: Iterable[str]) -> str:
  """Normalize occurrences of property labels to snake_case and lowercase.

  For each snake_case label like "is_about", we will also normalize variants
  appearing with spaces or hyphens (e.g., "is about", "is-about") and case
  differences to the canonical snake_case form.
  """
  out = text
  for lbl in sorted({l for l in snake_labels if l}, key=len, reverse=True):
    try:
      parts = [re.escape(p) for p in lbl.split("_") if p]
      if not parts:
        continue
      # Pattern that matches the label if written with spaces, hyphens, or underscores between parts
      sep = r"[ _-]+"
      pattern = r"\b" + sep.join(parts) + r"\b"
      out = re.sub(pattern, lbl, out, flags=re.IGNORECASE)
      # Also normalize already-underscore form but wrong casing to lowercase canonical
      u_pattern = r"\b" + re.escape(lbl) + r"\b"
      out = re.sub(u_pattern, lbl, out, flags=re.IGNORECASE)
    except re.error:
      continue
  return out