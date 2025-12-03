import re
from typing import List


def sanitize_llm_output(_strip_wrapping_quotes, text: str, fallback: str) -> str:
  """Reduce LLM output to a single, clean definition sentence.

  Removes any echoed instructions, reference lists, and metadata like
  "Improved definition:", "Reference BFO/CCO...", "Term:", bullets, etc.
  If the result is empty or too short, returns the provided fallback.
  """
  if not text:
    return (fallback or "").strip()

  # Normalize newlines
  t = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

  # Remove common leading markers/prefixes
  prefixes = [
    "improved definition:",
    "definition:",
    "improved:",
    "revised definition:",
  ]
  lower = t.lower().lstrip()
  for p in prefixes:
    if lower.startswith(p):
      t = t[len(p):].lstrip() if len(t) > len(p) else ""
      break

  # Drop lines that are clearly part of the prompt/instructions/references
  drop_starts = (
    "reference bfo/cco",
    "term:",
    "existing definition:",
    "improved definition:",
    "improved:",
    "note:",
  )
  kept_lines: List[str] = []
  for line in t.split("\n"):
    s = line.strip()
    if not s:
      # Stop at first blank separating paragraphs; keep what we have
      if kept_lines:
        break
      else:
        continue
    ls = s.lower()
    if ls.startswith(drop_starts) or ls.startswith("-") or ls.startswith("*"):
      # skip bullet/reference/instruction lines
      continue
    kept_lines.append(s)

  t = " ".join(kept_lines) if kept_lines else ""
  t = _strip_wrapping_quotes(t)

  # Keep only the first complete sentence (heuristic)
  # Look for . ! ? followed by space or end
  m = re.search(r"([^.?!]+[.?!])", t)
  if m:
    t = m.group(1)

  # Final cleanup: collapse whitespace
  t = re.sub(r"\s+", " ", t).strip().strip('"').strip("'")

  # Ensure not empty or trivial; otherwise fallback
  if len(t) < 5:
    t = (fallback or "").strip()

  return t


def _strip_wrapping_quotes(s: str) -> str:
  if not s:
    return s
  s = s.strip()
  # remove matching leading/trailing quotes/backticks
  if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")) or (
      s.startswith("```") and s.endswith("```")
  ):
    return s[1:-1] if len(s) >= 2 and s[0] != "`" else s.strip("`")
  return s
