from __future__ import annotations

import json as _json
from typing import Any


def extract_json(text: str | Any) -> Any:
    try:
        obj = _json.loads(text)
    except Exception:
        # Robustly handle Markdown code fences and stray text around JSON
        cleaned = str(text).strip()
        # If response is fenced as ```json ... ``` (or ``` ... ```), remove fences
        if cleaned.startswith("```"):
            # Drop the opening fence line (which may include a language tag like 'json')
            nl = cleaned.find("\n")
            if nl != -1:
                cleaned = cleaned[nl + 1:]
            # Remove trailing closing fence if present
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        # Try direct parse again after fence removal
        try:
            obj = _json.loads(cleaned)
        except Exception:
            # Fallback: extract the JSON object substring between the first '{' and last '}'
            l = cleaned.find("{")
            r = cleaned.rfind("}")
            if l != -1 and r != -1 and r > l:
                snippet = cleaned[l: r + 1]
                obj = _json.loads(snippet)
            else:
                # Re-raise to be handled by outer exception block
                raise
    return obj
