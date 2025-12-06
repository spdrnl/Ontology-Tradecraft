import re


def _extract_subclassof_line(text: str) -> str:
    if not text:
        return ""
    # Try to find an IRI inside angle brackets
    m = re.search(r"\[<?(http.*)>?\]", text)
    if not m or m.group(1) == "" :
        return "None"
    iri = m.group(1).strip()
    return f"<{iri}>"
