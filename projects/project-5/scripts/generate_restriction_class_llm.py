import logging
from typing import Any, List

import dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from common.io import read_csv, write_df_to_csv
from common.ontology_utils import (
    get_parent_property_by_label,
    get_parent_class_by_label, get_parent_by_label, get_turtle_snippet_by_label,
)
from common.settings import build_settings
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_ROOT / "phrase_differences.csv"

# Separate prompts for properties (verb phrases) and classes (noun phrases)
PROMPT_TEMPLATE_PROPERTY = (
    "Describe the difference between the first verb phrase 'X' and the second verb phrase 'Y' in plain\n"
    "English in a short phrase, use academic language. \n"
    "Explicitly state the two phrases with single quote, and start the output with 'The difference is:'."
    "Only output a short phrase, without explanations, considerations or other additional information.\n\n"
)

PROMPT_TEMPLATE_CLASS = (
    "Describe the difference between the first noun phrase 'X' and the second noun phrase 'Y' in plain\n"
    "English in a short phrase, use academic language. \n"
    "Explicitly state the two phrases with single quote, and start the output with 'The difference is:'."
    "Only output a short phrase, without explanations, considerations or other additional information.\n\n"
)


def _build_chain(llm: ChatOllama, template_text: str):
    tmpl = ChatPromptTemplate.from_messages([
        ("user", template_text),
    ])
    return tmpl, tmpl | llm


def main():
    logger.info("===================================================================")
    logger.info("Generating phrase differences (LLM via LangChain + Ollama)")
    logger.info("===================================================================")

    # Load settings
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    label = "Document Field Content"

    target_info = get_parent_by_label(label, "property", Path("src/ConsolidatedCCO.ttl"))
    turtle_snippet = get_turtle_snippet_by_label(label, "property", Path("src/ConsolidatedCCO.ttl"))

    # Read input CSV (definitions)
    input_csv = settings.get("input_file", DATA_ROOT / "definitions.csv")
    logger.info("Reading definitions from: %s", input_csv)
    df = read_csv(input_csv)

    # LLM
    logger.info("Initializing LLM with model: %s", settings["model_name"])
    llm = ChatOllama(model=settings["model_name"], temperature=settings["temperature"])
    # Build two chains: one for properties and one for classes
    prop_prompt_tmpl, prop_chain = _build_chain(llm, PROMPT_TEMPLATE_PROPERTY)
    class_prompt_tmpl, class_chain = _build_chain(llm, PROMPT_TEMPLATE_CLASS)

    # TTL path for parent lookup
    ttl_path = PROJECT_ROOT / "src" / "ConsolidatedCCO.ttl"
    if not ttl_path.exists():
        logger.warning("TTL file not found at %s; parent lookup will fail.", ttl_path)

    # Output rows
    df["status"] = "PENDING"
    df["difference"] = "None"
    df["raw"] = "None"
    df["parent_label"] = "None"
    for i in range(10):
        logger.info("Iteration %d", i)
        out_rows: List[dict[str, Any]] = []
        for _, r in df.iterrows():
            iri = str(r.get("iri", "") or "").strip()
            label = str(r.get("label", "") or "").strip()
            type_name = str(r.get("type", "") or "").strip().lower()
            status = str(r.get("status"))
            parent_label = str(r.get("parent_label"))
            difference = str(r.get("difference"))
            raw = str(r.get("raw"))

            if status == "OK":
                out_rows.append({
                    "iri": iri,
                    "label": label,
                    "type": type_name,
                    "parent_label": parent_label,
                    "difference": difference,
                    "status": status,
                    "raw": raw,
                })
                continue

            # Only process classes and properties; skip others
            if type_name not in {"property", "class"}:
                continue

            parent_label = None
            raw = ""

            try:
                if type_name == "property":
                    info = get_parent_property_by_label(label, ttl_path)
                else:
                    info = get_parent_class_by_label(label, ttl_path)
                parent_label = info.get("parent_label")
            except Exception as e:
                logger.warning("Parent lookup failed for %s: %s", label, e)
                status = "LOOKUP_FAIL"

            if not parent_label:
                status = "OK"
                difference = ""
            else:
                try:
                    # Choose appropriate prompt and chain based on element type
                    if type_name == "property":
                        tmpl = prop_prompt_tmpl
                        chain = prop_chain
                    else:
                        tmpl = class_prompt_tmpl
                        chain = class_chain

                    # Build prompt messages and inject labels for X and Y
                    msgs = tmpl.format_messages()
                    for m in msgs:
                        if hasattr(m, "content"):
                            m.content = m.content.replace("'X'", f"'{label}'").replace("'Y'", f"'{parent_label}'")
                    resp = llm.invoke(msgs)
                    raw = (getattr(resp, "content", "") or "").strip()
                    difference = raw.strip().strip('"')
                    if difference.startswith("The difference is:"):
                        status = "OK"
                        difference = difference[len("The difference is:"):].strip()
                    else:
                        status = "NOK"
                except Exception as e:
                    logger.warning("LLM invocation failed for %s: %s", label, e)
                    status = "NOK"
                    difference = ""

            logger.info(f"Label '{label}' has difference '{difference}' (status={status}).")
            out_rows.append({
                "iri": iri,
                "label": label,
                "type": type_name,
                "parent_label": parent_label,
                "difference": difference,
                "status": status,
                "raw": raw,
            })

        df = pd.DataFrame(out_rows)
        if (df["status"] == "OK").all():
            break

    logger.info("Writing phrase differences to: %s", OUTPUT_FILE)
    write_df_to_csv(df, OUTPUT_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    main()
