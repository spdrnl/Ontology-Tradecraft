import logging
from pathlib import Path
from typing import List, Any

import dotenv
from langchain_ollama import ChatOllama

from generate.io import _load_label_to_iri_map
from generate.post_processing import _extract_subclassof_line
from preprocessing.enriching import build_reference_context
from preprocessing.io import read_csv, write_df_to_csv, append_to_file, read_reference_entries
from common.settings import build_settings
from common.prompt_loading import load_markdown_prompt_templates, build_prompts, _inject_iris_into_context
from util.logger_config import config

import pandas as pd

logger = logging.getLogger(__name__)
config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_AXIOMS = DATA_ROOT / "candidate_parents.csv"


def _select_prompt_and_chain(elem_type: str, prompts, chains):
    t = (elem_type or "").strip().lower()
    key = t if t in {"class", "property"} else "class"
    return prompts.get(key), chains.get(key)


def main():
    logger.info("===================================================================")
    logger.info("Generating parent candidates (LLM via LangChain + Ollama)")
    logger.info("===================================================================")

    # Load settings and inputs
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)
    enriched_path = settings["output_file"]  # data/definitions_enriched.csv
    logger.info("Reading enriched definitions from: %s", enriched_path)
    df = read_csv(enriched_path)

    # LLM
    logger.info("Initializing LLM with model: %s", settings["model_name"])
    llm = ChatOllama(model=settings["model_name"], temperature=settings["temperature"])

    # Load prompt config file for candidate generation (separate class/property prompts)
    cand_cfg = settings.get("candidates_prompt_cfg_file")
    logger.info("Loading candidate prompt templates from: %s", cand_cfg)
    prompt_texts = load_markdown_prompt_templates(cand_cfg)
    prompts, chains = build_prompts(llm, prompt_texts)

    # Reference entries and label→IRI map
    label_to_iri = _load_label_to_iri_map(settings["bfo_cco_terms"])

    # Load BFO/CCO reference entries
    ref_entries = read_reference_entries(settings)
    ref_strings: List[str] = [f"{e['label']} — {e['definition']}" for e in ref_entries]

    # Build outputs
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    generated = 0
    df["choice"] = "None"
    for iteration in range(10):
        done = True
        out_rows = []
        logger.info("Building and querying reference context per row...")
        for _, r in df.iterrows():
            elem_iri = str(r.get("iri", "")).strip()
            elem_label = str(r.get("label", "")).strip()
            elem_type = str(r.get("type", "")).strip().lower() or "unknown"
            elem_def = str(r.get("definition", "")).strip()
            choice = str(r.get("choice", "")).strip()

            if r["choice"] == "None":
                # Build base reference context using existing retrieval logic
                ref_ctx = build_reference_context(elem_label, elem_def, elem_type, ref_entries, [], settings)

                    # Inject IRIs per option using label→IRI map
                ref_ctx_with_iris = _inject_iris_into_context(ref_ctx, label_to_iri, elem_label)

                vars = {
                    "label": elem_label,
                    "definition": elem_def,
                    "reference_context": ref_ctx_with_iris,
                }

                prompt_echo(chains, elem_iri, elem_label, elem_type, prompts, settings, vars)

                # Invoke LLM
                try:
                    _, chain_obj = _select_prompt_and_chain(elem_type, prompts, chains)
                    resp = chain_obj.invoke(vars)
                    raw = (getattr(resp, "content", "") or "").strip()
                except Exception as e:
                    logger.warning("LLM invocation failed for %s: %s", elem_label, e)
                    raw = ""

                choice = _extract_subclassof_line(raw)
                if choice == "None":
                    done = False
                    print(done, generated, elem_label, f"choice: [{choice}]", f"raw:[{raw}]")
                else:
                    generated += 1
                    print(done, generated, elem_label, f"choice: [{choice}]", f"raw:[{raw}]")

            out_rows.append({
                "iri": elem_iri,
                "label": elem_label,
                "type": elem_type,
                "choice": choice,
                "raw": raw,
            })
        out_df = pd.DataFrame(out_rows)
        df = out_df
        if done:
            break

    # Write CSV output

    logger.info("Writing candidate parents to: %s", GENERATED_AXIOMS)
    write_df_to_csv(df, GENERATED_AXIOMS)

    logger.info("Done generating candidates.")


def prompt_echo(chains: dict[Any, Any], elem_iri: str, elem_label: str, elem_type: str, prompts: dict[Any, Any],
                settings: dict, vars: dict[str, str]):
    # Echo prompt if enabled
    if settings.get("echo_prompts", False):
        try:
            prompt_tmpl, _ = _select_prompt_and_chain(elem_type, prompts, chains)
            msgs = prompt_tmpl.format_messages(**vars)
            rendered = "\n\n".join(
                f"[{getattr(m, 'type', getattr(m, 'role', ''))}]\n{getattr(m, 'content', '')}" for m in msgs)
            header = f"\n==== Candidate Parent Prompt for IRI: {elem_iri} | label: {elem_label} ===="
            out_text = f"{header}\n{rendered}\n==== End Prompt ====\n"
            logger.info(out_text)
            if settings.get("echo_prompts_file"):
                append_to_file(settings.get("echo_prompts_file"), out_text)
        except Exception as e:
            logger.warning("Failed to echo prompt for %s: %s", elem_label, e)


if __name__ == "__main__":
    main()
