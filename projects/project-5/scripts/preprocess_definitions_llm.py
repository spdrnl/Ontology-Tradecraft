import logging
from pathlib import Path

import dotenv
import pandas as pd
from langchain_ollama import ChatOllama

from common.prompt_loading import load_markdown_prompt_templates, build_prompts
from common.settings import build_settings
from preprocessing.enriching import enrich, normalize_prefix
from preprocessing.io import read_csv, write_df_to_csv, read_reference_entries
from preprocessing.string_normalization import remove_snake_case, apply_label_casing, apply_single_quotes, contains, \
    replace_x_and_y
from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def main():
    logger.info("===================================================================")
    logger.info("Preprocessing definitions (LLM enrichment via LangChain + Ollama)")
    logger.info("===================================================================")

    # Load settings
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    # Read input definitions CSV
    logger.info(f"Reading definitions CSV from: {settings['input_file']}")
    df = read_csv(settings["input_file"])

    # Configure LLM (Ollama must be running: `ollama serve` and model pulled)
    logger.info(f"Initializing LLM with model: {settings['model_name']}")
    llm = ChatOllama(model=settings["model_name"], temperature=settings["temperature"])

    # Load prompt config file
    logger.info(f"Loading prompt templates from: {settings['prompt_cfg_file']}")
    prompt_texts = load_markdown_prompt_templates(settings["prompt_cfg_file"])
    prompts, chains = build_prompts(llm, prompt_texts)

    # Load BFO/CCO reference entries
    ref_entries = read_reference_entries(settings)
    ref_labels = [element['label'] for element in ref_entries]

    # Perform the enrichment
    logger.info("Enriching definitions...")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Building and querying reference context per row...")
    df['status'] = 'None'
    success = 0
    for iter in range(10):
        out_rows = []
        for _, r in df.iterrows():
            iri = r["iri"]
            label = r["label"]
            status = r["status"]
            definition = r["definition"]
            type_name = r["type"]

            if status == "OK":
                improved_definition = r['definition']
            else:
                definition = remove_snake_case(definition)
                definition = apply_label_casing(definition, ref_labels)
                definition = apply_single_quotes(definition, ref_labels)
                definition = replace_x_and_y(definition)
                if not contains(definition, label):
                    if type_name == "class":
                        definition = f"individual i is a '{label}' iff individual i is a {definition}"
                    if type_name == "property":
                        definition = f"individual i '{label}' individual j iff {label} is a {definition}"

                improved_definition = enrich(iri,
                                             label,
                                             type_name,
                                             definition,
                                             chains,
                                             prompts,
                                             ref_entries,
                                             settings)

                improved_definition = improved_definition.replace("if and only if", "iff")
                if (('individual i' not in improved_definition)
                    or (r["type"] == "property" and "individual j" not in improved_definition)
                    or ('class D' in improved_definition)
                    or ('class C' in improved_definition)
                    or ('property Z' in improved_definition)
                    or ('iff' not in improved_definition)):
                    status = "NOK"
                    print(f"[{r['definition']}]")
                    print(f"[{improved_definition}]")
                    improved_definition = r["definition"]
                else:
                    improved_definition = remove_snake_case(improved_definition)
                    improved_definition = apply_label_casing(improved_definition, ref_labels)
                    improved_definition = apply_single_quotes(improved_definition, ref_labels)
                    improved_definition = normalize_prefix(improved_definition, type_name, label)
                    success += 1
                    print(f"Success {success} out of {len(df)} rows.")
                    print(f"[{definition}]")
                    print(f"[{improved_definition}]")
                    status = "OK"

            out_rows.append({
                "iri": iri,
                "label": label,
                "type": r["type"],
                "status": status,
                "definition": improved_definition
            })

        out_df = pd.DataFrame(out_rows)
        out_df = out_df[["iri", "label", "type", "status", "definition"]]
        write_df_to_csv(out_df, settings["output_file"])
        df = out_df
        if (df["status"] == "OK").all():
            break

    # Write result
    out_df = out_df[["iri", "label", "type", "status", "definition"]]
    logger.info(f"Writing enriched CSV to: {settings['output_file']}")
    write_df_to_csv(out_df, settings["output_file"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
