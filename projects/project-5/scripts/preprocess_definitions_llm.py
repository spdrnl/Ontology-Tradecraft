import logging
from pathlib import Path

import dotenv
from langchain_ollama import ChatOllama

from preprocessing.enriching import enrich
from preprocessing.io import read_csv, write_df_to_csv, read_reference_entries
from preprocessing.prompt_loading import load_markdown_prompt_templates, build_prompts
from preprocessing.settings import build_settings
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

  # Perform the enrichment
  logger.info("Enriching definitions...")
  df["definition"] = df.apply(
    lambda r: enrich(r["iri"],
                     r["label"],
                     r["type"],
                     r["definition"],
                     chains,
                     prompts,
                     ref_entries,
                     settings),
    axis=1,
  )

  # Write result
  df = df[["iri", "label", "type", "definition"]]
  logger.info(f"Writing enriched CSV to: {settings['output_file']}")
  write_df_to_csv(df, settings["output_file"])
  logger.info("Done.")


if __name__ == "__main__":
  main()
