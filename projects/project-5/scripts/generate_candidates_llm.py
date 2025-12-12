import argparse
import logging
import os
from pathlib import Path
from typing import List

import nltk
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from nltk.corpus import stopwords
from pydantic import BaseModel
from rdflib import Graph
from rdflib import URIRef

from common.io import load_entries, read_csv
from common.ontology_utils import get_turtle_snippet_by_uri_ref
from common.prompt_loading import load_markdown_prompt_templates
from common.settings import build_settings
from common.string_normalization import collect_labels, filter_labels, collect_words, filter_stopwords
from generate_candidates.subroutines import elk_check_axioms, parse_axiom, add_candidates_from_other_words, \
    add_candidates_from_labels
from util.logger_config import config

PROJECT_ROOT = Path(__file__).parent.parent
ROBOT_DIR = PROJECT_ROOT / "robot"
DATA_ROOT = PROJECT_ROOT / "data"
path = PROJECT_ROOT / Path("prompts/generate_candidates_prompts.md")

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

nltk.download('stopwords')


class LlmResponse(BaseModel):
    candidate_axioms: List[str]
    reasoning: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate candidate axioms for IEO using Ollama."
        )
    )
    p.add_argument(
        "--robot",
        default=None,
        help="Path to robot executable or robot.jar (if not provided, auto-detect)",
    )
    p.add_argument(
        "--max-mem",
        default=os.getenv("ROBOT_JAVA_MAX_MEM", "6g"),
        help="Max Java heap for ROBOT when using robot.jar (default: 6g)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    stop_words = set(stopwords.words('english'))

    # Get reference ontology
    reference_ontology = Graph()
    reference_ontology.parse(settings["reference_ontology"], format='turtle')

    # Read enriched definitions CSV
    logger.info(f"Reading definitions CSV from: {settings['enriched_definitions']}")
    df = read_csv(settings["enriched_definitions"])
    df.fillna("", inplace=True)

    # Get BFO and CCO reference terms
    csv_path: Path = settings["bfo_cco_terms"]
    if not csv_path.exists():
        logger.error("Reference terms CSV not found: %s", csv_path)
        return

    reference_terms_df = load_entries(csv_path)
    reference_terms_df.fillna("", inplace=True)
    if reference_terms_df.empty:
        logger.warning("No entries to index from %s", csv_path)
        return

    # Create mapping from class and property labels to IRIs
    labels_to_iri = {row['label']: {"iri": row["iri"], "type": row["type"]} for _, row in reference_terms_df.iterrows()}

    # Phrase difference
    phrase_differences_path: Path = settings["phrase_differences"]
    if not csv_path.exists():
        logger.error("Phrase difference CSV not found: %s", phrase_differences_path)
        return

    phrase_differences_df = pd.read_csv(phrase_differences_path, header=0)
    phrase_differences_df.fillna("", inplace=True)
    if reference_terms_df.empty:
        logger.warning("No entries to index from %s", phrase_differences_path)
        return

    # Create mapping from class and property labels to IRIs
    labels_to_phrase_difference = {row['label']: row["difference"] for _, row in phrase_differences_df.iterrows()}

    # BFO and CCO reference ontologies
    ttl_path: Path = settings["reference_ontology"]
    if not ttl_path.exists():
        logger.error("Reference ontology not found: %s", ttl_path)
        return

    reference_ontology = Graph()
    reference_ontology.parse(ttl_path.as_posix(), format='turtle')
    if len(reference_ontology.all_nodes()) == 0:
        logger.warning("No entries in reference ontology %s", ttl_path)
        return

    # Set-up LLM communication
    llm = ChatOllama(
        model=settings["model_name"],  # or "gemma3n:latest" or a specific tag
        temperature=settings["temperature"],
    )

    # Load prompt config file
    logger.info(f"Loading prompt templates from: {settings['prompt_cfg_file']}")
    prompt_texts = load_markdown_prompt_templates(path)

    # Perform the enrichment
    logger.info("Generating axioms...")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    settings["generated_root"].mkdir(parents=True, exist_ok=True)
    logger.info("Building and querying prompt per row...")
    df['status'] = 'PENDING'
    all_axioms = Graph()
    success = 0

    # Allow for retries
    for iter in range(10):
        out_rows = []
        for _, r in df.iterrows():
            iri = r["iri"]
            label = r["label"]
            status = r["status"]
            definition = r["definition"]
            type_name = r["type"]

            if type_name in {"property"}:
                logger.info(f"Skipping {iri} ({label}) of type {type_name}.")
                status = "OK"

            if status != 'OK':
                # Gather information about target class
                # definition = """
                # x is an 'Algorithm' if x is a Directive Information Content Entity that 'prescribes' some 'process' and contains a finite sequence of unambiguous instructions in order to achieve some Objective.
                # """.strip()
                # iri = "https://www.commoncoreontologies.org/ont00000653"
                # type_name = "class"
                # label = "Algorithm"
                owl_definition = get_turtle_snippet_by_uri_ref(URIRef(iri), type_name,
                                                               PROJECT_ROOT / "src/CommonCoreOntologiesMerged.ttl")

                # Create mapping candidates
                ref_top_k = 25  # settings.get("ref_top_k")
                property_candidates = []
                class_candidates = []

                # Collect labels and words
                work_definition = definition
                candidate_labels = collect_labels(work_definition, reference_terms_df["label"].tolist())
                work_definition = filter_labels(work_definition, candidate_labels)
                work_definition = work_definition + labels_to_phrase_difference[label]
                other_words = collect_words(work_definition)
                other_words = filter_stopwords(other_words, stop_words)

                # Collect directly recognized class and property labels
                labels = collect_labels(definition, reference_terms_df["label"].tolist())
                add_candidates_from_labels(labels, labels_to_iri, class_candidates, property_candidates)

                # Do a vector search for other candidates
                class_candidates, property_candidates = add_candidates_from_other_words(other_words, ref_top_k,
                                                                                        class_candidates,
                                                                                        property_candidates,
                                                                                        settings)

                # Create mapping sections
                properties_section = ""
                for candidate in set(property_candidates):
                    properties_section += f"'{candidate[0]}', {candidate[1]}\n"

                classes_section = ""
                for candidate in set(class_candidates):
                    classes_section += f"'{candidate[0]}', {candidate[1]}\n"

                # Read and prepare the prompt
                system_query = (prompt_texts[type_name]["system"]
                                .replace("{target_definition}", definition)
                                .replace("{target_uri_ref}", iri)
                                .replace("{owl_definition}", owl_definition)
                                .replace("{properties_section}", properties_section)
                                .replace("{classes_section}", classes_section)
                                )

                user_query = (prompt_texts[type_name]['user']
                              .replace("{target_definition}", definition)
                              .replace("{target_uri_ref}", iri)
                              .replace("{owl_definition}", owl_definition)
                              .replace("{properties_section}", properties_section)
                              .replace("{classes_section}", classes_section)
                              )

                print(user_query)

                ollama_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_query),
                        ("user", user_query),
                    ]
                )

                # Run the query
                print(f"Querying LLM for axiom candidates for {iri} ({label})...")
                print(ollama_prompt)
                try:
                    chain = ollama_prompt | llm | JsonOutputParser()
                    prompt_response = chain.invoke({})

                    # Parse LLM response
                    if "candidate_axioms" in prompt_response:
                        axiom_responses: List[str] = prompt_response['candidate_axioms']
                    else:
                        axiom_responses = []

                    # Accept empty responses
                    if len(axiom_responses) == 0:
                        success += 1
                        status = "OK"

                except Exception as e:
                    logger.warning("LLM response parsing failed for %s: %s", label, e)

                    # Do not accept malformed responses.
                    axiom_responses = []
                    status = "NOK"

                # Parse axioms
                for axiom in axiom_responses:
                    axiom_graph = parse_axiom(axiom)
                    if axiom_graph is not None:
                        elk_status = elk_check_axioms(reference_ontology, args, axiom_graph, ROBOT_DIR)
                        if elk_status != 0:
                            logger.info("Axiom failed ELK reasoner: %s")
                            logger.info("Axiom: %s", axiom)
                        else:
                            all_axioms += axiom_graph
                            status = "OK"

                if status == "OK":
                    success += 1

                print(f"Success: {success} / {len(df)}")

                if "reasoning" in prompt_response:
                    print('.\n'.join(prompt_response['reasoning'].split('.')))

                # Write results to file
                all_axioms.serialize(format="turtle", destination=settings["candidates_el"])

            out_rows.append({
                "iri": iri,
                "label": label,
                "type": type_name,
                "status": status,
                "definition": definition
            })

        # Check if all definitions have been processed successfully.
        df = pd.DataFrame(out_rows)
        if (df["status"] == "OK").all():
            break


if __name__ == "__main__":
    main()
