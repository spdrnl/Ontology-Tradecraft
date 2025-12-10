import argparse
import logging
import os
from pathlib import Path
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from pydantic import BaseModel
from rdflib import Graph
from rdflib import URIRef

from common.llm_communication import call_llm_over_http
from common.ontology_utils import get_turtle_snippet_by_uri_ref
from common.prompt_loading import load_markdown_prompt_templates
from common.settings import build_settings
from common.string_normalization import collect_labels, filter_labels, collect_words, filter_stopwords
from generate_candidates.subroutines import elk_check_axioms, create_axiom_graph, add_candidates_from_other_words, \
    add_candidates_from_labels
from util.logger_config import config

PROJECT_ROOT = Path(__file__).parent.parent
ROBOT_DIR = PROJECT_ROOT / "robot"
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
path = PROJECT_ROOT / Path("prompts/generate_candidates_prompts.md")

logger = logging.getLogger(__name__)
config(logger)

nltk.download('stopwords')


class LlmResponse(BaseModel):
    candidate_axioms: List[str]
    reasoning: str


def _load_entries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize expected columns
    for col in ("iri", "label", "definition", "type"):
        if col not in df.columns:
            df[col] = ""

    # Clean
    df["label"] = df["label"].astype(str).str.strip()
    df["definition"] = df["definition"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower().replace({"": "unknown"})
    df["iri"] = df["iri"].astype(str).str.strip()

    # Filter out empty labels/definitions
    df = df[(df["label"] != "") & (df["definition"] != "")]
    return df


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

    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)

    stop_words = set(stopwords.words('english'))

    # Get BFO and CCO reference terms
    csv_path: Path = settings["bfo_cco_terms"]
    if not csv_path.exists():
        logger.error("Reference terms CSV not found: %s", csv_path)
        return

    reference_terms_df = _load_entries(csv_path)
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

    # Load prompt config file
    logger.info(f"Loading prompt templates from: {settings['prompt_cfg_file']}")
    prompt_texts = load_markdown_prompt_templates(path)

    # Gather information about target class
    target_definition = """
    x is an 'Algorithm' if x is a Directive Information Content Entity that 'prescribes' some 'process' and contains a finite sequence of unambiguous instructions in order to achieve some Objective.
    """.strip()
    target_uri_ref = "https://www.commoncoreontologies.org/ont00000653"
    target_type = "class"
    target_label = "Algorithm"
    target_owl_definition = get_turtle_snippet_by_uri_ref(URIRef(target_uri_ref), target_type,
                                                          PROJECT_ROOT / "src/CommonCoreOntologiesMerged.ttl")

    # Create mapping candidates
    ref_top_k = 25  # settings.get("ref_top_k")
    property_candidates = []
    class_candidates = []

    # Collect labels and words
    work_definition = target_definition
    candidate_labels = collect_labels(work_definition, reference_terms_df["label"].tolist())
    work_definition = filter_labels(work_definition, candidate_labels)
    work_definition = work_definition + labels_to_phrase_difference[target_label]
    other_words = collect_words(work_definition)
    other_words = filter_stopwords(other_words, stop_words)

    # Collect directly recognized class and property labels
    labels = collect_labels(target_definition, reference_terms_df["label"].tolist())
    add_candidates_from_labels(labels, labels_to_iri, class_candidates, property_candidates)

    # Do a vector search for other candidates
    class_candidates, property_candidates = add_candidates_from_other_words(other_words, ref_top_k, class_candidates,
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
    prompt = f"{prompt_texts[target_type]["system"]}/n{prompt_texts[target_type]['user']}"

    query = (prompt
             .replace("{target_definition}", target_definition)
             .replace("{target_uri_ref}", target_uri_ref)
             .replace("{owl_definition}", target_owl_definition)
             .replace("{properties_section}", properties_section)
             .replace("{classes_section}", classes_section)
             )

    print(query)

    url = "http://localhost:11434/api/generate"
    json_response = call_llm_over_http(url, query)
    axiom_response = json_response['response']

    all_axioms = Graph()
    if "candidate_axioms" in axiom_response:
        for axiom in axiom_response['candidate_axioms']:
            axiom_graph = create_axiom_graph(axiom)
            status = elk_check_axioms(all_axioms, args, axiom_graph, ROBOT_DIR)
            if status != 0:
                logger.info("Axiom failed ELK reasoner: %s")
                logger.info("Axiom: %s", axiom)
            else:
                all_axioms += axiom_graph
    if "reasoning" in axiom_response:
        print(axiom_response['reasoning'])
    all_axioms.serialize(format="turtle", destination=GENERATED_ROOT / "output.ttl")


if __name__ == "__main__":
    main()
