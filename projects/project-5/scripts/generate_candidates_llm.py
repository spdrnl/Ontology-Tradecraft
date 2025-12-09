import argparse
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import nltk
import pandas as pd
import requests
from nltk.corpus import stopwords
from pydantic import BaseModel
from rdflib import URIRef

from rdflib import Graph

from common.json_extraction import extract_json
from common.ontology_utils import get_turtle_snippet_by_uri_ref
from common.robot import detect_robot, build_elk_robot_command, run
from common.settings import build_settings
from common.string_normalization import collect_labels, filter_labels, collect_words, filter_stopwords
from common.vectorization import vector_top_k
from util.logger_config import config

PROJECT_ROOT = Path(__file__).parent.parent
ROBOT_DIR = PROJECT_ROOT / "robot"
DATA_ROOT = PROJECT_ROOT / "data"
path = PROJECT_ROOT / Path("prompts/generate_candidates_prompts.md")

logger = logging.getLogger(__name__)
config(logger)

nltk.download('stopwords')

prefixes = """
@prefix cco: <https://www.commoncoreontologies.org/> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix bfo: <http://purl.obolibrary.org/obo/> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xml: <http://www.w3.org/XML/1998/namespace> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
""".strip()

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

    # Create mapping from class and property labels to IRIs
    labels_to_iri = {row['label']: {"iri": row["iri"], "type": row["type"]} for _, row in reference_terms_df.iterrows()}

    # Gather information about target class
    target_definition = """
    x is an 'Algorithm' if x is a Directive Information Content Entity that 'prescribes' some 'process' and contains a finite sequence of unambiguous instructions in order to achieve some Objective.
    """.strip()
    target_uri_ref = "https://www.commoncoreontologies.org/ont00000653"
    owl_definition = get_turtle_snippet_by_uri_ref(URIRef(target_uri_ref), "class",
                                                   PROJECT_ROOT / "src/CommonCoreOntologiesMerged.ttl")

    # Collect labels and words
    work_definition = target_definition
    candidate_labels = collect_labels(work_definition, reference_terms_df["label"].tolist())
    work_definition = filter_labels(work_definition, candidate_labels)
    other_words = collect_words(work_definition)
    other_words = filter_stopwords(other_words, stop_words)

    # Create mapping candidates
    property_candidates = []
    class_candidates = []

    # Collect directly recognized class and property labels
    labels = collect_labels(target_definition, reference_terms_df["label"].tolist())
    for label in labels:
        if label in labels_to_iri:
            if labels_to_iri[label]["type"] == "class":
                class_candidates.append((label, labels_to_iri[label]["iri"]))
            elif labels_to_iri[label]["type"] == "property":
                property_candidates.append((label, labels_to_iri[label]["iri"]))

    # Do a vector search for other candidates
    property_working_set = []
    class_working_set = []

    ref_top_k = 15  # settings.get("ref_top_k")

    for word in other_words:
        property_vector_suggestions = vector_top_k(word, "property", ref_top_k, settings)
        for suggestion in property_vector_suggestions:
            property_working_set.append(suggestion)

        class_vector_suggestions = vector_top_k(word, "class", ref_top_k, settings)
        for suggestion in class_vector_suggestions:
            class_working_set.append(suggestion)

    property_working_set.sort(key=lambda x: x["distance"], reverse=True)
    property_working_set = property_working_set[:ref_top_k]

    class_working_set.sort(key=lambda x: x["distance"], reverse=True)
    class_working_set = class_working_set[:ref_top_k]

    for suggestion in property_working_set:
        property_candidates.append((suggestion["label"], suggestion["iri"]))

    for suggestion in class_working_set:
        class_candidates.append((suggestion["label"], suggestion["iri"]))

    property_candidates = property_candidates[:ref_top_k]
    class_candidates = class_candidates[:ref_top_k]

    # Create mapping sections
    properties_section = ""
    for candidate in set(property_candidates):
        properties_section += f"{candidate[0]}, {candidate[1]}\n"

    classes_section = ""
    for candidate in set(class_candidates):
        classes_section += f"{candidate[0]}, {candidate[1]}\n"

    # Read and prepare the prompt
    prompt = path.read_text()
    query = (prompt
             .replace("{target_definition}", target_definition)
             .replace("{target_uri_ref}", target_uri_ref)
             .replace("{owl_definition}", owl_definition)
             .replace("{properties_section}", properties_section)
             .replace("{classes_section}", classes_section)
             )

    print(query)

    url = "http://localhost:11434/api/generate"

    data = {
        "model": "gemma3n",
        "prompt": query,
        "stream": False
    }

    response = requests.post(url, json=data)
    print(response.text)
    raw_reponse = json.loads(response.text)
    axiom_response = extract_json(raw_reponse['response'])
    all_axioms = Graph()
    if "candidate_axioms" in axiom_response:
        for axiom in axiom_response['candidate_axioms']:
            turtle = prefixes + "\n\n" +axiom
            print(turtle)
            axiom_graph = Graph()
            axiom_graph.parse(io.StringIO(turtle), format="turtle")
            test_graph = all_axioms + axiom_graph
            robot_cmd = detect_robot(args.robot, ROBOT_DIR)
            with tempfile.NamedTemporaryFile(suffix=".ttl") as in_path:
                with tempfile.NamedTemporaryFile(suffix=".ttl") as out_path:
                    test_graph.serialize(format="turtle", destination=in_path.name)
                    cmd = build_elk_robot_command(Path(in_path.name), Path(out_path.name), robot_cmd, args.max_mem)
                    if run(cmd) != 0:
                        logger.info("Axiom failed ELK reasoner: %s")
                        logger.info("Axiom: %s", axiom)
                    else:
                        all_axioms += axiom_graph
    if "reasoning" in axiom_response:
        print(axiom_response['reasoning'])
    all_axioms.serialize(format="turtle", destination=PROJECT_ROOT / "generated"/ "output.ttl")


if __name__ == "__main__":
    main()
