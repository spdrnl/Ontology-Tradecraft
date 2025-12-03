import logging
from pathlib import Path
import re

import pandas as pd
import rdflib

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
ONTOLOGIES_ROOT = PROJECT_ROOT / "src"

BFO_CORE_FILE = ONTOLOGIES_ROOT / "bfo-core.ttl"
CCO_MERGED_FILE = ONTOLOGIES_ROOT / "CommonCoreOntologiesMerged.ttl"
OUTPUT_FILE = DATA_ROOT / "bfo_cco_terms.csv"


def read_ttl(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return g


def extract_definitions(g):
    """Extract subject IRI, label, definition and entity type.

    Type is one of: 'class', 'property', or 'unknown'.
    """
    rows = []
    query = """
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?s ?label ?definition ?type
    WHERE {
      ?s skos:definition ?definition .
      OPTIONAL { ?s rdfs:label ?label . }
      {
        ?s a owl:Class .
        BIND("class" AS ?type)
      } UNION {
        ?s a owl:ObjectProperty .
        BIND("property" AS ?type)
      } UNION {
        ?s a owl:DatatypeProperty .
        BIND("property" AS ?type)
      }
    }
    """

    for row in g.query(query):
        iri = str(row[0])
        raw_label = None if row[1] is None else str(row[1])
        definition = str(row[2])
        ent_type = str(row[3]) if len(row) > 3 and row[3] is not None else "unknown"

        if raw_label is None:
            label = None
        elif ent_type == "property":
            #label = snake_case_label(raw_label)
            label = raw_label
        else:
            # default to class-style capitalization
            #label = capitalize_label(raw_label)
            label = raw_label

        rows.append([iri, label, definition, ent_type])
    return rows


def capitalize_label(label: str) -> str:
    if label is None: return None
    elif len(label) <= 2: return label
    else: return " ".join([label.capitalize() for label in label.split(" ")])

def snake_case_label(label: str) -> str:
    """Convert a label to snake_case (for property labels).

    - Lowercase
    - Convert camelCase/PascalCase boundaries to underscores
    - Replace any non-alphanumeric with underscores
    - Collapse multiple underscores and trim
    """
    if label is None:
        return None
    # Insert underscore between camelCase boundaries: "isAbout" -> "is_About"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", label)
    # Replace non-alphanumeric with underscores
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    # Lowercase
    s = s.lower()
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # Trim leading/trailing underscores
    s = s.strip("_")
    return s

def rows_to_df(rows):
    return pd.DataFrame(rows, columns=["iri", "label", "definition", "type"])

def write_df_to_csv(df, OUTPUT_FILE):
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    logger.info("======================================================")
    logger.info("Extracting labels and definitions from BFO and CCO")
    logger.info("======================================================")
    logger.info(f"Reading {BFO_CORE_FILE} file...")
    g = read_ttl(BFO_CORE_FILE)
    logger.info("Extracting labels and definitions...")
    bfo_rows = rows_to_df(extract_definitions(g))
    logger.info(f"Found {len(bfo_rows)} definitions.")
    g = read_ttl(CCO_MERGED_FILE)
    logger.info("Extracting labels and definitions...")
    cco_rows = rows_to_df(extract_definitions(g))
    logger.info(f"Found {len(cco_rows)} definitions.")

    # Merge and select columns
    df = pd.concat([bfo_rows, cco_rows])
    # Keep label/definition for downstream scripts, and include new 'type' column
    df = df[["label", "definition", "type"]]

    logger.info("Writing to CSV...")
    write_df_to_csv(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
