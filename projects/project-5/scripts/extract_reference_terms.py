import logging
from pathlib import Path

import pandas as pd
import rdflib

from logger_config import config

logger = logging.getLogger(__name__)
config(logger)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"

BFO_CORE_FILE = SRC_ROOT / "bfo-core.ttl"
CCO_MERGED_FILE = SRC_ROOT / "CommonCoreOntologiesMerged.ttl"
OUTPUT_FILE = DATA_ROOT / "bfo_cco_terms.csv"


def read_ttl(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return g


def extract_definitions(g):
    rows = []
    query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?s ?label ?definition {
        ?s skos:definition ?definition .
        OPTIONAL { 
            ?s rdfs:label ?label .
        }
    }
    """
    rows = []
    for row in g.query(query):
        rows.append([str(row[0]), capitalize_label(str(row[1])), str(row[2])])
    return rows


def capitalize_label(label: str) -> str:
    if label is None: return None
    elif len(label) <= 2: return label
    else: return " ".join([label.capitalize() for label in label.split(" ")])

def rows_to_df(rows):
    return pd.DataFrame(rows, columns=["iri", "label", "definition"])

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
    logger.info(f"Found {len(bfo_rows)} definitions.")

    # Merge and select columns
    df = pd.concat([bfo_rows, cco_rows])
    df = df[["label", "definition"]]

    logger.info("Writing to CSV...")
    write_df_to_csv(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
