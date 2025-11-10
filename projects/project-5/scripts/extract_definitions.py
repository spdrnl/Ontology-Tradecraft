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

INPUT_FILE = SRC_ROOT / "InformationEntityOntology.ttl"
OUTPUT_FILE = DATA_ROOT / "definitions.csv"


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
        rows.append([str(row[0]), str(row[1]), str(row[2])])
    return rows


def rows_to_df(rows):
    return pd.DataFrame(rows, columns=["iri", "label", "definition"])


def write_df_to_csv(df, OUTPUT_FILE):
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    logger.info("=======================================")
    logger.info("Extracting definitions from TTL file")
    logger.info("=======================================")
    logger.info("Reading TTL file...")
    g = read_ttl(INPUT_FILE)
    logger.info("Extracting definitions...")
    rows = extract_definitions(g)
    logger.info(f"Found {len(rows)} definitions.")
    logger.info("Writing to CSV...")
    df = rows_to_df(rows)
    write_df_to_csv(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
