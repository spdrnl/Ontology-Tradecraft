import logging
from pathlib import Path
import dotenv
import os
import pandas as pd

from logger_config import config

logger = logging.getLogger(__name__)
config(logger)

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"

INPUT_FILE = DATA_ROOT / "definitions.csv"
OUTPUT_FILE = DATA_ROOT / "definitions_enriched.csv"


def read_csv(path):
    return pd.read_csv(path, header=0)

def write_df_to_csv(df, OUTPUT_FILE):
    df.to_csv(OUTPUT_FILE, index=False)

def main():
    logger.info("=======================================")
    logger.info("Preprocessing definitions")
    logger.info("=======================================")
    logger.info("Reading definitions CSV...")
    df = read_csv(INPUT_FILE)
    logger.info("Enriching definitions...")
    print(os.environ["API_KEY"])
    logger.info("Writing to CSV...")
    write_df_to_csv(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
