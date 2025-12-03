from pathlib import Path
import extract_definitions
import preprocess_definitions_llm
import logging
from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"

def main():
    logger.info("=======================================")
    logger.info("Running all scripts")
    logger.info("=======================================")
    logger.info("Extracting definitions...")
    extract_definitions.main()
    logger.info("Preprocessing definitions...")
    preprocess_definitions_llm.main()


if __name__ == "__main__":
    main()