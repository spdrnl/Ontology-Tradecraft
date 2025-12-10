import logging
from pathlib import Path

import extract_definitions
import extract_reference_terms
import filter_candidates_hybrid
import generate_phrase_differences_llm
import merge_cco_with_ieo_robot
import preprocess_definitions_llm
import robot_elk
import robot_merge
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

    logger.info("Merging CCO with IEO...")
    merge_cco_with_ieo_robot.main()

    logger.info("Extracting reference terms...")
    extract_reference_terms.main()

    logger.info("Extracting definitions...")
    extract_definitions.main()

    logger.info("Generating phrase differences...")
    generate_phrase_differences_llm.main()

    logger.info("Preprocessing definitions...")
    preprocess_definitions_llm.main()

    logger.info("Filtering candidates ...")
    filter_candidates_hybrid.main()

    logger.info("Merging candidates ...")
    robot_merge.main()

    logger.info("Checking result with ELK ...")
    robot_elk.main()


if __name__ == "__main__":
    main()
