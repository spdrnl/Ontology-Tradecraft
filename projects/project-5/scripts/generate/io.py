import logging
from pathlib import Path
from typing import Dict

from preprocessing.io import read_csv
from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

def _load_label_to_iri_map(bfo_cco_terms_csv: Path) -> Dict[str, str]:
    try:
        df = read_csv(bfo_cco_terms_csv)
    except Exception as e:
        logger.warning("Failed to read %s for label→IRI map: %s", bfo_cco_terms_csv, e)
        return {}
    m = {}
    for _, row in df.iterrows():
        lbl = str(row.get("label", "")).strip()
        iri = str(row.get("iri", "")).strip()
        if lbl and iri:
            m[lbl] = iri
    return m
