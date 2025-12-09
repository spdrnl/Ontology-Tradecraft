import logging
import os
from pathlib import Path

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

def build_settings(PROJECT_ROOT, DATA_ROOT) -> dict:
  """Collect all runtime settings into a single dictionary.

  Values are derived from environment variables with sensible defaults
  and project-relative paths.
  """
  # File paths
  input_file = DATA_ROOT / "definitions.csv"
  bfo_cco_terms = DATA_ROOT / "bfo_cco_terms.csv"
  output_file = DATA_ROOT / "definitions_enriched.csv"

  # Environment-driven parameters
  reference_mode = os.getenv("REFERENCE_MODE", "retrieve").lower()  # off | full | retrieve | fuzzy
  ref_top_k = int(os.getenv("REF_TOP_K", "5"))
  ref_max_chars = int(os.getenv("REF_MAX_CHARS", "4000"))
  ref_fuzzy_scorer = os.getenv("REF_FUZZY_SCORER", "token_set_ratio").lower()
  ref_fuzzy_cutoff = int(os.getenv("REF_FUZZY_CUTOFF", "70"))  # 0–100
  enforce_label_casing = os.getenv("ENFORCE_LABEL_CASING", "false").lower() in {"1", "true", "yes", "on"}
  echo_prompts = os.getenv("ECHO_PROMPTS", "false").lower() in {"1", "true", "yes", "on"}
  echo_prompts_file = os.getenv("ECHO_PROMPTS_FILE", "").strip()
  prompt_cfg_file = Path(os.getenv("DEFINITIONS_PROMPT_CONFIG_FILE", "prompts/preprocessing_definitions_prompts.md").strip())
  reference_ontology = Path(os.getenv("REFERENCE_ONTOLOGY", PROJECT_ROOT / "src/CommonCoreOntologiesMerged.ttl"))
  model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
  temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
  candidates_prompt_cfg_file = Path(os.getenv("CANDIDATES_PROMPT_CONFIG_FILE", "prompts/generate_candidates_prompts.md").strip())

  # Vector DB / Embeddings (optional; used when REFERENCE_MODE=vector)
  vector_db_uri = os.getenv("VECTOR_DB_URI", str(DATA_ROOT / "milvus.db"))
  vector_collection_classes = os.getenv("VECTOR_COLLECTION_CLASSES", "ref_classes")
  vector_collection_properties = os.getenv("VECTOR_COLLECTION_PROPERTIES", "ref_properties")
  # Qwen/Qwen3-Embedding-8B
  # sentence-transformers/all-MiniLM-L6-v2
  # nvidia/llama-embed-nemotron-8b
  embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

  settings = {
    "project_root": PROJECT_ROOT,
    "data_root": DATA_ROOT,
    "input_file": input_file,
    "bfo_cco_terms": bfo_cco_terms,
    "output_file": output_file,
    "reference_mode": reference_mode,
    "ref_top_k": ref_top_k,
    "ref_max_chars": ref_max_chars,
    "ref_fuzzy_scorer": ref_fuzzy_scorer,
    "ref_fuzzy_cutoff": ref_fuzzy_cutoff,
    "enforce_label_casing": enforce_label_casing,
    "echo_prompts": echo_prompts,
    "echo_prompts_file": echo_prompts_file,
    "prompt_cfg_file": prompt_cfg_file,
    "candidates_prompt_cfg_file": candidates_prompt_cfg_file,
    "model_name": model_name,
    "temperature": temperature,
    "reference_ontology": reference_ontology,
    # Vector search
    "vector_db_uri": vector_db_uri,
    "vector_collection_classes": vector_collection_classes,
    "vector_collection_properties": vector_collection_properties,
    "embedding_model": embedding_model,
  }

  logger.info(
    "Settings: reference_mode=%s, ref_top_k=%s, ref_max_chars=%s, ref_fuzzy_scorer=%s, "
    "ref_fuzzy_cutoff=%s, enforce_label_casing=%s, echo_prompts=%s, prompt_cfg_file=%s, "
    "model=%s, temperature=%s",
    settings["reference_mode"], settings["ref_top_k"], settings["ref_max_chars"], settings["ref_fuzzy_scorer"],
    settings["ref_fuzzy_cutoff"], settings["enforce_label_casing"], settings["echo_prompts"],
    settings["prompt_cfg_file"], settings["model_name"], settings["temperature"],
  )

  return settings
