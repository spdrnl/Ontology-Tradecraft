import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)


PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"


DECLARATIONS_CONSTRUCT = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

CONSTRUCT {
  ?s rdf:type ?t .
}
WHERE {
  VALUES ?t {
    owl:Class
    owl:ObjectProperty
    owl:DatatypeProperty
    owl:AnnotationProperty
    rdf:Property
    owl:NamedIndividual
  }
  ?s rdf:type ?t .
  FILTER(isIRI(?s))
}
"""


def _resolve_robot_command() -> list[str]:
    """Return the command list to invoke ROBOT.

    Preference order:
    1) `robot` on PATH
    2) `java -jar <PROJECT_ROOT>/robot/robot.jar` if present
    """
    robot_path = shutil.which("robot")
    if robot_path:
        return [robot_path]
    jar_candidate = PROJECT_ROOT / "robot" / "robot.jar"
    if jar_candidate.exists():
        java_path = shutil.which("java")
        if not java_path:
            raise RuntimeError("Java not found in PATH and robot binary not available; cannot run ROBOT.")
        return [java_path, "-jar", str(jar_candidate)]
    raise RuntimeError(
        "ROBOT executable not found in PATH and robot/robot.jar is missing. "
        "Please install ROBOT or place robot.jar under the project's 'robot' directory."
    )


def run_robot(args: list[str], cwd: Path | None = None):
    cmd = _resolve_robot_command() + args
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        logger.error("ROBOT command failed (exit %s)\nSTDOUT:\n%s\nSTDERR:\n%s", proc.returncode, proc.stdout, proc.stderr)
        raise RuntimeError(f"ROBOT failed with exit code {proc.returncode}")
    if proc.stdout.strip():
        logger.debug("ROBOT STDOUT:\n%s", proc.stdout)
    if proc.stderr.strip():
        # ROBOT often logs to stderr with INFO level; keep as debug to avoid noise
        logger.debug("ROBOT STDERR:\n%s", proc.stderr)


def merge_with_declarations_only(cco_path: Path, ieo_path: Path, out_path: Path):
    if not cco_path.exists():
        raise FileNotFoundError(f"CCO file not found: {cco_path}")
    if not ieo_path.exists():
        raise FileNotFoundError(f"IEO file not found: {ieo_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        q_file = tmp_dir / "decl_construct.sparql"
        decls_ttl = tmp_dir / "ieo_declarations.ttl"

        # Write construct query
        q_file.write_text(DECLARATIONS_CONSTRUCT, encoding="utf-8")

        # Step 1: extract declarations from IEO
        # Syntax: robot query --input IEO --query <sparql> <output>
        run_robot([
            "query",
            "--input", str(ieo_path),
            "--query", str(q_file), str(decls_ttl),
        ])

        # Step 2: merge CCO with IEO declarations only
        # robot merge --input CCO --input decls.ttl --output out
        run_robot([
            "merge",
            "--input", str(cco_path),
            "--input", str(decls_ttl),
            "--output", str(out_path),
        ])

    logger.info("Wrote merged ontology to: %s", out_path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Merge CCO with IEO declarations using ROBOT")
    p.add_argument(
        "--cco",
        type=Path,
        default=SRC_ROOT / "CommonCoreOntologiesMerged.ttl",
        help="Path to CommonCoreOntologiesMerged.ttl",
    )
    p.add_argument(
        "--ieo",
        type=Path,
        default=SRC_ROOT / "InformationEntityOntology.ttl",
        help="Path to InformationEntityOntology.ttl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=SRC_ROOT / "ConsolidatedCCO.ttl",
        help="Output path for consolidated ontology",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        merge_with_declarations_only(args.cco, args.ieo, args.out)
    except Exception as e:
        logger.error("Failed to merge ontologies: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
