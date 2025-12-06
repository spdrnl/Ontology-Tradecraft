#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from common.robot import detect_robot, run
from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "generated"
SRC_ROOT = PROJECT_ROOT / "src"
ROBOT_DIR = PROJECT_ROOT / "robot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge generated/accepted_el.ttl with src/cco-module.ttl using ROBOT, reason with ELK, "
            "and write src/module_augmented.ttl"
        )
    )
    p.add_argument(
        "--base",
        default=str(SRC_ROOT / "InformationEntityOntology.ttl"),
        help="Base ontology TTL to merge into (default: src/InformationEntityOntology.ttl)",
    )
    p.add_argument(
        "--add",
        default=str(GENERATED_ROOT / "accepted_el.ttl"),
        help="Generated axioms TTL to add (default: generated/accepted_el.ttl)",
    )
    p.add_argument(
        "--out",
        default=str(SRC_ROOT / "module_augmented.ttl"),
        help="Output TTL path (default: src/module_augmented.ttl)",
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


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    add = Path(args.add)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        raise FileNotFoundError(f"Base ontology not found: {base}")
    if not add.exists():
        raise FileNotFoundError(f"Generated axioms file not found: {add}")

    robot_cmd = detect_robot(args.robot, ROBOT_DIR)
    cmd = build_robot_command(base, add, out, robot_cmd, args.max_mem)
    run(cmd)

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ROBOT reported success but output missing or empty: {out}")
    logger.info("Wrote augmented ontology: %s", out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("merge_ttl failed: %s", e)
        sys.exit(1)


def build_robot_command(
    base_ttl: Path, add_ttl: Path, out_ttl: Path, robot_cmd: list[str], max_mem: str
) -> list[str]:
    """Construct the ROBOT command to merge, reason with ELK, and save result."""
    # We support two invocation modes:
    # - If robot_cmd[0] endswith .jar, we call via java -Xmx{max_mem} -jar robot.jar ...
    # - Else, we call the executable directly
    if robot_cmd and robot_cmd[0].endswith(".jar"):
        cmd = [
            "java",
            f"-Xmx{max_mem}",
            "-jar",
            robot_cmd[0],
            "merge",
            "--input",
            base_ttl.as_posix(),
            "--input",
            add_ttl.as_posix(),
            "reason",
            "--reasoner",
            "ELK",
            "--output",
            out_ttl.as_posix(),
        ]
    else:
        cmd = [
            robot_cmd[0],
            "merge",
            "--input",
            base_ttl.as_posix(),
            "--input",
            add_ttl.as_posix(),
            "reason",
            "--reasoner",
            "ELK",
            "--output",
            out_ttl.as_posix(),
        ]
    return cmd
