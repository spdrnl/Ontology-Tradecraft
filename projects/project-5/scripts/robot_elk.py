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
            "Run ROBOT with the ELK reasoner on an input Turtle ontology and write the reasoned output."
        )
    )
    p.add_argument(
        "--input",
        default=str(SRC_ROOT / "module_augmented.ttl"),
        help="Input ontology TTL to reason over (default: src/module_augmented.ttl)",
    )
    p.add_argument(
        "--out",
        default=str(SRC_ROOT / "module_reasoned.ttl"),
        help="Output TTL path for reasoned ontology (default: src/module_reasoned.ttl)",
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


def build_robot_command(
    in_ttl: Path, out_ttl: Path, robot_cmd: list[str], max_mem: str
) -> list[str]:
    """Construct the ROBOT command to run ELK reasoning on a single input TTL and save result."""
    # We support two invocation modes:
    # - If robot_cmd[0] endswith .jar, we call via java -Xmx{max_mem} -jar robot.jar ...
    # - Else, we call the executable directly
    if robot_cmd and robot_cmd[0].endswith(".jar"):
        cmd = [
            "java",
            f"-Xmx{max_mem}",
            "-jar",
            robot_cmd[0],
            "reason",
            "--reasoner",
            "ELK",
            "--input",
            in_ttl.as_posix(),
            "--output",
            out_ttl.as_posix(),
        ]
    else:
        cmd = [
            robot_cmd[0],
            "reason",
            "--reasoner",
            "ELK",
            "--input",
            in_ttl.as_posix(),
            "--output",
            out_ttl.as_posix(),
        ]
    return cmd


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {in_path}")

    robot_cmd = detect_robot(args.robot, ROBOT_DIR)
    cmd = build_robot_command(in_path, out, robot_cmd, args.max_mem)
    run(cmd)

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ROBOT reported success but output missing or empty: {out}")
    logger.info("Wrote reasoned ontology: %s", out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("merge_ttl failed: %s", e)
        sys.exit(1)
