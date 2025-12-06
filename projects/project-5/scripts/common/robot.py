from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)

def detect_robot(robot_arg: str, robot_dir: Path) -> list[str]:
    """Return the command (argv list) to invoke ROBOT.

    Preference order:
    1) --robot argument (file or command)
    2) robot/robot.jar in repo (launch via `java -Xmx{mem} -jar ...`)
    3) `robot` available on PATH

    We return just the base command; memory flag will be added later if needed.
    """
    # If user provided a path/command, trust it
    if robot_arg:
        return [robot_arg]

    # Local jar inside repo
    jar = robot_dir / "robot.jar"
    if jar.exists():
        # We'll prepend java and -Xmx when building final command
        return [jar.as_posix()]  # marker that it's a jar

    # System robot on PATH
    robot_on_path = shutil.which("robot")
    if robot_on_path:
        return [robot_on_path]

    raise FileNotFoundError(
        "ROBOT not found. Provide --robot, place robot.jar in ./robot, or install 'robot' on PATH."
    )


def run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # Show a concise snippet of stderr to help debugging
        err = proc.stderr.strip()
        if len(err) > 2000:
            err = err[:2000] + "..."
        logger.error("ROBOT failed (exit %d):\n%s", proc.returncode, err)
        raise RuntimeError(f"ROBOT failed with exit code {proc.returncode}")
    if proc.stdout:
        logger.debug(proc.stdout)
