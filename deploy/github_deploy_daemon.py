#!/usr/bin/env python3
"""Poll GitHub for new commits and redeploy the Docker Compose app."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import urllib.request
from pathlib import Path


LOGGER = logging.getLogger("golf-tracer-deploy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Redeploy golf-tracer when GitHub changes.")
    parser.add_argument("--repo", default="JedBorseth/open-golf-tracer")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--workdir", default="/home/jedborseth/golf-tracer")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    workdir = Path(args.workdir)
    last_seen_sha = _current_head(workdir)
    LOGGER.info("starting deploy daemon repo=%s branch=%s head=%s", args.repo, args.branch, last_seen_sha)

    while True:
        try:
            remote_sha = _github_branch_sha(args.repo, args.branch)
            if remote_sha != last_seen_sha:
                LOGGER.info("new commit detected remote=%s local=%s", remote_sha, last_seen_sha)
                _redeploy(workdir, args.branch)
                last_seen_sha = _current_head(workdir)
                LOGGER.info("redeploy complete head=%s", last_seen_sha)
        except Exception:
            LOGGER.exception("deploy poll failed")
        time.sleep(args.interval)


def _github_branch_sha(repo: str, branch: str) -> str:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/commits/{branch}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "golf-tracer-deploy"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["sha"]


def _current_head(workdir: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], workdir).strip()


def _redeploy(workdir: Path, branch: str) -> None:
    _run(["git", "fetch", "origin", branch], workdir)
    _run(["git", "reset", "--hard", f"origin/{branch}"], workdir)
    _run(["docker", "compose", "up", "-d", "--build"], workdir)


def _run(command: list[str], workdir: Path) -> str:
    LOGGER.info("running: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        LOGGER.info(result.stdout.strip())
    if result.stderr:
        LOGGER.info(result.stderr.strip())
    return result.stdout


if __name__ == "__main__":
    main()
