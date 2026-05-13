from __future__ import annotations

import argparse
import os


def add_neo4j_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    )
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))


def require_password(password: str | None) -> str:
    if not password:
        raise RuntimeError("Set NEO4J_PASSWORD or pass --password.")
    return password
