#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_coding_agents.agent_compare import compare_agent_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    summary = compare_agent_runs(args.run_dirs, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
