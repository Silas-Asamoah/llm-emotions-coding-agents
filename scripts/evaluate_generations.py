#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_coding_agents.evaluation import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    summary = {}
    for run_dir in args.run_dirs:
        frame = evaluate_run(run_dir)
        summary[str(run_dir)] = {
            "num_generations": len(frame),
            "visible_pass_rate": round(float(frame["visible_pass"].mean()), 4),
            "hidden_pass_rate": round(float(frame["hidden_pass"].mean()), 4),
            "task_pass_rate": round(float(frame["task_pass"].mean()), 4),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
