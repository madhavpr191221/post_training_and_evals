from __future__ import annotations

import argparse
import json

from .project_01.data import prepare_data
from .project_01.inference import run_model
from .project_01.reporting import build_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-training and evaluation experiments")
    parser.add_argument("--config", default="configs/project_01.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Audit and copy Project 1 images")
    prepare.add_argument("--audit-config", default="configs/project_01_audit.yaml")

    run = subparsers.add_parser("run", help="Run one Project 1 checkpoint")
    run.add_argument("--model", choices=("base", "instruct"), required=True)
    run.add_argument("--limit", type=int, default=None, help="Optional integration-test limit")

    subparsers.add_parser("report", help="Compare completed base and instruct runs")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-data":
        result = prepare_data(args.config, args.audit_config)
    elif args.command == "run":
        result = run_model(args.model, args.config, limit=args.limit)
    else:
        result = build_report(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
