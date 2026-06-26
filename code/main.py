from __future__ import annotations

import argparse

from claim_verifier.errors import summarize_exception
from claim_verifier.runner import run_predictions_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final claim verification predictions.")
    parser.add_argument("--claims", default="dataset/claims.csv")
    parser.add_argument("--history", default="dataset/user_history.csv")
    parser.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--runs-dir", default="code/runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--trace-prompts", action="store_true")
    args = parser.parse_args()

    try:
        run_dir = run_predictions_sync(
            claims_path=args.claims,
            history_path=args.history,
            requirements_path=args.requirements,
            output_path=args.output,
            runs_dir=args.runs_dir,
            limit=args.limit,
            no_audit=args.no_audit,
            trace_prompts=args.trace_prompts,
            run_prefix="final",
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {summarize_exception(exc)}") from None
    print(f"Wrote predictions to {args.output}")
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
