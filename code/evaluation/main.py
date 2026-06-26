from __future__ import annotations

import argparse
from pathlib import Path

from claim_verifier.errors import summarize_exception
from claim_verifier.evaluation import evaluate_predictions
from claim_verifier.runner import run_predictions_sync
from claim_verifier.tracing import make_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate claim verification predictions.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--claims", default="dataset/sample_claims.csv")
    parser.add_argument("--history", default="dataset/user_history.csv")
    parser.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    parser.add_argument("--runs-dir", default="code/runs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--trace-prompts", action="store_true")
    args = parser.parse_args()

    if args.generate:
        run_id = make_run_id("sample")
        run_dir = Path(args.runs_dir) / run_id
        predictions = run_dir / "sample_predictions.csv"
        try:
            actual_run_dir = run_predictions_sync(
                claims_path=args.claims,
                history_path=args.history,
                requirements_path=args.requirements,
                output_path=str(predictions),
                runs_dir=args.runs_dir,
                limit=args.limit,
                no_audit=args.no_audit,
                trace_prompts=args.trace_prompts,
                run_prefix="sample",
                run_id=run_id,
            )
        except Exception as exc:
            raise SystemExit(f"ERROR: {summarize_exception(exc)}") from None
        run_dir = actual_run_dir
    else:
        if not args.predictions:
            raise SystemExit("--predictions is required unless --generate is set")
        predictions = Path(args.predictions)
        run_dir = Path(args.runs_dir) / make_run_id("score")
        run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "evaluation_report.json"
    mismatches_path = run_dir / "mismatches.csv"
    try:
        report = evaluate_predictions(
            args.claims,
            predictions,
            report_path=report_path,
            mismatches_path=mismatches_path,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {summarize_exception(exc)}") from None
    print(f"Run artifacts: {run_dir}")
    print(f"Predictions: {predictions}")
    print(f"Headline score: {report['headline_score']:.4f}")
    print(f"Mismatch rows: {report['mismatch_count']}")


if __name__ == "__main__":
    main()
