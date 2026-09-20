"""Prepare, preflight, or run the one-time authorized final-test evaluation."""

import argparse
from pathlib import Path

from eris_ml.evaluation.final_evaluation import (
    LEDGER_PATH,
    assert_unused_ledger,
    execute_once,
    load_yaml,
    mark_failure,
    preflight,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Keep explicit non-access modes separate from the one-time execute mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Authorize without final access")
    action.add_argument("--preflight", action="store_true", help="Validate development only")
    action.add_argument("--execute", action="store_true", help="Consume the one-time final run")
    args = parser.parse_args()
    if args.prepare:
        record = prepare(ROOT)
        print(
            f"Authorized {record['authorized_candidate']} at {record['created_at_utc']}; "
            "final test not accessed."
        )
    elif args.preflight:
        try:
            manifest, _, development, definition, _ = preflight(ROOT)
        except Exception:
            ledger_path = ROOT / LEDGER_PATH
            ledger = load_yaml(ledger_path)
            assert_unused_ledger(ledger)
            mark_failure(ledger_path, ledger, after_access=False)
            raise
        print(f"Preflight PASS: {manifest['candidate_name']}; {len(development)} development rows; "
              f"{len(definition.all_features)} features; 11/11 hashes; final test not accessed.")
    else:
        result = execute_once(ROOT)
        print(f"One-time final evaluation completed: {result['result']}; "
              f"rows={result['row_count']}; ledger run_count=1.")


if __name__ == "__main__":
    main()
