#!/usr/bin/env python3
"""
Calculate accuracy for codex_artifacts runs.

Compares answer.txt (with [ANSWER]X[/ANSWER] tags) against expected_answer.txt.

Metrics follow LAB-Bench evaluator definitions:
- accuracy: correct / total
- precision: correct / sure
- coverage: sure / total

Output format matches Harbor's result.json structure.
"""

import argparse
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path


def parse_answer(content: str) -> tuple[str | None, str | None]:
    """Extract answer letter from [ANSWER]X[/ANSWER] format.

    Matches Harbor's strict behavior: only accepts clean letters (A, B, C, etc.)
    Does NOT accept <A> or other malformed answers.

    Returns:
        Tuple of (valid_answer, raw_extracted)
        - valid_answer: The letter if valid (A-Z), None otherwise
        - raw_extracted: The raw extracted content for error reporting
    """
    match = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", content, flags=re.S | re.I)
    if match:
        raw = match.group(1).strip().upper()
        # Only accept single letters A-Z (matches Harbor's behavior)
        if len(raw) == 1 and raw.isalpha():
            return raw, raw
        # Invalid format (e.g., <A>)
        return None, raw
    return None, None


def calculate_accuracy(artifacts_dir: Path) -> dict:
    """Calculate accuracy across all run directories.

    Uses LAB-Bench evaluator definitions:
    - "sure" = agent did NOT select the unsure/refuse option
    - precision = correct / sure
    - coverage = sure / total
    """
    results = {
        "correct": 0,
        "incorrect": 0,
        "unsure": 0,  # Chose the "Insufficient information" option
        "unanswered": 0,  # No answer or parse error
        "total": 0,
        "details": [],
    }

    # Find all run directories
    run_dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

    for run_dir in run_dirs:
        answer_file = run_dir / "answer.txt"
        expected_file = run_dir / "expected_answer.txt"
        unsure_file = run_dir / "unsure_answer.txt"

        results["total"] += 1

        # Check if expected answer exists
        if not expected_file.exists():
            results["details"].append({
                "run": run_dir.name,
                "status": "missing_expected",
                "answer": None,
                "expected": None,
                "unsure_choice": None,
                "is_sure": False,
            })
            results["unanswered"] += 1
            continue

        expected = expected_file.read_text(encoding="utf-8").strip().upper()

        # Load unsure choice if available
        unsure_choice = None
        if unsure_file.exists():
            unsure_choice = unsure_file.read_text(encoding="utf-8").strip().upper()

        # Check if answer exists
        if not answer_file.exists():
            results["details"].append({
                "run": run_dir.name,
                "status": "no_answer",
                "answer": None,
                "expected": expected,
                "unsure_choice": unsure_choice,
                "is_sure": False,
            })
            results["unanswered"] += 1
            continue

        # Parse answer
        answer_content = answer_file.read_text(encoding="utf-8")
        answer, raw_extracted = parse_answer(answer_content)

        if answer is None:
            results["details"].append({
                "run": run_dir.name,
                "status": "parse_error",
                "answer": raw_extracted or answer_content[:50],
                "expected": expected,
                "unsure_choice": unsure_choice,
                "is_sure": False,
            })
            results["unanswered"] += 1
            continue

        # Check if agent chose the unsure option
        is_sure = (unsure_choice is None) or (answer != unsure_choice)

        if not is_sure:
            # Agent chose "Insufficient information" option
            results["unsure"] += 1
            results["details"].append({
                "run": run_dir.name,
                "status": "unsure",
                "answer": answer,
                "expected": expected,
                "unsure_choice": unsure_choice,
                "is_sure": False,
            })
            continue

        # Compare (agent gave a "sure" answer)
        is_correct = answer == expected
        if is_correct:
            results["correct"] += 1
            status = "correct"
        else:
            results["incorrect"] += 1
            status = "incorrect"

        results["details"].append({
            "run": run_dir.name,
            "status": status,
            "answer": answer,
            "expected": expected,
            "unsure_choice": unsure_choice,
            "is_sure": True,
        })

    return results


def build_harbor_result(results: dict, eval_name: str = "labbench") -> dict:
    """Build a result dict in Harbor's result.json format."""
    n_total = results["total"]
    n_sure = results["correct"] + results["incorrect"]
    n_errors = results["unanswered"]

    # Calculate metrics
    accuracy = results["correct"] / n_total if n_total > 0 else 0.0
    precision = results["correct"] / n_sure if n_sure > 0 else 0.0
    coverage = n_sure / n_total if n_total > 0 else 0.0

    # Build reward_stats (group runs by status)
    reward_stats = {
        "reward": {
            "1.0": [d["run"] for d in results["details"] if d["status"] == "correct"],
            "0.0": [d["run"] for d in results["details"] if d["status"] in ("incorrect", "unsure")],
        }
    }

    # Build exception_stats (group by error type)
    exception_stats = {
        "NoAnswer": [d["run"] for d in results["details"] if d["status"] == "no_answer"],
        "ParseError": [d["run"] for d in results["details"] if d["status"] == "parse_error"],
        "MissingExpected": [d["run"] for d in results["details"] if d["status"] == "missing_expected"],
    }
    # Remove empty categories
    exception_stats = {k: v for k, v in exception_stats.items() if v}

    # Build the Harbor-style result
    eval_key = f"codex__{eval_name}"
    harbor_result = {
        "id": str(uuid.uuid4()),
        "finished_at": datetime.now().isoformat(),
        "n_total_trials": n_total,
        "stats": {
            "n_trials": n_total,
            "n_errors": n_errors,
            "evals": {
                eval_key: {
                    "n_trials": n_total,
                    "n_errors": n_errors,
                    "metrics": [
                        {"accuracy": accuracy},
                        {"precision": precision},
                        {"coverage": coverage},
                    ],
                    "reward_stats": reward_stats,
                    "exception_stats": exception_stats,
                    # Additional LAB-Bench specific stats
                    "labbench_stats": {
                        "correct": results["correct"],
                        "incorrect": results["incorrect"],
                        "unsure": results["unsure"],
                        "unanswered": results["unanswered"],
                        "sure": n_sure,
                        "total": n_total,
                    },
                    # Unsure runs (LAB-Bench specific)
                    "unsure_runs": [d["run"] for d in results["details"] if d["status"] == "unsure"],
                }
            }
        }
    }

    return harbor_result


def print_summary(results: dict, artifacts_dir: Path) -> None:
    """Print human-readable summary to stderr."""
    n_total = results["total"]
    n_sure = results["correct"] + results["incorrect"]

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Accuracy Report: {artifacts_dir.name}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"Total runs:    {n_total}", file=sys.stderr)
    print(f"Correct:       {results['correct']}", file=sys.stderr)
    print(f"Incorrect:     {results['incorrect']}", file=sys.stderr)
    print(f"Unsure:        {results['unsure']}", file=sys.stderr)
    print(f"Unanswered:    {results['unanswered']}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    if n_total > 0:
        accuracy = results["correct"] / n_total
        print(f"Accuracy (correct/total):     {accuracy:.2%} ({results['correct']}/{n_total})", file=sys.stderr)

    if n_sure > 0:
        precision = results["correct"] / n_sure
        print(f"Precision (correct/sure):     {precision:.2%} ({results['correct']}/{n_sure})", file=sys.stderr)

    if n_total > 0:
        coverage = n_sure / n_total
        print(f"Coverage (sure/total):        {coverage:.2%} ({n_sure}/{n_total})", file=sys.stderr)

    print(f"{'='*50}\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy for codex_artifacts runs. Outputs JSON in Harbor result.json format."
    )
    parser.add_argument(
        "artifacts_dir",
        nargs="?",
        default=None,
        help="Path to artifacts directory (default: ./codex_artifacts)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path for JSON result (default: stdout)"
    )
    parser.add_argument(
        "--eval-name",
        default="labbench",
        help="Evaluation name for the result (default: labbench)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress human-readable summary on stderr"
    )
    args = parser.parse_args()

    # Determine artifacts directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = Path.cwd() / "codex_artifacts"

    if not artifacts_dir.exists():
        print(f"Error: {artifacts_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Calculate results
    results = calculate_accuracy(artifacts_dir)

    # Print human-readable summary to stderr (unless quiet)
    if not args.quiet:
        print_summary(results, artifacts_dir)

    # Build Harbor-format result
    harbor_result = build_harbor_result(results, args.eval_name)

    # Output JSON
    json_output = json.dumps(harbor_result, indent=4)
    if args.output:
        args.output.write_text(json_output, encoding="utf-8")
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
