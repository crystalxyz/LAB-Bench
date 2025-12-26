#!/usr/bin/env python3
"""
Calculate accuracy for codex_artifacts runs.

Compares answer.txt (with [ANSWER]X[/ANSWER] tags) against expected_answer.txt.
"""

import re
import sys
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
    """Calculate accuracy across all run directories."""
    results = {
        "correct": 0,
        "incorrect": 0,
        "unanswered": 0,
        "total": 0,
        "details": [],
    }

    # Find all run directories
    run_dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

    for run_dir in run_dirs:
        answer_file = run_dir / "answer.txt"
        expected_file = run_dir / "expected_answer.txt"

        results["total"] += 1

        # Check if expected answer exists
        if not expected_file.exists():
            results["details"].append({
                "run": run_dir.name,
                "status": "missing_expected",
                "answer": None,
                "expected": None,
            })
            results["unanswered"] += 1
            continue

        expected = expected_file.read_text(encoding="utf-8").strip().upper()

        # Check if answer exists
        if not answer_file.exists():
            results["details"].append({
                "run": run_dir.name,
                "status": "no_answer",
                "answer": None,
                "expected": expected,
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
            })
            results["unanswered"] += 1
            continue

        # Compare
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
        })

    return results


def main():
    # Default to codex_artifacts in current directory
    if len(sys.argv) > 1:
        artifacts_dir = Path(sys.argv[1])
    else:
        artifacts_dir = Path.cwd() / "codex_artifacts"

    if not artifacts_dir.exists():
        print(f"Error: {artifacts_dir} does not exist")
        sys.exit(1)

    results = calculate_accuracy(artifacts_dir)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Accuracy Report: {artifacts_dir.name}")
    print(f"{'='*50}")
    print(f"Total runs:    {results['total']}")
    print(f"Correct:       {results['correct']}")
    print(f"Incorrect:     {results['incorrect']}")
    print(f"Unanswered:    {results['unanswered']}")
    print(f"{'='*50}")

    answered = results["correct"] + results["incorrect"]
    if answered > 0:
        precision = results["correct"] / answered
        print(f"Precision (correct/answered): {precision:.2%} ({results['correct']}/{answered})")

    if results["total"] > 0:
        accuracy = results["correct"] / results["total"]
        coverage = answered / results["total"]
        print(f"Accuracy (correct/total):     {accuracy:.2%} ({results['correct']}/{results['total']})")
        print(f"Coverage (answered/total):    {coverage:.2%} ({answered}/{results['total']})")

    print(f"{'='*50}\n")

    # Print incorrect answers
    incorrect = [d for d in results["details"] if d["status"] == "incorrect"]
    if incorrect:
        print(f"Incorrect answers ({len(incorrect)}):")
        for detail in incorrect:
            print(f"  {detail['run']}: got {detail['answer']}, expected {detail['expected']}")
        print()

    # Print unanswered breakdown
    no_answer = [d for d in results["details"] if d["status"] == "no_answer"]
    parse_error = [d for d in results["details"] if d["status"] == "parse_error"]
    missing_expected = [d for d in results["details"] if d["status"] == "missing_expected"]

    print(f"Unanswered breakdown:")
    print(f"  No answer (timeout/error): {len(no_answer)}")
    print(f"  Parse error:               {len(parse_error)}")
    print(f"  Missing expected:          {len(missing_expected)}")
    print()

    if no_answer:
        print(f"No answer runs ({len(no_answer)}):")
        for detail in no_answer:
            print(f"  {detail['run']}")
        print()

    if parse_error:
        print(f"Parse error runs ({len(parse_error)}):")
        for detail in parse_error:
            print(f"  {detail['run']}: {detail['answer']}")
        print()

    if missing_expected:
        print(f"Missing expected runs ({len(missing_expected)}):")
        for detail in missing_expected:
            print(f"  {detail['run']}")
        print()


if __name__ == "__main__":
    main()
