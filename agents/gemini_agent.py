"""
LAB-Bench agent that shells out to the `gemini` CLI (npm package @google/gemini-cli).
Use this if you prefer the CLI workflow instead of the Python SDK.

Prereqs:
    npm install -g @google/gemini-cli
    export GEMINI_API_KEY=your_key  # or GOOGLE_API_KEY

Notes:
    - Gemini CLI automatically scans the workspace directory (cwd) for images
    - Figures are saved to workspace and referenced by path in the prompt
    - Mimics Harbor adapter behavior: agent writes to answer.txt, we read it back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from .base_cli_agent import BaseCliAgent
from .calculate_accuracy import calculate_accuracy, build_harbor_result, print_summary
from labbench import Eval, Evaluator
from labbench.evaluator import UnanswerableError


class GeminiCliZeroShotAgent(BaseCliAgent):
    """
    Zero-shot agent that calls the gemini CLI for model inference.
    Mimics Harbor adapter behavior by having the agent write to answer.txt.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", use_cot: bool = True, timeout: float = 300.0):
        if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY for the CLI.")
        super().__init__(model_name, use_cot)
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timeout = timeout

    @property
    def workspace_prefix(self) -> str:
        return f"gemini_artifacts_{self._run_timestamp}"

    def _build_cli_command(self, text_prompt: str, figure_paths: list[Path]) -> list[str]:
        """Build gemini CLI command with the text prompt.

        Note: Gemini CLI automatically scans the workspace directory (cwd) for images.
        Figure paths should be referenced in the prompt, not passed as CLI arguments.
        """
        _ = figure_paths  # Gemini CLI scans workspace automatically; no CLI attachment needed
        return ["gemini", "-y", "-m", self._model_name, text_prompt]

    async def _run_cli(self, cmd: list[str], cwd: Path, task_info: dict | None = None) -> str:
        """Execute gemini CLI command in the workspace directory.

        Uses `tee` to stream output to file in real-time (like Harbor does),
        so partial trajectory is preserved even on timeout.

        Args:
            cmd: Command to execute
            cwd: Working directory
            task_info: Optional task information to include in trajectory (prompt, input, etc.)
        """
        import shlex

        trajectory_file = cwd / "gemini_trajectory.txt"


        # Build shell command with tee to stream output to file in real-time
        # This ensures partial trajectory is saved even on timeout (like Harbor does)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        shell_cmd = f"{cmd_str} 2>&1 | tee {shlex.quote(str(trajectory_file))}"

        proc = await asyncio.create_subprocess_shell(
            shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        timed_out = False
        try:
            await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            timed_out = True
            # Kill the process - trajectory file already has partial output from tee
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        finally:
            # Explicitly close subprocess transport to prevent
            # "Event loop is closed" errors during garbage collection
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            # Close any remaining pipe transports
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()

        # Read trajectory from file (tee already wrote it)
        stdout_text = ""
        if trajectory_file.exists():
            stdout_text = trajectory_file.read_text(encoding="utf-8").strip()

        # Save expected answer and unsure choice AFTER agent finishes
        # (prevents agent from cheating by reading them during execution)
        if task_info and "expected_answer" in task_info:
            (cwd / "expected_answer.txt").write_text(task_info["expected_answer"], encoding="utf-8")
        if task_info and "unsure_answer" in task_info:
            (cwd / "unsure_answer.txt").write_text(task_info["unsure_answer"], encoding="utf-8")

        # Check for errors and raise exceptions
        if timed_out:
            raise UnanswerableError("Gemini CLI command timed out!")

        if proc.returncode != 0:
            error_msg = f"gemini CLI failed ({proc.returncode})"

            # Treat context/size errors as unanswerable
            lowered = stdout_text.lower()
            if "token count exceeds" in lowered or "invalid_argument" in lowered:
                raise UnanswerableError(f"Input too large: {error_msg}")
            if "400" in lowered or "bad request" in lowered:
                raise UnanswerableError(f"API error: {error_msg}")

            raise RuntimeError(error_msg)

        return stdout_text


async def main(debug: bool = False, timeout: float = 300.0) -> None:
    """Run FigQA evaluation against the gemini CLI."""
    agent = GeminiCliZeroShotAgent(model_name="gemini-2.5-flash", use_cot=True, timeout=timeout)
    evaluator = Evaluator(Eval.FigQA, debug=debug)
    results = await evaluator.score_agent(agent.run_task, n_threads=4)
    print(results["metrics_all"])

    # Auto-save result.json to artifacts directory
    artifacts_dir = Path.cwd() / agent.workspace_prefix
    if artifacts_dir.exists():
        accuracy_results = calculate_accuracy(artifacts_dir)
        print_summary(accuracy_results, artifacts_dir)
        harbor_result = build_harbor_result(accuracy_results, eval_name="labbench")
        result_path = artifacts_dir / "result.json"
        result_path.write_text(json.dumps(harbor_result, indent=4), encoding="utf-8")
        print(f"Result saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LAB-Bench FigQA evaluation with Gemini CLI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only 4 tasks)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each task (default: 300)"
    )
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug, timeout=args.timeout))
