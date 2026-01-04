"""
LAB-Bench agent that shells out to the `codex` CLI instead of calling OpenAI directly.
Mirrors the Gemini CLI workflow and Harbor-style workspace behavior.

Prereqs:
    Install Codex CLI and authenticate (uses OPENAI_API_KEY or CODEX_API_KEY).

Notes:
    - Figures are saved to workspace PNGs and referenced in the prompt (no CLI image flags).
    - The CLI is invoked in a read-only sandbox; answer handling is prompt-driven.
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


class CodexZeroShotAgent(BaseCliAgent):
    """
    Zero-shot agent that calls the codex CLI for model inference.
    Mimics Harbor adapter behavior by having the agent write to answer.txt.
    """

    def __init__(self, model_name: str = "gpt-4.1", use_cot: bool = True, timeout: float = 300.0):
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError("Set OPENAI_API_KEY for the CLI.")
        super().__init__(model_name, use_cot)
        # Set timestamp once at agent creation for consistent directory naming
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timeout = timeout

    @property
    def workspace_prefix(self) -> str:
        return f"codex_artifacts_{self._run_timestamp}"

    def _build_cli_command(self, text_prompt: str, figure_paths: list[Path]) -> list[str]:
        """Build codex CLI command with the text prompt.
        """
        _ = figure_paths  # figure handling is prompt-driven; no CLI attachments
        return [
            "codex",
            "exec",
            # "--dangerously-bypass-approvals-and-sandbox",
            "-m",
            self._model_name,
            "-s",
            "workspace-write",
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            text_prompt,
        ]


    async def _run_cli(self, cmd: list[str], cwd: Path, task_info: dict | None = None) -> str:
        """Execute codex CLI command in the workspace directory.

        Uses `tee` to stream output to file in real-time (like Harbor does),
        so partial trajectory is preserved even on timeout.

        Args:
            cmd: Command to execute
            cwd: Working directory
            task_info: Optional task information to include in trajectory (prompt, input, etc.)
        """
        import shlex
        import time

        trajectory_file = cwd / "codex_trajectory.json"

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

        # Save task info AFTER agent finishes
        # (prevents agent from cheating by reading expected_answer during execution)
        if task_info:
            if "prompt" in task_info:
                (cwd / "prompt.txt").write_text(task_info["prompt"], encoding="utf-8")
            if "expected_answer" in task_info:
                (cwd / "expected_answer.txt").write_text(task_info["expected_answer"], encoding="utf-8")
            if "unsure_answer" in task_info:
                (cwd / "unsure_answer.txt").write_text(task_info["unsure_answer"], encoding="utf-8")

        # Check for errors and raise exceptions
        if timed_out:
            raise UnanswerableError("Codex CLI command timed out!")

        if proc.returncode != 0:
            error_msg = f"codex CLI failed ({proc.returncode})"

            # Treat context/size errors as unanswerable
            lowered = stdout_text.lower()
            if "context" in lowered and ("length" in lowered or "token" in lowered):
                raise UnanswerableError(f"Input too large: {error_msg}")
            if "400" in lowered or "bad request" in lowered:
                raise UnanswerableError(f"API error: {error_msg}")

            raise RuntimeError(error_msg)

        return stdout_text


async def main(debug: bool = False, timeout: float = 300.0) -> None:
    """Run FigQA evaluation against the codex CLI."""
    agent = CodexZeroShotAgent(model_name="gpt-5-codex", use_cot=True, timeout=timeout)
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
    parser = argparse.ArgumentParser(description="Run LAB-Bench FigQA evaluation with Codex CLI")
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
