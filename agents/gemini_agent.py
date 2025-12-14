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

import asyncio
import os
from pathlib import Path

from .base_cli_agent import BaseCliAgent
from labbench import Eval, Evaluator
from labbench.evaluator import UnanswerableError


class GeminiCliZeroShotAgent(BaseCliAgent):
    """
    Zero-shot agent that calls the gemini CLI for model inference.
    Mimics Harbor adapter behavior by having the agent write to answer.txt.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", use_cot: bool = True):
        if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY for the CLI.")
        super().__init__(model_name, use_cot)

    @property
    def workspace_prefix(self) -> str:
        return "gemini_cli_artifacts"

    def _build_cli_command(self, text_prompt: str, figure_paths: list[Path]) -> list[str]:
        """Build gemini CLI command with the text prompt.

        Note: Gemini CLI automatically scans the workspace directory (cwd) for images.
        Figure paths should be referenced in the prompt, not passed as CLI arguments.
        """
        _ = figure_paths  # Gemini CLI scans workspace automatically; no CLI attachment needed
        return ["gemini", "-y", "-m", self._model_name, text_prompt]

    async def _run_cli(self, cmd: list[str], cwd: Path) -> str:
        """Execute gemini CLI command in the workspace directory."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()

        # Save logs to workspace for debugging
        stdout_text = stdout.decode() if stdout else ""
        stderr_text = stderr.decode() if stderr else ""

        (cwd / "gemini_stdout.txt").write_text(stdout_text, encoding="utf-8")
        (cwd / "gemini_stderr.txt").write_text(stderr_text, encoding="utf-8")
        (cwd / "gemini_returncode.txt").write_text(str(proc.returncode), encoding="utf-8")

        if proc.returncode != 0:
            error_msg = f"gemini CLI failed ({proc.returncode}): {stderr_text}"

            # Check for token limit errors
            if "token count exceeds" in stderr_text.lower() or "INVALID_ARGUMENT" in stderr_text:
                raise UnanswerableError(f"Input too large: {error_msg}")

            # Check for other API errors that should be marked as unanswerable
            if "400" in stderr_text or "Bad Request" in stderr_text:
                raise UnanswerableError(f"API error: {error_msg}")

            raise RuntimeError(error_msg)

        return stdout_text.strip()


async def main() -> None:
    """Example: run a small FigQA slice against the gemini CLI."""
    agent = GeminiCliZeroShotAgent(model_name="gemini-2.5-flash", use_cot=True)
    evaluator = Evaluator(Eval.FigQA, debug=True)  # debug=True -> first 8 items
    results = await evaluator.score_agent(agent.run_task, n_threads=2)
    print(results["metrics_all"])

    # Run each task one by one to see the correct answer
    # for idx, (subset, instance) in enumerate(evaluator.eval_set.instances):
    #     inp, correct_answer, unsure = instance.get_input_output()
    #     print(f"\n{'='*60}")
    #     print(f"Question {idx + 1} (ID: {instance.id})")
    #     print(f"Correct answer: {correct_answer}")
    #     print(f"{'='*60}\n")

    #     agent_answer = await agent.run_task(inp)

    #     is_correct = agent_answer == correct_answer
    #     print(f"\nAgent answered: {agent_answer}")
    #     print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}\n")


if __name__ == "__main__":
    asyncio.run(main())
