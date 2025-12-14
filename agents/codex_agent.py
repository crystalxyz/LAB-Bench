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

import asyncio
import os
from pathlib import Path

from .base_cli_agent import BaseCliAgent
from labbench import Eval, Evaluator
from labbench.evaluator import UnanswerableError


class CodexZeroShotAgent(BaseCliAgent):
    """
    Zero-shot agent that calls the codex CLI for model inference.
    Mimics Harbor adapter behavior by having the agent write to answer.txt.
    """

    def __init__(self, model_name: str = "gpt-4.1", use_cot: bool = True):
        if (
            "CODEX_API_KEY" not in os.environ
            and "OPENAI_API_KEY" not in os.environ
        ):
            raise EnvironmentError("Set CODEX_API_KEY or OPENAI_API_KEY for the CLI.")
        super().__init__(model_name, use_cot)

    @property
    def workspace_prefix(self) -> str:
        return "codex_artifacts"

    def _build_cli_command(self, text_prompt: str, figure_paths: list[Path]) -> list[str]:
        """Build codex CLI command with the text prompt."""
        _ = figure_paths  # figure handling is prompt-driven; no CLI attachments
        return [
            "codex",
            "exec",
            "-m",
            self._model_name,
            "-s",
            "workspace-write",
            "--color",
            "never",
            "--skip-git-repo-check",
            text_prompt,
        ]


    async def _run_cli(self, cmd: list[str], cwd: Path) -> str:
        """Execute codex CLI command in the workspace directory."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            stderr_text = stderr.decode() if stderr else ""
            error_msg = f"codex CLI failed ({proc.returncode}): {stderr_text}"

            # Treat context/size errors as unanswerable
            lowered = stderr_text.lower()
            if "context" in lowered and ("length" in lowered or "token" in lowered):
                raise UnanswerableError(f"Input too large: {error_msg}")
            if "400" in lowered or "bad request" in lowered:
                raise UnanswerableError(f"API error: {error_msg}")

            raise RuntimeError(error_msg)

        stdout_text = stdout.decode().strip()

        # Debug: Print stdout
        print("=" * 60)
        print("CODEX CLI STDOUT:")
        print("=" * 60)
        print(stdout_text)
        print("=" * 60)

        # Debug: Print answer.txt if it exists
        answer_file = cwd / "answer.txt"
        if answer_file.exists():
            print("\nANSWER.TXT CONTENT:")
            print("=" * 60)
            print(answer_file.read_text(encoding="utf-8"))
            print("=" * 60)
        else:
            print("\nNO ANSWER.TXT FILE FOUND")
        print()

        return stdout_text


async def main() -> None:
    """Example: run a small FigQA slice against the codex CLI."""
    agent = CodexZeroShotAgent(model_name="gpt-5.1-codex", use_cot=True)
    evaluator = Evaluator(Eval.FigQA, debug=True)  # debug=True -> first 8 items
    results = await evaluator.score_agent(agent.run_task, n_threads=2)
    print(results["metrics_all"])


if __name__ == "__main__":
    asyncio.run(main())
