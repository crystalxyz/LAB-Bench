"""
Base class for CLI-based agents that mimic Harbor's workspace workflow.

This provides common functionality for agents that:
- Create a workspace directory for each task
- Save figures to the workspace
- Run a CLI tool with the workspace as cwd
- Expect the agent to write answers to answer.txt
"""

from __future__ import annotations

import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from labbench.evaluator import UnanswerableError
from labbench.utils import AgentInput

# Chain-of-thought prompt
COT_PROMPT = """Think step by step."""

# Custom prompt template that references the actual figure path in the workspace
MCQ_INSTRUCT_TEMPLATE = """You are solving a multiple-choice question in biology.

Instruction:
- Please answer by responding with the letter of the correct answer.
- Wrap your answer with the following tags: [ANSWER]X[/ANSWER] where X is the letter.
- Write your answer to `{answer_file}`.
- {cot}

Task description:
Read the image file at `{figure_path}` to answer the question: {question}

Options:
{answers}
"""


class BaseCliAgent(ABC):
    """
    Base class for CLI-based zero-shot agents.
    Mimics Harbor adapter behavior by having the agent write to answer.txt.
    """

    def __init__(self, model_name: str, use_cot: bool = True):
        self._model_name = model_name
        self.cot_prompt = COT_PROMPT if use_cot else ""
        self.task_buffer: list[dict] = []

    @property
    @abstractmethod
    def workspace_prefix(self) -> str:
        """Return the prefix for workspace directories (e.g., 'codex_artifacts', 'gemini_cli_artifacts')."""
        pass

    @abstractmethod
    async def _run_cli(self, cmd: list[str], cwd: Path, task_info: dict | None = None) -> str:
        """
        Execute the CLI command in the workspace directory.

        Args:
            cmd: Command and arguments to execute
            cwd: Working directory for the command

        Returns:
            stdout from the command

        Raises:
            UnanswerableError: If the command fails with an error that should mark the task as unanswerable
            RuntimeError: For other command failures
        """
        pass

    def _parse_answer_file(self, answer_file: Path) -> str | None:
        """Parse [ANSWER]X[/ANSWER] from answer.txt, mimicking Harbor's test.sh."""
        if not answer_file.exists():
            return None
        content = answer_file.read_text(encoding="utf-8")
        match = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", content, flags=re.S | re.I)
        if match:
            return match.group(1).strip()
        return None

    def _save_figure_to_workspace(self, input: AgentInput, run_dir: Path) -> tuple[str, list[Path]]:
        """
        Save the first figure from input to the workspace directory.

        Args:
            input: Agent input containing figures
            run_dir: Workspace directory path

        Returns:
            Tuple of (figure filename for prompt, list of absolute figure paths)
        """
        figure_path_for_prompt = "N/A"
        figure_paths: list[Path] = []

        if input.figures and len(input.figures) > 0:
            # Try to get original filename from PIL Image, otherwise use default
            fig = input.figures[0]
            if hasattr(fig, 'filename') and fig.filename:
                fig_name = Path(fig.filename).name
            else:
                fig_name = "fig_0.png"

            fig_path = run_dir / fig_name
            fig.save(fig_path, format="PNG")
            figure_path_for_prompt = fig_name  # Relative path in workspace
            figure_paths.append(fig_path)  # Absolute path for CLI

        return figure_path_for_prompt, figure_paths

    def _build_prompt(self, input: AgentInput, figure_path: str) -> str:
        """
        Build the MCQ instruction prompt.

        Args:
            input: Agent input with question and choices
            figure_path: Path to figure file (relative to workspace)

        Returns:
            Formatted prompt string
        """
        return MCQ_INSTRUCT_TEMPLATE.format(
            answer_file="answer.txt",
            question=input.question,
            answers="\n".join(input.choices),
            cot=self.cot_prompt,
            figure_path=figure_path,
        )

    def _parse_answer_from_output(self, raw_output: str, answer_file: Path) -> str:
        """
        Parse answer strictly from answer.txt only.

        Args:
            raw_output: stdout from CLI command (unused, kept for API compatibility)
            answer_file: Path to answer.txt

        Returns:
            Parsed answer letter

        Raises:
            ValueError: If answer cannot be parsed in the expected [ANSWER]X[/ANSWER] format
        """
        _ = raw_output  # Strict mode: only parse from answer.txt, not stdout

        # Parse answer from file only (strict mode to match calculate_accuracy.py)
        answer_from_file = self._parse_answer_file(answer_file)

        if answer_from_file:
            return answer_from_file

        # Strict: no fallback to stdout parsing
        raise ValueError(
            "Could not parse answer: answer.txt missing or not in [ANSWER]X[/ANSWER] format"
        )

    async def run_task(self, input: AgentInput) -> str:
        """
        Run a single task, mimicking Harbor adapter workspace behavior.

        Args:
            input: Agent input with question, choices, and figures

        Returns:
            The answer letter (A, B, C, etc.)

        Raises:
            UnanswerableError: If the task cannot be answered
        """
        try:
            # Create a workspace directory for this run
            workspace_dir = Path.cwd() / self.workspace_prefix
            # Use index if available (0-padded), otherwise fall back to UUID
            if input.index is not None:
                run_dir = workspace_dir / f"run_{input.index:04d}"
            else:
                run_dir = workspace_dir / f"run_{input.id}"

            answer_file = run_dir / "answer.txt"

            # Skip if run_dir already exists (task was already attempted)
            if run_dir.exists():
                existing_answer = self._parse_answer_file(answer_file)
                if existing_answer:
                    print(f"[SKIP] {run_dir.name} already completed, answer: {existing_answer}")
                    return existing_answer
                else:
                    print(f"[SKIP] {run_dir.name} already attempted (no answer)")
                    raise UnanswerableError(f"Task already attempted but no answer found in {run_dir.name}")

            run_dir.mkdir(parents=True, exist_ok=True)

            # Save figure to workspace
            figure_path_for_prompt, figure_paths = self._save_figure_to_workspace(input, run_dir)

            # Build the prompt
            text_prompt = self._build_prompt(input, figure_path_for_prompt)

            # Track task in buffer
            task_buffer_entry = {
                "id": str(input.id),
                "text_prompt": text_prompt,
                "workspace": str(run_dir),
                "raw_output": None,
                "answer_from_file": None,
                "answer": None,
            }
            self.task_buffer.append(task_buffer_entry)

            # Build and run CLI command
            cmd = self._build_cli_command(text_prompt, figure_paths)
            # Pass task info for trajectory logging
            task_info = {
                "id": str(input.id),
                "question": input.question,
                "choices": input.choices,
                "prompt": text_prompt,
                "figure_path": figure_path_for_prompt,
                "expected_answer": input.expected_answer,
                "unsure_answer": input.unsure_answer,
            }
            raw_output = await self._run_cli(cmd, cwd=run_dir, task_info=task_info)
            task_buffer_entry["raw_output"] = raw_output

            # Parse answer
            answer = self._parse_answer_from_output(raw_output, answer_file)
            task_buffer_entry["answer_from_file"] = self._parse_answer_file(answer_file)
            task_buffer_entry["answer"] = answer

            return answer

        except UnanswerableError:
            raise
        except Exception as e:
            raise UnanswerableError(f"Unexpected error in run_task: {e}") from e

    @abstractmethod
    def _build_cli_command(self, text_prompt: str, figure_paths: list[Path]) -> list[str]:
        """
        Build the CLI command to execute.

        Args:
            text_prompt: The formatted prompt text
            figure_paths: List of paths to figure files in the workspace

        Returns:
            List of command arguments
        """
        pass
