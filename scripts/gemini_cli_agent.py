"""
LAB-Bench agent that shells out to the `gemini` CLI (npm package @google/gemini-cli).
Use this if you prefer the CLI workflow instead of the Python SDK.

Prereqs:
    npm install -g @google/gemini-cli
    export GEMINI_API_KEY=your_key  # or GOOGLE_API_KEY

Notes:
    - Figures are saved to temp PNGs and their paths are included in the prompt for context.
    - Uses BaseZeroShotAgent for prompt construction and MCQ parsing.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from io import BytesIO
from pathlib import Path

import base64
from labbench import Eval, Evaluator
from labbench.evaluator import UnanswerableError
from labbench.zero_shot import BaseZeroShotAgent
from PIL import Image


class GeminiCliZeroShotAgent(BaseZeroShotAgent):
    """
    Zero-shot agent that delegates prompt/parse to BaseZeroShotAgent and calls
    the gemini CLI for model inference.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)
        if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY for the CLI.")
        self._model_name = model_name

    async def _run_gemini(self, cmd: list[str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            stderr_text = stderr.decode() if stderr else ""
            error_msg = f"gemini CLI failed ({proc.returncode}): {stderr_text}"
            
            # Check for token limit errors
            if "token count exceeds" in stderr_text.lower() or "INVALID_ARGUMENT" in stderr_text:
                raise UnanswerableError(f"Input too large: {error_msg}")
            
            # Check for other API errors that should be marked as unanswerable
            if "400" in stderr_text or "Bad Request" in stderr_text:
                raise UnanswerableError(f"API error: {error_msg}")
            
            raise RuntimeError(error_msg)

        answer = stdout.decode().strip()
        print("--------------------------------")
        print(answer)
        print("--------------------------------")
        return answer


    async def get_completion(self, text_prompt: str, figs: list[Image.Image] | None) -> str:
        """Persist figures and construct prompt with file paths."""
        try:
            workspace_dir = Path.cwd() / "gemini_cli_artifacts"
            run_dir = workspace_dir / f"run_{uuid.uuid4().hex}"
            run_dir.mkdir(parents=True, exist_ok=True)

            fig_paths: list[str] = []
            if figs:
                for idx, fig in enumerate(figs):
                    fig_path = run_dir / f"fig_{idx}.png"
                    fig.save(fig_path, format="PNG")
                    fig_paths.append(str(fig_path))

            full_prompt = text_prompt

            cmd = ["gemini", "-y", "-m", self._model_name, full_prompt] + fig_paths
            return await self._run_gemini(cmd)
        except Exception as e:
            # Convert any other unexpected errors to UnanswerableError
            # so they don't crash the entire evaluation
            raise UnanswerableError(f"Unexpected error in get_completion: {e}") from e


async def main() -> None:
    """Example: run a small FigQA slice against the gemini CLI."""
    agent = GeminiCliZeroShotAgent(model_name="gemini-2.5-flash", use_cot=True)
    evaluator = Evaluator(Eval.FigQA, debug=True)  # debug=True -> first 8 items
    results = await evaluator.score_agent(agent.run_task, n_threads=2)
    print(results["metrics_all"])


if __name__ == "__main__":
    asyncio.run(main())
