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

    async def _run_gemini(self, prompt: str) -> str:
        cmd: list[str] = [
            "gemini",
            "-y",
            "-m",
            self._model_name,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=prompt.encode())

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
        except UnanswerableError:
            # Re-raise UnanswerableError as-is so evaluator can handle it
            raise
        except Exception as e:
            # Convert any other unexpected errors to UnanswerableError
            # so they don't crash the entire evaluation
            raise UnanswerableError(f"Unexpected error in gemini CLI: {e}") from e

    async def get_completion(self, text_prompt: str, figs: list[Image.Image] | None) -> str:
        """Persist figures and construct prompt with base64-encoded images."""
        try:
            workspace_dir = Path.cwd() / "gemini_cli_artifacts"
            run_dir = workspace_dir / f"run_{uuid.uuid4().hex}"
            run_dir.mkdir(parents=True, exist_ok=True)

            image_blocks: list[str] = []
            if figs:
                for idx, fig in enumerate(figs):
                    try:
                        fig_path = run_dir / f"fig_{idx}.png"
                        fig.save(fig_path, format="PNG")
                        
                        # Optimize image for API: convert to JPEG and compress to reduce token usage
                        img = fig.copy()
                        # Convert RGBA/LA/P to RGB (JPEG doesn't support transparency)
                        if img.mode in ("RGBA", "LA", "P"):
                            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "P":
                                img = img.convert("RGBA")
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                            img = rgb_img
                        
                        # Resize if image is very large (reduce token count)
                        max_dimension = 2048
                        if max(img.size) > max_dimension:
                            ratio = max_dimension / max(img.size)
                            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Save as JPEG with compression
                        jpeg_buffer = BytesIO()
                        img.save(jpeg_buffer, format="JPEG", quality=85, optimize=True)
                        jpeg_bytes = jpeg_buffer.getvalue()
                        
                        # Base64 encode
                        fig_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                        data_uri = f"data:image/jpeg;base64,{fig_b64}"
                        image_blocks.append(
                            f'<image index="{idx}" path="{fig_path}">{data_uri}</image>'
                        )
                    except Exception as e:
                        # If image processing fails, mark as unanswerable but continue
                        raise UnanswerableError(f"Failed to process image {idx}: {e}") from e

            uploaded_block = "\n".join(image_blocks) if image_blocks else "No figures."

            full_prompt = (
                f"{text_prompt}\n\n"
                "You have access to the following encoded figure.\n"
                f"{uploaded_block}\n"
            )

            return await self._run_gemini(full_prompt)
        except UnanswerableError:
            # Re-raise UnanswerableError as-is so evaluator can handle it
            raise
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
