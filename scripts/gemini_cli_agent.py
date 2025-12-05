"""
Async LAB-Bench agent that calls Gemini (1.5 Pro/Flash) via the official Python SDK.

Requirements:
  pip install google-generativeai
  export GEMINI_API_KEY=your_key

Notes:
  - Supports FigQA/TableQA since images are passed through as PNG bytes.
  - Uses BaseZeroShotAgent to handle prompt construction and MCQ parsing.
"""

from __future__ import annotations

import asyncio
import os
from io import BytesIO

import google.generativeai as genai
from labbench import Eval, Evaluator
from labbench.zero_shot import BaseZeroShotAgent
from PIL.Image import Image


def _image_part(fig: Image) -> dict:
    """Convert a PIL image to a Gemini content part."""
    buf = BytesIO()
    fig.save(buf, format="PNG")
    return {"mime_type": "image/png", "data": buf.getvalue()}


def list_available_models() -> None:
    """Print all available Gemini models and their supported methods."""
    if "GEMINI_API_KEY" not in os.environ:
        raise EnvironmentError("Set GEMINI_API_KEY to use Gemini.")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    print("Available Gemini models:")
    print("=" * 80)
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"  {model.name}")
            print(f"    Display Name: {model.display_name}")
            print(f"    Supported Methods: {', '.join(model.supported_generation_methods)}")
            if hasattr(model, "input_token_limit"):
                print(f"    Input Token Limit: {model.input_token_limit}")
            if hasattr(model, "output_token_limit"):
                print(f"    Output Token Limit: {model.output_token_limit}")
            print()


class GeminiZeroShotAgent(BaseZeroShotAgent):
    """
    Zero-shot agent that delegates prompt/parse to BaseZeroShotAgent and
    forwards the prompt + figures to Gemini.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        if "GEMINI_API_KEY" not in os.environ:
            raise EnvironmentError("Set GEMINI_API_KEY to use Gemini.")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._model = genai.GenerativeModel(model_name=model_name)

    async def get_completion(self, text_prompt: str, figs: list[Image] | None) -> str:
        parts: list = []
        if figs:
            parts.extend(_image_part(fig) for fig in figs)
        parts.append(text_prompt)
        
        # print(text_prompt)
        # print("--------------------------------")

        # google-generativeai is sync; run in a thread to keep our API async.
        result = await asyncio.to_thread(self._model.generate_content, parts)
        # print(result.text)
        return result.text or ""


async def main() -> None:
    """Example: run a small FigQA slice against Gemini."""
    agent = GeminiZeroShotAgent(model_name="gemini-2.5-flash", use_cot=True)
    evaluator = Evaluator(Eval.FigQA, debug=True)  # debug=True -> first 8 items
    results = await evaluator.score_agent(agent.run_task, n_threads=2)
    print(results["metrics_all"])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        list_available_models()
    else:
        asyncio.run(main())
