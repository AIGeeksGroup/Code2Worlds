"""
SGS (Semantic Grounding Score) calculator.

Evaluates how well one or more images match a text prompt. Scores are returned
on a 0-100 scale.
"""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from openai import OpenAI


ImageInput = Union[str, Path]


class GPT4VScorer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key explicitly.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model
        self.system_prompt = """You are a strict visual semantic evaluator.
Your task is to score how well the provided image or images match the text prompt.

Focus on:
1. Object identity and attributes.
2. Spatial layout and relationships.
3. Material, color, pose, and visible state.
4. Whether requested effects or scene elements are present.

Return only JSON:
{
  "score": <0-100>,
  "explanation": "Brief explanation of prompt-image alignment."
}"""

    @staticmethod
    def encode_image(image_path: ImageInput, quality: int = 85) -> str:
        image_path = Path(str(image_path).strip('"').strip("'"))
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _evaluate_single_image(
        self,
        prompt: str,
        image_path: ImageInput,
        max_retries: int = 3,
    ) -> tuple[float, str]:
        data_url = self.encode_image(image_path)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Prompt:\n{prompt}"},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
                return float(data["score"]), data.get("explanation", "")
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("All retries failed.")

    def calculate_score(
        self,
        prompt: str,
        image_path: Union[ImageInput, List[ImageInput]],
        max_retries: int = 3,
    ) -> dict:
        image_paths = [image_path] if isinstance(image_path, (str, Path)) else image_path
        scores = []
        explanations = []

        for path in image_paths:
            score, explanation = self._evaluate_single_image(
                prompt=prompt,
                image_path=path,
                max_retries=max_retries,
            )
            scores.append(score)
            explanations.append(explanation)

        return {
            "prompt": prompt,
            "image_paths": [str(path) for path in image_paths],
            "num_views": len(image_paths),
            "average_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "std": float(np.std(scores)),
            "individual_scores": scores,
            "explanations": explanations,
        }

    def batch_calculate_scores(
        self,
        examples: List[dict],
        output_file: Optional[Union[str, Path]] = None,
    ) -> List[dict]:
        results = []
        for index, example in enumerate(examples, 1):
            try:
                result = self.calculate_score(
                    prompt=example["prompt"],
                    image_path=example["images"],
                )
                result["name"] = example.get("name", f"Example_{index}")
                results.append(result)
                print(f"{result['name']}: {result['average_score']:.1f}/100")
            except Exception as exc:
                results.append(
                    {
                        "name": example.get("name", f"Example_{index}"),
                        "prompt": example.get("prompt", ""),
                        "images": example.get("images", []),
                        "error": str(exc),
                        "average_score": None,
                    }
                )

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return results

    @staticmethod
    def print_result(result: dict) -> None:
        print("=" * 60)
        print(f"Prompt: {result['prompt']}")
        print(f"Views: {result['num_views']}")
        print(f"Average SGS: {result['average_score']:.2f}/100")
        print(f"Range: {result['min_score']:.1f} - {result['max_score']:.1f}")
        for index, (score, explanation) in enumerate(
            zip(result["individual_scores"], result["explanations"]),
            1,
        ):
            print(f"\nView {index}: {score:.1f}/100")
            print(f"Explanation: {explanation}")
        print("=" * 60)


if __name__ == "__main__":
    scorer = GPT4VScorer()

    examples = [
        {
            "name": "example_object",
            "prompt": "A blue ceramic cup with a curved handle on a wooden table.",
            "images": [
                "/home/user/code2worlds/examples/images/cup_front.png",
                "/home/user/code2worlds/examples/images/cup_side.png",
            ],
        }
    ]

    # Example only. Replace the Linux paths above with your generated image paths.
    # results = scorer.batch_calculate_scores(
    #     examples=examples,
    #     output_file="/home/user/code2worlds/examples/output/sgs_results.json",
    # )
    # GPT4VScorer.print_result(results[0])
