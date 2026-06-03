"""
Object Richness Score calculator.

Evaluates scene richness from a single image: object variety, object count,
detail level, and scene complexity. Scores are returned on a 0-100 scale.
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


class ObjectRichnessScorer:
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
        self.model = model
        self.system_prompt = """You are a Scene Richness Evaluator for 3D environments.
Evaluate only the visual richness and diversity of objects in the image.
Do not evaluate prompt alignment.

Evaluate:
1. Object Variety: number and diversity of visible object categories.
2. Object Count: estimated number of distinguishable visible objects.
3. Detail Level: texture richness, small props, wear, surface detail.
4. Scene Complexity: layering, spatial arrangement, visual depth.

Return only JSON:
{
  "overall_score": <0-100>,
  "object_variety": <0-100>,
  "object_count": <number>,
  "detail_level": <0-100>,
  "scene_complexity": <0-100>,
  "detected_objects": ["category1", "category2"],
  "reasoning": "Brief explanation of the richness assessment."
}"""

    @staticmethod
    def encode_image(image_path: Union[str, Path], quality: int = 85) -> str:
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

    def calculate_score(
        self,
        image_path: Union[str, Path],
        scene_description: str = "",
        max_retries: int = 3,
    ) -> dict:
        data_url = self.encode_image(image_path)
        text = "Evaluate object richness for this scene image."
        if scene_description:
            text += f"\nOptional scene note: {scene_description}"

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
                return {
                    "image_path": str(image_path),
                    "richness_score": float(data["overall_score"]),
                    "object_variety": float(data["object_variety"]),
                    "object_count": int(data["object_count"]),
                    "detail_level": float(data["detail_level"]),
                    "scene_complexity": float(data["scene_complexity"]),
                    "detected_objects": data.get("detected_objects", []),
                    "reasoning": data.get("reasoning", ""),
                }
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("All retries failed.")

    def batch_calculate_scores(
        self,
        examples: List[dict],
        output_file: Optional[Union[str, Path]] = None,
        skip_missing: bool = True,
    ) -> List[dict]:
        results = []
        for index, example in enumerate(examples, 1):
            image_path = Path(str(example["picture"]).strip('"').strip("'"))
            if skip_missing and not image_path.exists():
                results.append(
                    {
                        "name": example.get("name", f"Example_{index}"),
                        "image_path": str(image_path),
                        "error": "Image not found",
                        "richness_score": None,
                    }
                )
                continue

            try:
                result = self.calculate_score(
                    image_path=image_path,
                    scene_description=example.get("prompt", ""),
                )
                result["name"] = example.get("name", f"Example_{index}")
                results.append(result)
                print(f"{result['name']}: {result['richness_score']:.1f}/100")
            except Exception as exc:
                results.append(
                    {
                        "name": example.get("name", f"Example_{index}"),
                        "image_path": str(image_path),
                        "error": str(exc),
                        "richness_score": None,
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
    def summarize_scores(results: List[dict]) -> dict:
        valid_scores = [
            item["richness_score"]
            for item in results
            if item.get("richness_score") is not None
        ]
        if not valid_scores:
            return {"count": len(results), "valid": 0}
        return {
            "count": len(results),
            "valid": len(valid_scores),
            "mean": float(np.mean(valid_scores)),
            "min": float(np.min(valid_scores)),
            "max": float(np.max(valid_scores)),
            "std": float(np.std(valid_scores)),
        }


if __name__ == "__main__":
    scorer = ObjectRichnessScorer()

    examples = [
        {
            "name": "example_scene",
            "picture": "/home/user/code2worlds/examples/images/living_room.png",
            "prompt": "A furnished living room with a table, sofa, lamps, books, and small props.",
        }
    ]

    # Example only. Replace the Linux path above with your generated image path.
    # results = scorer.batch_calculate_scores(
    #     examples=examples,
    #     output_file="/home/user/code2worlds/examples/output/richness_results.json",
    # )
    # print(ObjectRichnessScorer.summarize_scores(results))
