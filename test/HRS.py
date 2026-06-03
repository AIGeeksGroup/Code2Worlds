"""
HRS (Holistic Realism Score) calculator.

Evaluates generated videos on physics plausibility, visual fidelity, and
temporal stability. Scores are returned on a 0-100 scale.
"""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI


class HRSScorer:
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
        self.system_prompt = """You are an expert VFX Supervisor and Physics Simulation Evaluator.
Rate the overall quality of a generated video based on a text prompt.

Evaluate:
1. Physics Plausibility: gravity, collision, inertia, clipping, floating objects.
2. Visual Fidelity: textures, lighting, geometry, realism.
3. Temporal Stability: flickering, morphing, jitter, frame-to-frame consistency.

Return only JSON:
{
  "score": <0-100>,
  "reasoning": "Brief explanation based on physics, visuals, and stability."
}"""

    def sample_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 12,
    ) -> List[np.ndarray]:
        video_path = str(video_path).strip('"').strip("'")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no readable frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        if not frames:
            raise ValueError(f"No frames sampled from video: {video_path}")
        return frames

    @staticmethod
    def frame_to_data_url(frame: np.ndarray, quality: int = 85) -> str:
        image = Image.fromarray(frame)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def calculate_score(
        self,
        prompt: str,
        video_path: Union[str, Path],
        num_frames: int = 12,
        max_retries: int = 3,
    ) -> dict:
        frames = self.sample_frames(video_path, num_frames=num_frames)
        content = [
            {
                "type": "text",
                "text": (
                    f"Text prompt:\n{prompt}\n\n"
                    f"The attached {len(frames)} frames are sampled uniformly from the video. "
                    "Evaluate holistic realism."
                ),
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": self.frame_to_data_url(frame)}}
            for frame in frames
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                data = json.loads(response.choices[0].message.content)
                return {
                    "prompt": prompt,
                    "video_path": str(video_path),
                    "frames_sampled": len(frames),
                    "hrs_score": float(data["score"]),
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
        num_frames: int = 12,
        output_file: Optional[Union[str, Path]] = None,
    ) -> List[dict]:
        results = []
        for index, example in enumerate(examples, 1):
            try:
                result = self.calculate_score(
                    prompt=example["prompt"],
                    video_path=example["video"],
                    num_frames=num_frames,
                )
                result["name"] = example.get("name", f"Example_{index}")
                results.append(result)
                print(f"{result['name']}: {result['hrs_score']:.1f}/100")
            except Exception as exc:
                results.append(
                    {
                        "name": example.get("name", f"Example_{index}"),
                        "prompt": example.get("prompt", ""),
                        "video_path": example.get("video", ""),
                        "error": str(exc),
                        "hrs_score": None,
                    }
                )

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return results


if __name__ == "__main__":
    scorer = HRSScorer()

    examples = [
        {
            "name": "example_video",
            "prompt": "A ceramic cup tips over on a table and water spills naturally across the surface.",
            "video": "/home/user/code2worlds/examples/videos/cup_spill.mp4",
        }
    ]

    # Example only. Replace the Linux path above with your generated video path.
    # scorer.batch_calculate_scores(
    #     examples=examples,
    #     num_frames=12,
    #     output_file="/home/user/code2worlds/examples/output/hrs_results.json",
    # )
