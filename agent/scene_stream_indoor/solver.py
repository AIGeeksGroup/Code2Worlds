import json
import os
import sys
from pathlib import Path

from openai import OpenAI


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None
MODEL_NAME = os.getenv("OPENAI_MODEL", "claude-sonnet-4-5-20250929")

INPUT_MANIFEST = Path("./output/indoor/manifest_scene_indoor.json")
OUTPUT_STRATEGY = Path("./output/indoor/solver_scene_indoor.json")


class ConceptualAgent:
    def __init__(self, api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def run(self, task_data, original_user_prompt=""):
        system_prompt = """
You are the Conceptual Agent. You translate indoor scene intent into procedural generation strategies.
You know the logic of Infinigen Indoor generation:

RULES:
1. To make a room "Messy": INCREASE solver steps for small objects and do not select fast_solve.
2. To make a room "Minimalist": ENABLE fast_solve and SET solver steps to low.
3. To view "Overhead": ENABLE overhead config and invisible ceilings.
4. Single-room requests need singleroom config.
5. HighQuality implies using real_geometry and avoiding fast_solve.
6. FirstPerson requests should avoid overhead and keep ceilings visible.

Based on the input JSON, decide:
1. Which base GIN files to load? Choose from: singleroom, fast_solve, overhead, real_geometry, studio.
2. What parameter overrides are needed? For example: room restriction, step counts, ceiling visibility.

Output a strict JSON object with this schema:
{
  "selected_gin_files": ["file1", "file2"],
  "parameter_strategies": {
    "restrict_room": "String (e.g. 'Kitchen') or null",
    "solver_mode": "String (e.g. 'HighClutter', 'Minimalist', 'Default')",
    "ceiling_visibility": "Boolean",
    "quality_mode": "String"
  },
  "reasoning": "Brief explanation of why these gins and parameters were chosen."
}

Output ONLY the raw JSON object. Do not use Markdown formatting.
"""

        user_input_content = f"""
Original User Description: "{original_user_prompt}"

Task Agent Analysis:
{json.dumps(task_data, indent=2)}

Generate the Infinigen indoor strategy JSON.
"""

        print("[Indoor Conceptual Agent] Building generation strategy...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input_content},
                ],
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            return result.strip()
        except Exception as e:
            print(f"API Error in Indoor Conceptual Agent: {e}")
            return None


def save_strategy_to_file(json_str, output_path=OUTPUT_STRATEGY):
    if not json_str:
        print("No conceptual-agent strategy to save.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        fallback_path = output_path.with_suffix(output_path.suffix + ".txt")
        fallback_path.write_text(json_str, encoding="utf-8")
        print(f"Strategy output is not valid JSON. Raw output saved to: {fallback_path}")
        return None

    output_path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Strategy saved to: {output_path}")
    return data


def main():
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else ""

    if not INPUT_MANIFEST.exists():
        print(f"Error: {INPUT_MANIFEST} not found. Please run the indoor planner first.")
        sys.exit(1)

    task_data = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    agent = ConceptualAgent()
    strategy_json_str = agent.run(task_data, user_prompt)
    strategy = save_strategy_to_file(strategy_json_str)
    if strategy is None:
        sys.exit(1)

    gins = strategy.get("selected_gin_files", [])
    print(f"Selected indoor gin configs: {gins}")


if __name__ == "__main__":
    main()
