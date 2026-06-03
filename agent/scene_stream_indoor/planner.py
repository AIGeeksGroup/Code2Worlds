import json
import os
import sys
from pathlib import Path

from openai import OpenAI


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") or None
MODEL_NAME = os.getenv("OPENAI_MODEL", "claude-sonnet-4-5-20250929")

OUTPUT_MANIFEST = Path("./output/indoor/manifest_scene_indoor.json")


class TaskAgent:
    def __init__(self, api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def run_task_agent(self, scene_description):
        system_prompt = """
You are the Task Agent for an Infinigen indoor 3D procedural generation system.
Your goal is to parse user natural-language descriptions into a structured Intent Schema JSON.

Analyze the user input and extract the following fields:

1. "room_type":
   - Infer the room type from the user's description.
   - Valid values: "Kitchen", "Bedroom", "LivingRoom", "DiningRoom", "Bathroom".
   - If not specified, infer from context objects (e.g., "bed" -> Bedroom) or default to "LivingRoom".

2. "vibe":
   - The atmosphere, complexity, or clutter level.
   - Examples: "Messy", "Cluttered", "Minimalist", "Empty", "Standard", "Luxury".
   - Default to "Standard" if not specified.

3. "quality":
   - Rendering/solving quality.
   - "Draft" if the user mentions "fast", "quick", "simple".
   - "HighQuality" otherwise.

4. "viewpoint":
   - SYSTEM CONSTRAINT: This field must ALWAYS be "FirstPerson".
   - Ignore any user requests for "overhead", "top-down", or "plan" views.
   - Override strictly to "FirstPerson".

Output ONLY the raw JSON object. Do not use Markdown formatting.
"""

        user_prompt = f"""
User Scene Description: "{scene_description}"

Parse this into the JSON schema defined above.
"""

        print(f"[Indoor Task Agent] Parsing scene description: {scene_description}")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
            print(f"API Error in Indoor Task Agent: {e}")
            return None


def save_result_to_file(json_str, output_path=OUTPUT_MANIFEST):
    if not json_str:
        print("No task-agent data to save.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        fallback_path = output_path.with_suffix(output_path.suffix + ".txt")
        fallback_path.write_text(json_str, encoding="utf-8")
        print(f"Task-agent output is not valid JSON. Raw output saved to: {fallback_path}")
        return False

    output_path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Task manifest saved to: {output_path}")
    return True


def main():
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    if not user_prompt:
        print('Error: Please provide an indoor scene prompt, e.g. "a cozy bedroom".')
        sys.exit(1)

    agent = TaskAgent()
    result_json_str = agent.run_task_agent(user_prompt)
    if not save_result_to_file(result_json_str):
        sys.exit(1)


if __name__ == "__main__":
    main()
