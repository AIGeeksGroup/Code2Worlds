import os
import json
import re
from typing import List, Optional, Dict, Union
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

class ObjSelectAgent:
    
    SYSTEM_PROMPT = """You are an expert 4D Dynamics Analyst.
Your goal is to identify the SINGLE most critical "Key Object" from a scene description that requires dynamic simulation (physics, motion, or deformation).

### OUTPUT FORMAT:
You must return a strictly formatted JSON Object (not a list) with exactly two keys:
1. "key_obj": A single, lowercase, common noun representing the object's category (No adjectives, no quantities). If no object should be selected, return null.
2. "reason": A brief explanation of why this object is the dynamic focal point, or why no object was selected.

### SELECTION RULES (Priority Order):
0. **Environmental Changes Only:** If the scene describes ONLY environmental/atmospheric changes (e.g., "24-hour lighting cycle", "weather changes", "rain", "sunrise to sunset", "fog rolling in"), return {"key_obj": null, "reason": "Scene describes environmental changes only, no specific object dynamics."}.
1. **Active vs. Passive:** Select the object moving, falling, breaking, or deforming. Ignore static colliders (e.g., floor, table, wall).
2. **The "Victim" or "Agent":** If an object is being acted upon (e.g., "can" being crushed), it is the key object.
3. **Complexity:** Prefer objects requiring simulation (Cloth, Soft Body, Fluid Emitter) over simple rigid translation.

### FORMATTING RULES:
1. **Strict Noun Only:** - BAD: "red cup", "shattering glass", "a pair of shoes".
   - GOOD: "cup", "glass", "shoe".
2. **Singular Form:** Always convert to singular (e.g., "leaves" -> "leaf").
3. **No Backgrounds:** Never select "ground", "floor", "sky", or "room".

### EXAMPLES:

User: "24-hour lighting cycle from dawn to dusk."
Output: {
  "key_obj": null,
  "reason": "Scene describes environmental changes only, no specific object dynamics."
}

User: "Rain falling on a city street."
Output: {
  "key_obj": null,
  "reason": "Scene describes environmental changes only, no specific object dynamics."
}

User: "A heavy iron anvil crushing a soda can."
Output: {
  "key_obj": "can",
  "reason": "The can is the object undergoing deformation (soft body physics), while the anvil is just a rigid collider."
}

User: "Thousands of golden maple leaves falling in the wind."
Output: {
  "key_obj": "leaf",
  "reason": "The leaves are the active dynamic elements controlled by wind forces."
}

User: "A glass of water spilling onto a wooden table."
Output: {
  "key_obj": "glass",
  "reason": "The glass is the source of the fluid interaction and motion, whereas the table is a static passive collider."
}

User: "A snake slithering across the hot desert sand."
Output: {
  "key_obj": "snake",
  "reason": "The snake is the character performing complex articulation/motion."
}
"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"): 
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def run(self, user_instruction: str, output_path: Optional[str] = None) -> List[dict]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_instruction}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1, 
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            # print(f"Debug - LLM Response: {content}") 

            obj_data = self._parse_json_response(content)

            result_list = [obj_data] if obj_data else []
            
            if output_path and result_list:
                self.save_to_json(result_list, user_instruction, output_path)
            
            return result_list

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return []
    
    def _parse_json_response(self, content: str) -> Optional[dict]:

        cleaned_content = content
        if "```" in content:
            code_block = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if code_block:
                cleaned_content = code_block.group(1)
            else:
                cleaned_content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned_content)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except json.JSONDecodeError:
            pass
        
        json_pattern = r'\{[\s\S]*?\}' 
        match = re.search(json_pattern, content)
        
        if match:
            try:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                print("Regex match found but failed to decode JSON")
                pass
        
        print("Failed to parse JSON from response")
        return None
    
    def save_to_json(self, obj_list: List[dict], user_instruction: str, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(obj_list, f, ensure_ascii=False, indent=2)


def main():
    agent = ObjSelectAgent()
    
    # prompt example
    test_case = ""
    
    # output file
    output_file = "./output/obj/obj_select.json"
    
    result = agent.run(test_case, output_path=output_file)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()