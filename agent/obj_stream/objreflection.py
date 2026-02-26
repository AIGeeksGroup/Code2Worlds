import base64
import json
import os
import sys
from openai import OpenAI

API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gemini-3-pro-preview"

# Default paths
FRONT_IMAGE_PATH = "./infinigen/outputs/obj/render/front.png"
SIDE_IMAGE_PATH = "./infinigen/outputs/obj/render/side.png"
OUTPUT_FEEDBACK_PATH = "./output/obj/reflection_feedback.json"

class VLMCritic:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model = MODEL_NAME

    def _encode_image(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def evaluate(self, front_image_path, side_image_path, instruction):
        """
        Evaluate generated 3D object from front and side views
        
        Args:
            front_image_path (str): Path to front view render
            side_image_path (str): Path to side view render
            instruction (str): User's original instruction
            
        Returns:
            tuple: (is_valid: bool, feedback: str)
            - is_valid (V): Whether validation passed
            - feedback (F_obj): If failed, return modification suggestions; if successful, return empty or praise.
        """
        
        # Prepare images
        base64_front = self._encode_image(front_image_path)
        base64_side = self._encode_image(side_image_path)
        
        system_prompt = """
        You are the **Semantic Visual Critic** for a 3D procedural generation system.
        Your task is to verify if the generated 3D object images (front and side views) match the user's text description.

        ### CRITERIA:
        1. **Semantic Alignment**: Does the object match the description? (e.g., "Dead tree" must have no leaves).
        2. **Visual Quality**: Are there obvious artifacts (broken mesh, floating parts)?
        3. **Multi-view Consistency**: Do the front and side views show the same object consistently?

        ### OUTPUT FORMAT (JSON ONLY):
        {
            "valid": boolean,      // Set to true ONLY if the images strictly meet the instruction.
            "feedback": "string"   // If valid=false, provide SPECIFIC parameter-level advice to fix it. 
                                   // Example: "The tree is too green for a 'dead tree'. Set leaf_density to 0.0."
        }
        """

        print(f"[*] VLM-Critic is evaluating images:")
        print(f"    - Front view: {os.path.basename(front_image_path)}")
        print(f"    - Side view: {os.path.basename(side_image_path)}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"User Instruction: {instruction}\n\nPlease evaluate both the front view and side view:"},
                        {"type": "text", "text": "Front View:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_front}"}},
                        {"type": "text", "text": "Side View:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_side}"}}
                    ]}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            is_valid = result_json.get("valid", False)
            feedback = result_json.get("feedback", "No feedback provided.")
            
            return is_valid, feedback

        except Exception as e:
            print(f"[Error] VLM Evaluation failed: {e}")
            return False, f"Critic Error: {str(e)}"

if __name__ == "__main__":
    # Get user prompt from command line arguments
    if len(sys.argv) > 1:
        user_instruction = sys.argv[1]
    else:
        user_instruction = ""
        print(f"Warning: No instruction provided. Using default: {user_instruction}")
    
    front_img = FRONT_IMAGE_PATH
    side_img = SIDE_IMAGE_PATH
    
    if not os.path.exists(front_img):
        print(f"Error: Front image not found: {front_img}")
        sys.exit(1)
    
    if not os.path.exists(side_img):
        print(f"Error: Side image not found: {side_img}")
        sys.exit(1)
    
    critic = VLMCritic()
    is_valid, feedback_obj = critic.evaluate(front_img, side_img, user_instruction)
    
    print("\n=== Evaluation Result ===")
    print(f"Instruction: {user_instruction}")
    print(f"Validation Passed (V): {is_valid}")
    print(f"Feedback (F_obj): {feedback_obj}")
    
    result = {
        "instruction": user_instruction,
        "front_image": front_img,
        "side_image": side_img,
        "valid": is_valid,
        "feedback": feedback_obj
    }
    
    output_dir = os.path.dirname(OUTPUT_FEEDBACK_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(OUTPUT_FEEDBACK_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nFeedback saved to: {OUTPUT_FEEDBACK_PATH}")
