import cv2
import base64
import json
import os
import sys
import numpy as np
from openai import OpenAI

API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gpt-4o" 


DEFAULT_VIDEO_PATH = "./infinigen/outputs/scene/simulation_output.mp4"
OUTPUT_FEEDBACK_PATH = "./output/postprocess/dynreflection_feedback.json"

class VLMMotionCritic:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model = MODEL_NAME

    def _process_video(self, video_path, target_sample_count=24, skip_ratio=0.05):
        """
        Sampling strategy for 24fps short videos
        
        Args:
            target_sample_count (int): Target number of sampled frames.
                                     Recommended: 24 (VLM sees 1 second of condensed essence).
                                     For 5-second video, samples 1 frame every 5 frames.
            skip_ratio (float): Ratio of beginning to skip.
                                Set to 0.05 (skip first 5% of time) to avoid frame 0 initialization errors.
                                Samples continuously until video end.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("Video file is empty.")

        start_frame = int(total_frames * skip_ratio)

        indices = np.linspace(start_frame, total_frames - 1, target_sample_count, dtype=int)
        
        frames_base64 = []
        print(f"[*] Total video frames: {total_frames} (24fps). Sampling strategy: full coverage, extracting {target_sample_count} frames.")

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                max_dim = 512
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                
                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                _, buffer = cv2.imencode('.jpg', resized_frame)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(b64_str)
        
        cap.release()
        return frames_base64

    def evaluate_video(self, video_path, instruction):
        """
        Evaluate video dynamics against user instruction
        
        Args:
            video_path (str): Path to rendered video (.mp4)
            instruction (str): User's instruction about dynamics (e.g., "Leaves drifting gently")
        
        Returns:
            tuple: (is_valid: bool, feedback: str)
        """
        
        try:
            video_frames = self._process_video(video_path)
        except Exception as e:
            return False, f"Video Processing Error: {str(e)}"

        # 2. Construct Prompt (emphasize Temporal Dynamics)
        system_prompt = """
        You are the **Motion Critic** (VLM-Motion) for a 4D Scene Generation system.
        Your task is to analyze a sequence of video frames to determine if the **Temporal Dynamics** match the user's instruction.

        ### ANALYSIS FOCUS:
        1. **Magnitude Check**: Does the intensity of motion match? 
           - Instruction "Gentle breeze" vs Video "Trees thrashing" -> FAIL.
        2. **Physics Logic**: Are interactions realistic?
           - e.g., "Cup spills water" -> Verify liquid flows downward and spreads.
        3. **Consistency**: Does the lighting change correctly if requested (e.g., "sunset")?

        ### OUTPUT FORMAT (JSON ONLY):
        {
            "valid": boolean,      // True only if motion intensity and logic perfectly align.
            "feedback": "string"   // If valid=false, provide PHYSICS parameter adjustments.
                                   // Example: "Wind is too strong. Reduce wind_velocity_factor by 50%."
        }
        """

        content_payload = [{"type": "text", "text": f"User Instruction: {instruction}"}]

        for b64_frame in video_frames:
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}
            })

        print(f"VLM-Motion is analyzing dynamics: {os.path.basename(video_path)}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_payload}
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            # 4. Parse result
            result_json = json.loads(response.choices[0].message.content)
            is_valid = result_json.get("valid", False)
            feedback = result_json.get("feedback", "No feedback.")
            
            return is_valid, feedback

        except Exception as e:
            print(f"[Error] Motion Evaluation failed: {e}")
            return False, str(e)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        user_instruction = sys.argv[1]
    else:
        user_instruction = ""
        print(f"Warning: No instruction provided. Using default: {user_instruction}")
    
    video_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VIDEO_PATH
    
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)

    critic = VLMMotionCritic()
    
    try:
        is_valid, feedback = critic.evaluate_video(video_file, user_instruction)
        
        print("\n=== 4D Dynamics Evaluation Result ===")
        print(f"Instruction: {user_instruction}")
        print(f"Validation Passed (valid): {is_valid}")
        print(f"Physics Feedback (F_dyn): {feedback}")
        
        result = {
            "instruction": user_instruction,
            "video_path": video_file,
            "valid": is_valid,
            "feedback": feedback
        }
        
        output_dir = os.path.dirname(OUTPUT_FEEDBACK_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(OUTPUT_FEEDBACK_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\nFeedback saved to: {OUTPUT_FEEDBACK_PATH}")
        
    except Exception as e:
        print(f"\n[Test interrupted] Requires real video file to run OpenCV processing.\nError: {e}")
        sys.exit(1)
