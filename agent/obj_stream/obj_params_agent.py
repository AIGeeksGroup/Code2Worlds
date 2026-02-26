import json
import os
import re
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

INPUT_JSON_PATH = "./output/obj/obj_select.json"
KNOWLEDGE_FILE_PATH = "./library/obj_nature.txt"
OUTPUT_RESULT_PATH = "./output/obj/obj_param.txt"
FEEDBACK_FILE_PATH = "./output/obj/reflection_feedback.json"


class SemanticKnowledgeBase:
    def __init__(self, file_path, model_name='all-MiniLM-L6-v2'):
        self.file_path = file_path
        self.embedder = SentenceTransformer(model_name)
        self.chunks, self.factory_names, self.clean_names, self.embeddings = [], [], [], None
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.file_path):
            print(f" Error: Knowledge file {self.file_path} not found.")
            return

        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        main_factories = [
            'LeafFactory',
            'CactusFactory',
            'CloudFactory',
            'CrustaceanFactory',
            'MonocotFactory',
            'BoulderFactory',
            'BlenderRockFactory',
            'TreeFactory',
            'BranchFactory',
            'FlowerFactory',
            'MushroomFactory',
            'FruitFactory',
            'CoralFactory',
            'MolluskFactory',
            'SeaweedFactory',
            'UrchinFactory',
            'GrassTuftFactory',
            'DandelionFactory',
            'FernFactory',
            'FishFactory',
            'JellyfishFactory',
            'PalmTreeFactory',
            'ChoppedTrees',  
        ]
        
        valid_chunks, raw_names, clean_names = [], [], []
        current_factory = None
        current_chunk_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            is_main_factory = False
            factory_name = None
            
            for main_factory in main_factories:
                if stripped_line == main_factory:
                    is_main_factory = True
                    factory_name = main_factory
                    break
            
            if is_main_factory and current_factory is not None:
                chunk_text = ''.join(current_chunk_lines).strip()
                if chunk_text:
                    valid_chunks.append(chunk_text)
                    raw_names.append(current_factory)
                    core_name = current_factory.replace("Factory", "")
                    semantic_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', core_name)
                    clean_names.append(semantic_name)
                current_chunk_lines = []
            
            if is_main_factory:
                current_factory = factory_name
                current_chunk_lines = [line]  
            elif current_factory is not None:
                current_chunk_lines.append(line)
        
        if current_factory is not None:
            chunk_text = ''.join(current_chunk_lines).strip()
            if chunk_text:
                valid_chunks.append(chunk_text)
                raw_names.append(current_factory)
                core_name = current_factory.replace("Factory", "")
                semantic_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', core_name)
                clean_names.append(semantic_name)
        

        texts_to_embed = [f"{clean}. {clean}. {chunk}" for clean, chunk in zip(clean_names, valid_chunks)]
        self.embeddings = self.embedder.encode(texts_to_embed, convert_to_tensor=True)
        self.chunks = valid_chunks
        self.factory_names = raw_names
        self.clean_names = clean_names

    def search(self, query: str, top_k=1):
        if self.embeddings is None: return []
        
        query_lower = query.lower().strip()
        exact_matches = []
        
        for idx, (factory_name, clean_name) in enumerate(zip(self.factory_names, self.clean_names)):
            core_name = factory_name.replace("Factory", "").lower()
            clean_name_lower = clean_name.lower().strip()
            
            if query_lower == core_name or query_lower == clean_name_lower:
                exact_matches.append((idx, factory_name, clean_name))
        
        if exact_matches:
            results = []
            for idx, factory_name, clean_name in exact_matches[:top_k]:
                results.append((factory_name, clean_name, self.chunks[idx], 1.0))  
            return results
        
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)
        
        results = []
        for hit in hits[0]: 
            idx = hit['corpus_id']
            results.append((self.factory_names[idx], self.clean_names[idx], self.chunks[idx], hit['score']))
        return results

class ParamGenAgent:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o" 

    def generate(self, factory_name, doc_content, key_obj, scene_prompt, previous_params=None, feedback=None):

        # Build refinement context if feedback exists
        refinement_context = ""
        if previous_params and feedback:
            refinement_context = f"""
### REFINEMENT MODE (CRITICAL):
You are in **REFINEMENT MODE**. This is NOT the first generation attempt.

**Previous Parameters:**
{previous_params}

**Visual Feedback from VLM Critic:**
{feedback}

**Your Task:**
1. **Analyze the Feedback:** Understand what went wrong with the previous parameters.
2. **Adjust Parameters:** Modify the previous parameters to address the specific issues mentioned in the feedback.
3. **Preserve What Works:** Keep parameters that were not criticized.
4. **Be Specific:** If feedback says "too small", increase the relevant parameter (e.g., `scale`, `depth`). If it says "too green", adjust color-related parameters.

**Example:**
- Feedback: "The tree is too green for a 'dead tree'. Set leaf_density to 0.0."
- Action: Set `leaf_density = 0.0` and possibly adjust `color` parameters to brown/gray tones.

**IMPORTANT:** Return the COMPLETE parameter dictionary (not just changes), incorporating the feedback adjustments.
"""

        system_prompt = f"""
        You are an expert 3D Procedural Generation Engineer for Infinigen.
        Your task is to generate a Python dictionary of parameters for the `{factory_name}` class based on the API Documentation.

        ### KNOWLEDGE BASE (API Documentation):
        {doc_content}

        {refinement_context}

        ### INSTRUCTIONS:
        1. **Analyze User Intent:** Read the "Scene Prompt". Infer visual attributes.
           - If the prompt is vague (e.g. "The cup tipped over"), **YOU MUST INFER** a standard, representative shape for that object. Do NOT return an empty dictionary.
           - Example: For a generic "cup", generate parameters for a standard coffee mug (`is_short=False`, `thickness=0.02`).
           - Example: For "delicate cup", set `thickness` low and `scale` small.
        
        2. **Map to Parameters:** STRICTLY use the parameter names defined in the KNOWLEDGE BASE. 
        
        3. **Logic & Defaults:** - **Action-Oriented Prompts:** If the prompt describes an action (e.g., "spilling water"), infer the object's state that allows this (e.g., a cup must be open, `has_lid` might be false).
           - **Always Generate:** Try to generate at least 3-4 core parameters (like `scale`, `depth`, `thickness`) to define a specific look, rather than relying entirely on random defaults.
        
        4. **Output Format:** Return ONLY a valid Python dictionary string. 
           - Example: {{'depth': 0.3, 'scale': 0.2, 'is_short': True}}
           - Do NOT wrap in markdown code blocks.
        """

        user_prompt = f"""
        Target Key Object: "{key_obj}"
        Scene Prompt: "{scene_prompt}"
        
        {"Generate REFINED parameters based on the feedback above." if refinement_context else "Generate the parameter dictionary for " + factory_name + " now. Ensure the output is NOT empty."}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3 
            )
            return self._clean_output(response.choices[0].message.content)
        except Exception as e:
            return f"Error: {e}"

    def _clean_output(self, content):
        content = content.strip()
        if content.startswith("```"):
            lines = content.split('\n')
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            content = "\n".join(lines)
        return content.strip()

def load_input_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0: return data[0]
    return data if isinstance(data, dict) else None

def load_previous_params(path):
    """Load previous parameters from obj_param.txt"""
    if not os.path.exists(path):
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract params dictionary from the file
    match = re.search(r'params\s*=\s*(\{.*?\})', content, re.DOTALL)
    if match:
        return match.group(1)
    return None

def load_feedback(path):
    """Load feedback from reflection_feedback.json"""
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Only return feedback if validation failed
        if not data.get('valid', True):
            return data.get('feedback', None)
    except:
        pass
    
    return None

def main():
    import sys
    
    json_data = load_input_json(INPUT_JSON_PATH)
    
    key_obj = json_data.get("key_obj", "object")

    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = ""
    
    if user_prompt:
        print(f"User Prompt: {user_prompt}")



    kb = SemanticKnowledgeBase(KNOWLEDGE_FILE_PATH)
    results = kb.search(key_obj, top_k=1)
    
    if not results:
        print("No matching factory found.")
        return

    factory_name, clean_name, doc_content, score = results[0]
    print(f"Matched: '{clean_name}' ({factory_name}) | Score: {score:.4f}")
    print(doc_content)
    
    # Check for previous parameters and feedback
    previous_params = load_previous_params(OUTPUT_RESULT_PATH)
    feedback = load_feedback(FEEDBACK_FILE_PATH)
    
    if previous_params and feedback:
        print("REFINEMENT MODE ACTIVATED")
        print(f"Previous Parameters Found:\n{previous_params}")
        print(f"\nFeedback:\n{feedback}")
    
    agent = ParamGenAgent()
    params_str = agent.generate(factory_name, doc_content, key_obj, user_prompt, previous_params, feedback)
    
    output_content = f"""# Result for: "{user_prompt}"
# Key Object: {key_obj}
# Factory: {factory_name}

params = {params_str}
"""
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(output_content)

if __name__ == "__main__":
    main()