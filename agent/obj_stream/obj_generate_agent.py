import re
import os
from openai import OpenAI


os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

PARAM_FILE_PATH = "./output/obj/obj_param.txt"
CODE_TEMPLATE_PATH = "./library/obj_nature_generate.txt"
OUTPUT_SCRIPT_PATH = "./infinigen/obj_code.py"

def load_file(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_target_info(param_content):
    factory_match = re.search(r"# Factory:\s*(\w+)", param_content)
    factory_name = factory_match.group(1) if factory_match else None
    
    params_match = re.search(r"params\s*=\s*(\{.*?\})", param_content, re.DOTALL)
    params_str = params_match.group(1) if params_match else "{}"
    
    return factory_name, params_str

def retrieve_code_context_in_memory(template_content, factory_name):
    lines = template_content.split('\n')
    header_lines = []
    for line in lines[:50]: 
        if line.startswith("import ") or line.startswith("from ") or "sys.path" in line:
            header_lines.append(line)
        if line.strip().startswith("# ==") and len(header_lines) > 0:
            break
    header = "\n".join(header_lines)
    
    if "import bpy" not in header:
        header = "import sys\nimport bpy\nimport numpy as np\n# sys.path.append(...) # Check path"


    pattern = re.compile(
        rf"(# =+.*{re.escape(factory_name)}.*?)(\n# =+ |\Z)", 
        re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(template_content)
    
    if match:
        body = match.group(1).strip()
        return f"{header}\n\n# ... [Global Setup] ...\n\n{body}"
    else:
        return None


class CodeGenAgent:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gemini-3-pro-preview"

    def generate_script(self, factory_name, params_str, code_context):
        system_prompt = f"""
        You are an expert Python Developer for Blender (Infinigen).
        Your task is to write a ROBUST, ERROR-FREE Python script by strictly following a Reference Code Context.

        ### INPUTS:
        1. **Code Context**: A working example script containing imports, paths, and instantiation logic.
        2. **Parameters**: A dictionary of specific values to apply to the factory instance.

        ### INSTRUCTIONS (Step-by-Step):

        1. **Environment Setup (Copy-Paste)**:
           - You MUST copy the `import` statements and `sys.path.append(...)` EXACTLY as they appear in the **Code Context**.
           - Do not change the path strings.

        2. **Scene Cleaning**:
           - Include `bpy.ops.object.select_all(action="SELECT")` and `bpy.ops.object.delete()` to start fresh.

        3. **Instantiation (Strict Mimicry)**:
           - Look at how `{factory_name}` is instantiated in the Code Context.
           - **REQUIRED**: Define a seed variable first (e.g., `factory_seed = 12345`).
           - **REQUIRED**: Pass the seed to the constructor: `{factory_name}(factory_seed=factory_seed, coarse=False)`.
           - Do NOT use `factory = {factory_name}(coarse=False)` (missing seed is forbidden).

        4. **Parameter Assignment (Dependency Aware)**:
           - Parse the **Parameters** dictionary: `{params_str}`.
           - Assign them explicitly: `factory.param_name = value`.
           - **CRITICAL CHECK**: If the Code Context shows that `param_B` is calculated based on `param_A` (e.g., `handle_radius = 0.3 * depth`), ensure you set `param_A` (`depth`) BEFORE setting `param_B`.
           - Handle `scale` and `thickness` early, as other params might depend on them.

        5. **Execution & Saving**:
           - Call `.create_asset()` and assign the result to `obj`.
           - Save to `r"{FINAL_BLEND_PATH}"` using `bpy.ops.wm.save_as_mainfile(...)`.

        6. **Final Review (Self-Correction)**:
           - Verify: Did I include `sys.path.append`?
           - Verify: Did I set the seed?
           - Verify: Are there any undefined variables?

        ### OUTPUT FORMAT:
        Return ONLY the valid Python code. No markdown blocks, no explanations.
        """

        user_prompt = f"""
        Target Factory: {factory_name}
        
        ### 1. Code Context (The "Gold Standard" to mimic):
        {code_context}

        ### 2. Parameters to Apply (The Data):
        {params_str}

        Generate the production-ready script now.
        """

        print(f" [CodeGen] Assembling script for {factory_name}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return self._clean_output(response.choices[0].message.content)
        except Exception as e:
            return f"# Error generating code: {e}"

    def _clean_output(self, content):
        content = content.strip()
        if content.startswith("```"):
            lines = content.split('\n')
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            content = "\n".join(lines)
        return content.strip()


def main():

    param_content = load_file(PARAM_FILE_PATH)
    if not param_content: return

    template_content = load_file(CODE_TEMPLATE_PATH)
    if not template_content: return

    factory_name, params_str = extract_target_info(param_content)
    
    if not factory_name:
        if "CupFactory" in param_content: factory_name = "CupFactory"
        elif "BowlFactory" in param_content: factory_name = "BowlFactory"
        else:
            print("Error: Could not identify target factory.")
            return

    code_context = retrieve_code_context_in_memory(template_content, factory_name)
    print(code_context)

    agent = CodeGenAgent()
    final_script = agent.generate_script(factory_name, params_str, code_context)

    with open(OUTPUT_SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(final_script)

if __name__ == "__main__":
    main()