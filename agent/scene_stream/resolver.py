import json
import os
from openai import OpenAI

API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "gemini-3-pro-preview"  

# 路径配置
INPUT_MANIFEST = "./output/scene/manifest_scene.json"   
OUTPUT_PARAMS = "./output/scene/scene_params.json" 

class ParameterResolver:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def resolve_parameters(self, manifest_data, user_prompt=None):
        """
        Agent 2: Parameter Concretization
        
        Args:
            manifest_data: Execution Manifest from Agent 1 (Planner)
            user_prompt: Original user instruction (optional, for context)
        """
        

        manifest_str = json.dumps(manifest_data, indent=2)

        user_prompt_section = ""
        if user_prompt:
            user_prompt_section = f"""
### ORIGINAL USER PROMPT:
{user_prompt}

"""

        system_prompt = f"""
You are the **Parameter Resolver** (Agent 2) for Code2Worlds.
Your goal is to address **Scale Ambiguity** by grounding qualitative semantic descriptors into precise continuous scalars.

### INPUT DATA:
{user_prompt_section}User's Execution Manifest:
{manifest_str}

### TASK 1: QUANTITATIVE GROUNDING
Translate abstract descriptions into specific scalar values for the Infinigen engine based on actual parameter ranges from Infinigen's gin configuration files:

**Density Mapping** (based on generate_nature.py and scene_types/*.gin):
- "High density" -> 0.11 to 0.15 (e.g., forest tree_density = 0.11)
- "Medium density" -> 0.05 to 0.10
- "Low density" or "Sparse" -> 0.01 to 0.05 (e.g., desert tree_density = 0.02, snowy = 0.01)
- "Very sparse" -> 0.01 to 0.02

**Lighting Mood Mapping**:
- "Spooky/Dim" -> sun_elevation: 6-15 degrees, fog_density: 0.01-0.02
- "Bright/Sunny" -> sun_elevation: 40-70 degrees, fog_density: 0.0-0.001
- "Dawn/Dusk" -> sun_elevation: 6-20 degrees, fog_density: 0.0-0.01

### TASK 2: LOGICAL CONSISTENCY & CONFLICT RESOLUTION (STRICT RULES)
You MUST enforce these physical rules based on Infinigen scene configurations:

1. **Biome Consistency**: 
   - If manifest implies 'Rainforest' or 'Tropical', FORCE `weather.snow_chance` = 0.0.
   - If manifest implies 'Desert', FORCE `vegetation.tree_density` ≤ 0.02 (typical desert value) and `weather.rain_chance` = 0.0.
   - If manifest implies 'Arctic' or 'Snowy', FORCE `weather.snow_chance` = 1.0.

2. **Coupled Parameter Calibration (The "Black Sky" Fix)**:
   - If `atmosphere.fog_density` > 0.01, you MUST increase `lighting.sun_intensity` proportionally (min 3.0, up to 15.0) to penetrate the fog.
   - Otherwise, the scene may render as a black screen (rendering anomaly).
   - If `atmosphere.dust_density` > 0.01, similarly boost `lighting.sun_intensity`.

### TARGET PARAMETER SCHEMA (JSON ONLY):
Return a flat JSON dictionary with these keys (do not invent new keys).
**Parameter ranges are based on actual Infinigen gin configs and generate_nature.py**:

{{
    "terrain.overall_scale": float (5.0 to 50.0),
    "scene.ground_chance": float (0.0 or 1.0),
    "scene.water_chance": float (0.0 or 1.0),
    "vegetation.bush_density": float (0.03 to 0.12),
    "vegetation.tree_density": float (0.01 to 0.15),
    "vegetation.max_tree_species": int (1 to 10),
    "atmosphere.fog_density": float (0.0 to 0.02),
    "atmosphere.dust_density": float (0.0 to 0.02),
    "weather.snow_chance": float (0.0 or 1.0),
    "weather.rain_chance": float (0.0 or 1.0),
    "lighting.sun_elevation": float (6.0 to 90.0),
    "lighting.sun_intensity": float (0.5 to 15.0)
}}

**Parameter Reference Notes** (for your understanding, not to include in JSON):
- terrain.overall_scale: Ground.scale default=5, forest.gin=10. Range 5-50 for reasonable scenes.
- scene.ground_chance: Boolean flag (0.0=disabled, 1.0=enabled).
- scene.water_chance: Maps to scene.waterbody_chance (0.0=disabled, 1.0=enabled).
- vegetation.bush_density: From generate_nature.py uniform(0.03, 0.12).
- vegetation.tree_density: From generate_nature.py uniform(0.045, 0.15). Actual: desert=0.02, forest=0.11, snowy=0.01.
- vegetation.max_tree_species: Default 3, configurable up to 10.
- atmosphere.fog_density: Maps to shader_atmosphere.density. Actual: desert=0-0.0015, snowy_mountain=0.02.
- atmosphere.dust_density: Maps to nishita_lighting.dust_density. Typically 0.0, can be up to 0.01-0.02.
- weather.snow_chance: Boolean flag (0.0=no snow particles, 1.0=snow_particles enabled).
- weather.rain_chance: Boolean flag (0.0=no rain particles, 1.0=rain_particles enabled).
- lighting.sun_elevation: Maps to nishita_lighting.sun_elevation. Indoor: 6-70, outdoor: 0-90. Typical: 15-70.
- lighting.sun_intensity: Typical 0.6-0.8. Higher values (5-15) needed for foggy scenes (Black Sky Fix).
"""

        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Resolve the parameters now based on the manifest."}
                ],
                temperature=0.2, 
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content.strip())
            
        except Exception as e:
            print(f"API Error in Resolver: {e}")
            return None

def main():
    import sys
    
    if not os.path.exists(INPUT_MANIFEST):
        print(f"Error: {INPUT_MANIFEST} not found. Please run Agent 1 (Planner) first.")
        return

    with open(INPUT_MANIFEST, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
        print(f"Successfully read Manifest: {INPUT_MANIFEST}")

    resolver = ParameterResolver(API_KEY, BASE_URL, MODEL_NAME)

    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = ""
    
    if user_prompt:
        print(f"User Prompt: {user_prompt}")
    
    params = resolver.resolve_parameters(manifest_data, user_prompt)

    if params:
        with open(OUTPUT_PARAMS, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4)
        
        print(f"\nParameter resolution successful! Saved to: {OUTPUT_PARAMS}")

if __name__ == "__main__":
    main()