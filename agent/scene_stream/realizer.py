import json
import os
from openai import OpenAI


API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "gemini-3-pro-preview"  

INPUT_MANIFEST = "./output/scene/manifest_scene.json"      
INPUT_PARAMS = "./output/scene/scene_params.json"      
REF_GIN_PATH = "./library/gin.txt"                
REF_CODE_PATH = "./library/nature_example.py"     
OUTPUT_GIN = "./infinigen/infinigen_examples/configs_nature/scene_types/generated_scene.gin"       

class SceneRealizer:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def read_file(self, path):
        if not os.path.exists(path):
            print(f"[Warning] Reference file not found: {path}")
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def synthesize_code(self, param_dict, ref_gin_content, ref_code_content, manifest_data=None, user_prompt=None):
        """
        Agent 3: 3D Scene Realization
        Function: Acts as a domain-specific compiler to translate verified parameters into executable Gin codes.
        
        Args:
            param_dict: Resolved parameters from Agent 2 (Resolver)
            ref_gin_content: Reference gin syntax file content
            ref_code_content: Reference source code (generate_nature.py) content
            manifest_data: Execution Manifest from Agent 1 (Planner) - contains qualitative descriptions
            user_prompt: Original user instruction (optional, for context)
        """
        
        params_str = json.dumps(param_dict, indent=2)
        
        user_prompt_section = ""
        if user_prompt:
            user_prompt_section = f"""
### ORIGINAL USER PROMPT:
{user_prompt}

"""
        
        manifest_section = ""
        if manifest_data:
            manifest_str = json.dumps(manifest_data, indent=2)
            manifest_section = f"""
0. **Execution Manifest (Qualitative Descriptions)**:
{manifest_str}

**IMPORTANT**: The Manifest contains qualitative descriptions that are NOT in the Resolved Parameters.
You MUST use the Manifest to generate additional gin configuration lines that are missing from Resolved Parameters.

**CRITICAL Terrain Material Configuration** (MUST be generated from Manifest):
- `terrain.ground_cover` -> `Terrain.ground_collection` and `Terrain.mountain_collection` (see mapping rules in section 1)
- `terrain.landforms` -> `LandTiles.land_processes` and `scene.ground_ice_chance` (see mapping rules in section 1)
- `terrain.water_bodies` -> `Terrain.liquid_collection` (see mapping rules in section 1)

**Other Manifest-Based Parameters**:
- `ecosystem.primary_vegetation` -> `compose_nature.grass_chance`, `compose_nature.pinecone_chance`, etc.
- `ecosystem.creatures` -> `compose_nature.ground_creature_registry`, `compose_nature.flying_creature_registry`
- `surface_coverage` -> `populate_scene.snow_layer_chance`, `populate_scene.lichen_chance`, etc.
- `dynamics.other_effects` -> `compose_nature.fancy_clouds_chance`, `compose_nature.rocks_chance`, etc.
- `atmosphere.season` -> affects tree season settings (if applicable)

"""

        system_prompt = f"""
You are the **Scene Realizer** (Agent 3) for the Code2Worlds framework.
Your role is a **Domain-Specific Compiler**. You must translate a high-level Parameter Dictionary into valid, executable Infinigen `.gin` configuration code.

### OBJECTIVE:
Map the input `Resolved Parameters` to strict Infinigen internal schema identifiers found in the `Reference Context`.
**CRITICAL**: Only use parameter names that exist in the reference files. Do NOT invent new parameter names.
**CRITICAL**: Use the Manifest to supplement missing configuration items that are not in Resolved Parameters.

### INPUT DATA:
{user_prompt_section}{manifest_section}1. **Resolved Parameters (Quantitative Values)**:
{params_str}

2. **Reference Gin Syntax (Valid Schema)**:
{ref_gin_content[:100000]} 

3. **Reference Source Code (Parameter Definitions)**:
{ref_code_content[:100000]} 

### PARAMETER MAPPING RULES (Based on Actual Infinigen Gin Files):

**1. Terrain & Scene Parameters:**
   - `terrain.overall_scale` (JSON) -> `Ground.scale` (Gin)
     Example: `Ground.scale = 10` (from forest.gin)
   
   - `scene.ground_chance` (JSON) -> `scene.ground_chance` (Gin)
     Example: `scene.ground_chance = 1.0`
   
   - `scene.water_chance` (JSON) -> `scene.waterbody_chance` (Gin)
     Example: `scene.waterbody_chance = 1.0` (from snowy_mountain.gin)
   
   **Terrain Material Configuration (From Manifest `terrain.ground_cover` and `terrain.landforms`):**
   - `terrain.ground_cover: "snow"` (Manifest) -> `Terrain.ground_collection = "mountain"` AND `Terrain.mountain_collection = "mountain"` (Gin)
     Example: `Terrain.ground_collection = "mountain"` (from snowy_mountain.gin)
   
   - `terrain.ground_cover: "grass"` or `"forest"` (Manifest) -> `Terrain.ground_collection = "forest_soil"` (Gin)
     Example: `Terrain.ground_collection = "forest_soil"` (from forest.gin)
   
   - `terrain.ground_cover: "sand"` (Manifest) -> `Terrain.ground_collection = [("infinigen.assets.materials.terrain.sand.Sand", 1)]` (Gin)
     Example: `Terrain.ground_collection = [("infinigen.assets.materials.terrain.sand.Sand", 1)]` (from desert.gin)
   
   - `terrain.water_bodies` (Manifest) -> `Terrain.liquid_collection` (Gin):
     - If `water_bodies` contains `"river"` or `"lake"` -> `Terrain.liquid_collection = "liquid"` (Gin)
     - If `water_bodies` is `["none"]` -> May omit or set to default
     Example: `Terrain.liquid_collection = "liquid"` (from snowy_mountain.gin, cave.gin)
   
   - `terrain.landforms` containing `"snowy_mountain"` or `"arctic"` (Manifest) -> `LandTiles.land_processes` (Gin):
     - `"snowy_mountain"` -> `LandTiles.land_processes = "snowfall"` (Gin)
     - `"arctic"` -> `LandTiles.land_processes = "ice_erosion"` (Gin)
     Examples: 
       - `LandTiles.land_processes = "snowfall"` (from snowy_mountain.gin)
       - `LandTiles.land_processes = "ice_erosion"` (from arctic.gin)
   
   - `terrain.landforms` containing `"snowy_mountain"` or `"arctic"` (Manifest) -> `scene.ground_ice_chance` (Gin):
     - For `"snowy_mountain"` -> `scene.ground_ice_chance = 0.5` (Gin)
     - For `"arctic"` -> `scene.ground_ice_chance = 1.0` (Gin)
     Examples:
       - `scene.ground_ice_chance = 0.5` (from snowy_mountain.gin)
       - `scene.ground_ice_chance = 1` (from arctic.gin)

**2. Vegetation Parameters:**
   - `vegetation.bush_density` (JSON) -> `compose_nature.bush_density` (Gin)
     Example: `compose_nature.bush_density = 0.01` (from snowy_mountain.gin)
   
   - `vegetation.tree_density` (JSON) -> `compose_nature.tree_density` (Gin)
     Example: `compose_nature.tree_density = 0.11` (from forest.gin), `compose_nature.tree_density = 0.02` (from desert.gin)
   
   - `vegetation.max_tree_species` (JSON) -> `compose_nature.max_tree_species` (Gin)
     Example: `compose_nature.max_tree_species = 1` (from snowy_mountain.gin)

**3. Atmosphere Parameters:**
   - `atmosphere.fog_density` (JSON) -> `shader_atmosphere.density` (Gin) OR `atmosphere_light_haze.shader_atmosphere.density` (Gin)
     Examples: 
       - `shader_atmosphere.density = 0.02` (from snowy_mountain.gin)
       - `atmosphere_light_haze.shader_atmosphere.density = ("uniform", 0, 0.0015)` (from desert.gin)
     **Note**: Check reference files to see which form is used. Prefer `shader_atmosphere.density` if both exist.
   
   - `atmosphere.dust_density` (JSON) -> `nishita_lighting.dust_density` (Gin)
     Example: `nishita_lighting.dust_density = 0.0` (from snowy_mountain.gin, coast.gin)

**4. Weather Parameters:**
   - `weather.snow_chance` (JSON) -> `compose_nature.snow_particles_chance` (Gin)
     Example: `compose_nature.snow_particles_chance = 1.0` (from snowy_mountain.gin)
   
   - `weather.rain_chance` (JSON) -> `compose_nature.rain_particles_chance` (Gin)
     Example: `compose_nature.rain_particles_chance = 0.0` (from forest.gin, desert.gin)

**5. Lighting Parameters:**
   - `lighting.sun_elevation` (JSON) -> `nishita_lighting.sun_elevation` (Gin)
     Example: `nishita_lighting.sun_elevation = 15.0` (from snowy_mountain.gin)
   
   - `lighting.sun_intensity` (JSON) -> **UNMAPPED** (No direct gin binding exists)
     **Action**: Add a comment line: `# Unmapped JSON key: lighting.sun_intensity (no matching binding in gin.txt)`
     **Note**: This parameter does not have a direct gin equivalent. Do NOT create a new parameter name.

**6. Manifest-Based Parameters (From Execution Manifest):**
   These parameters are NOT in Resolved Parameters but MUST be generated from the Manifest:
   
   - `ecosystem.primary_vegetation` (array) -> Multiple `compose_nature.*_chance` parameters:
     - `"grass"` -> `compose_nature.grass_chance = 1.0`
     - `"ferns"` -> `compose_nature.ferns_chance = 1.0`
     - `"flowers"` -> `compose_nature.flowers_chance = 1.0`
     - `"monocots"` -> `compose_nature.monocots_chance = 1.0`
     - `"pinecone"` -> `compose_nature.pinecone_chance = 1.0`
     - `"pine_needle"` -> `compose_nature.pine_needle_chance = 1.0`
     - `"decorative_plants"` -> `compose_nature.decorative_plants_chance = 1.0`
     - `"ground_leaves"` -> `compose_nature.ground_leaves_chance = 1.0`
     - `"ground_twigs"` -> `compose_nature.ground_twigs_chance = 1.0`
     - `"chopped_trees"` -> `compose_nature.chopped_trees_chance = 1.0`
     - `"cactus"` -> `compose_nature.cactus_chance = 1.0`
     - `"kelp"` -> `compose_nature.kelp_chance = 1.0`
     - `"corals"` -> `compose_nature.corals_chance = 1.0`
     - `"seaweed"` -> `compose_nature.seaweed_chance = 1.0`
     - `"urchin"` -> `compose_nature.urchin_chance = 1.0`
     - `"jellyfish"` -> `compose_nature.jellyfish_chance = 1.0`
     - `"seashells"` -> `compose_nature.seashells_chance = 1.0`
     - `"mushroom"` -> `compose_nature.mushroom_chance = 1.0`
   
   - `ecosystem.creatures.ground` (array) -> `compose_nature.ground_creature_registry`:
     - `["herbivore"]` -> `compose_nature.ground_creature_registry = [(@HerbivoreFactory, 1)]`
     - `["carnivore"]` -> `compose_nature.ground_creature_registry = [(@CarnivoreFactory, 1)]`
     - `["snake"]` -> `compose_nature.ground_creature_registry = [(@SnakeFactory, 1)]`
     - `["bird"]` -> `compose_nature.ground_creature_registry = [(@BirdFactory, 1)]`
     - Multiple creatures: `compose_nature.ground_creature_registry = [(@HerbivoreFactory, 1), (@SnakeFactory, 1)]`
     - Also set: `compose_nature.ground_creatures_chance = 1.0` if array is non-empty
   
   - `ecosystem.creatures.flying` (array) -> `compose_nature.flying_creature_registry`:
     - `["flyingbird"]` -> `compose_nature.flying_creature_registry = [(@FlyingBirdFactory, 1)]`
     - `["dragonfly"]` -> `compose_nature.flying_creature_registry = [(@DragonflyFactory, 1)]`
     - Also set: `compose_nature.flying_creatures_chance = 1.0` if array is non-empty
   
   - `surface_coverage` (array) -> `populate_scene.*_chance` parameters:
     - `["snow_layer"]` -> `populate_scene.snow_layer_chance = 1.0`
     - `["lichen"]` -> `populate_scene.lichen_chance = 1.0`
     - `["ivy"]` -> `populate_scene.ivy_chance = 1.0`
     - `["moss"]` -> `populate_scene.moss_chance = 1.0`
     - `["slime_mold"]` -> `populate_scene.slime_mold_chance = 1.0`
     - `["mushroom"]` -> `populate_scene.mushroom_chance = 1.0`
   
   - `dynamics.other_effects` (array) -> Multiple chance parameters:
     - `["fancy_clouds"]` -> `compose_nature.fancy_clouds_chance = 1.0`
     - `["rocks"]` -> `compose_nature.rocks_chance = 1.0`
     - `["boulders"]` -> `compose_nature.boulders_chance = 1.0`
     - `["glowing_rocks"]` -> `compose_nature.glowing_rocks_chance = 1.0`
     - `["wind"]` -> `compose_nature.wind_chance = 1.0`
     - `["turbulence"]` -> `compose_nature.turbulence_chance = 1.0`
   
   - `dynamics.particles` (array) -> Particle chance parameters (if not already in Resolved Parameters):
     - `["snow"]` -> `compose_nature.snow_particles_chance = 1.0` (if weather.snow_chance = 1.0)
     - `["rain"]` -> `compose_nature.rain_particles_chance = 1.0` (if weather.rain_chance = 1.0)
     - `["falling_leaves"]` -> `compose_nature.leaf_particles_chance = 1.0`
     - `["dust"]` -> `compose_nature.dust_particles_chance = 1.0`
     - `["marine_snow"]` -> `compose_nature.marine_snow_particles_chance = 1.0`
   
   - `terrain.landforms` (array) -> Terrain configuration (see section 1 above for material mappings):
     - If contains `"desert"` -> Consider `Ground.with_sand_dunes = 1`
     - **Note**: Material and land_processes mappings are handled in section 1 above
   
   - `atmosphere.season` -> May affect tree season (if trees are present):
     - `"winter"` -> Trees will use winter season automatically (via `trees.random_season`)
     - This is usually handled automatically by Infinigen, but you can note it in comments

### COMPILATION RULES (STRICT ADHERENCE):

1. **Symbol Validation (CRITICAL)**:
   - ONLY use parameter names that explicitly appear in the Reference Gin Syntax or Reference Source Code.
   - Do NOT invent, guess, or create new parameter names.
   - If a JSON parameter has no matching gin binding, add a comment explaining it's unmapped.

2. **Parameter Lookup Strategy**:
   - First, check the Reference Gin Syntax for exact parameter names.
   - If found, use that exact name (e.g., `compose_nature.tree_density`, not `tree.density`).
   - If not found in gin syntax, check Reference Source Code for `params.get("key_name")` patterns.
   - If still not found, add a comment: `# Unmapped JSON key: key.name (no matching binding found)`

3. **Value Format**:
   - Use exact values from JSON (e.g., `0.11`, `1.0`, `0.0`).
   - For boolean chances: `0.0` = disabled, `1.0` = enabled.
   - Preserve tuple formats if they exist in reference (e.g., `("uniform", 0, 0.0015)`).

4. **Syntax Integrity**:
   - Output must be a valid, flat `.gin` file.
   - No Markdown formatting (no ```gin blocks).
   - Use `#` for comments.
   - One parameter per line.

5. **Missing Parameters**:
   - If a JSON parameter cannot be mapped to any existing gin parameter, DO NOT create a new one.
   - Instead, add a comment: `# Unmapped JSON key: parameter.name (no matching binding found)`
   - It's better to omit than to guess incorrectly.

6. **Manifest Supplementation (CRITICAL)**:
   - The Manifest contains qualitative descriptions that MUST be converted to gin parameters.
   - **MOST IMPORTANT**: Generate terrain material configuration from `terrain.ground_cover` and `terrain.landforms`:
     - `Terrain.ground_collection` and `Terrain.mountain_collection` (based on `ground_cover`)
     - `Terrain.liquid_collection` (based on `water_bodies`)
     - `LandTiles.land_processes` (based on `landforms` containing "snowy_mountain" or "arctic")
     - `scene.ground_ice_chance` (based on `landforms` containing "snowy_mountain" or "arctic")
   - For each item in `ecosystem.primary_vegetation`, generate the corresponding `compose_nature.*_chance = 1.0`.
   - For creatures, generate `compose_nature.*_creature_registry` and set the corresponding `*_creatures_chance = 1.0`.
   - For `surface_coverage`, generate `populate_scene.*_chance = 1.0`.
   - For `dynamics.other_effects`, generate the corresponding chance parameters.
   - **DO NOT skip these parameters just because they're not in Resolved Parameters.**
   - The Manifest is the source of truth for what should be included in the scene.

### OUTPUT FORMAT:
Return ONLY the raw `.gin` file content. No markdown code blocks, no explanations outside the file.
Start directly with gin configuration lines.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Compile the parameters into a final .gin file."}
                ],
                temperature=0.1,  
            )
            
            content = response.choices[0].message.content.strip()
            
            clean_content = content.replace("```gin", "").replace("```", "").strip()
            return clean_content

        except Exception as e:
            print(f"API Error in Realizer: {e}")
            return None

def main():
    import sys
    
    if not os.path.exists(INPUT_PARAMS):
        print(f"Error: {INPUT_PARAMS} not found. Please run Agent 2 (Resolver) first.")
        return

    realizer = SceneRealizer(API_KEY, BASE_URL, MODEL_NAME)
    
    ref_gin = realizer.read_file(REF_GIN_PATH)
    ref_code = realizer.read_file(REF_CODE_PATH)
    
    if not ref_gin or not ref_code:
        print("Warning: Missing reference files (gin.txt or nature_example.py), the generated result may be inaccurate.")

    with open(INPUT_PARAMS, 'r', encoding='utf-8') as f:
        param_dict = json.load(f)
    
    manifest_data = None
    if os.path.exists(INPUT_MANIFEST):
        with open(INPUT_MANIFEST, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
        print(f"Successfully read Manifest: {INPUT_MANIFEST}")
    else:
        print(f"Manifest file not found: {INPUT_MANIFEST}, will only use Resolved Parameters")

    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = ""
    
    if user_prompt:
        print(f"User Prompt: {user_prompt}")
    
    gin_code = realizer.synthesize_code(param_dict, ref_gin, ref_code, manifest_data, user_prompt)

    if gin_code:
        with open(OUTPUT_GIN, 'w', encoding='utf-8') as f:
            f.write(gin_code)

if __name__ == "__main__":
    main()