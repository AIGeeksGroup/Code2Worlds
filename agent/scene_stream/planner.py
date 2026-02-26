import json
import os
from pathlib import Path
from openai import OpenAI


API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "gemini-3-pro-preview"  

# 输出文件路径
OUTPUT_MANIFEST = "./output/scene/manifest_scene.json"

class EnvironmentPlanner:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def infer_manifest(self, user_instruction):
        """
        Agent 1: Semantic Decomposition
        """
        
        system_prompt = """
You are the **Environment Planner** (Agent 1) for a 4D Procedural Scene Generation system using Infinigen.
Your goal is to act as a **Creative Extrapolation Brain**. You must bridge the gap between sparse user instructions and the dense reality of a 3D world.

### CORE TASKS (Based on Paper Methodology):

1. **Inference of Latent Variables (Context)**:
   - If the user specifies a mood (e.g., "spooky"), you MUST infer the corresponding season (e.g., "autumn"), weather (e.g., "foggy"), and lighting (e.g., "dim").
   - If unspecified, infer a logical default based on the terrain type.

2. **Geomorphological Consistency**:
   - If a water source is implied (e.g., "fishing spot", "bridge"), you MUST explicitly add "river" or "lake" to water_bodies.
   - Ensure terrain types match the biome (e.g., "sand" for desert, "snow" for arctic).

3. **Ecosystem & Detail Population**:
   - Do not just list trees. You must populate the **understory** (bushes, rocks, mushrooms, ferns, etc.) to enhance richness.
   - Define the density qualitatively (low/medium/high) which maps to actual density values.

### INFINIGEN-SPECIFIC CONSTRAINTS:

**Seasons**: MUST be one of: "spring", "summer", "autumn", "winter"

**Weather/Particles**: 
- "sunny" → no particles
- "rainy" → rain_particles
- "foggy" → dust_particles or atmosphere density
- "snowy" → snow_particles

**Terrain Landforms** (from LandTiles.tiles and scene configs):
- "mountain", "canyon", "cliff", "cave", "plain", "coast", "arctic", "desert", "forest", "river", "coral_reef", "kelp_forest", "under_water", "snowy_mountain"

**Ground Cover Materials** (from Terrain.ground_collection):
- "grass" → forest_soil, dirt, soil
- "sand" → sand, sandstone
- "snow" → snow, ice
- "rocky" → cracked_ground, stone
- "dirt" → dirt, soil

**Vegetation Types** (from compose_nature.*_chance parameters):
- Trees: "trees" (always available)
- Bushes: "bushes"
- Ground vegetation: "grass", "ferns", "flowers", "monocots", "mushroom", "pinecone", "pine_needle", "decorative_plants"
- Ground debris: "ground_leaves", "ground_twigs", "chopped_trees"
- Special: "cactus", "kelp", "corals", "seaweed", "urchin", "jellyfish", "seashells"

**Creatures** (from compose_nature.*_creature_registry):
- Ground: "snake", "carnivore", "herbivore", "bird", "beetle", "crab", "crustacean", "fish"
- Flying: "dragonfly", "flyingbird"
- Swarms: "bug_swarm", "fish_school"

**Surface Coverage** (from populate_scene.*_chance):
- "slime_mold", "lichen", "ivy", "moss", "mushroom" (on trees/boulders)
- "snow_layer" (on surfaces)

**Dynamic Elements** (from compose_nature.*_particles_chance):
- "falling_leaves" → leaf_particles
- "rain" → rain_particles
- "snow" → snow_particles
- "dust" → dust_particles
- "marine_snow" → marine_snow_particles

**Other Features**:
- "wind" → wind_chance
- "turbulence" → turbulence_chance
- "fancy_clouds" → fancy_clouds_chance
- "glowing_rocks" → glowing_rocks_chance
- "rocks" → rocks_chance (pebbles)
- "boulders" → boulders_chance
- "simulated_river" → simulated_river_enabled
- "tilted_river" → tilted_river_enabled

### OUTPUT FORMAT:
Return **ONLY** a raw JSON object. Do not include markdown formatting (like ```json).
The JSON structure must be:

{
  "atmosphere": {
    "season": "string (MUST be: spring/summer/autumn/winter)",
    "weather": "string (sunny/rainy/foggy/snowy)",
    "time_of_day": "string (dawn/noon/sunset/night)",
    "lighting_mood": "string (e.g., dramatic, peaceful, spooky)"
  },
  "terrain": {
    "landforms": ["array of strings from: mountain, canyon, cliff, cave, plain, coast, arctic, desert, forest, river, coral_reef, kelp_forest, under_water, snowy_mountain"],
    "water_bodies": ["array of strings: river, lake, none"],
    "ground_cover": "string (grass/sand/snow/rocky/dirt)"
  },
  "ecosystem": {
    "biome_type": "string (e.g., deciduous_forest, desert, tundra, coral_reef, kelp_forest)",
    "primary_vegetation": ["array of strings from: trees, bushes, grass, ferns, flowers, monocots, mushroom, pinecone, pine_needle, decorative_plants, cactus, kelp, corals, seaweed, urchin, jellyfish, seashells"],
    "ground_debris": ["array of strings from: ground_leaves, ground_twigs, chopped_trees"],
    "vegetation_density": "string (low/medium/high - maps to 0.01-0.05/0.05-0.15/0.15-0.3)",
    "creatures": {
      "ground": ["array of strings from: snake, carnivore, herbivore, bird, beetle, crab, crustacean, fish"],
      "flying": ["array of strings from: dragonfly, flyingbird"],
      "swarms": ["array of strings from: bug_swarm, fish_school"]
    }
  },
  "surface_coverage": ["array of strings from: slime_mold, lichen, ivy, moss, mushroom, snow_layer"],
  "dynamics": {
    "wind_status": "string (calm/breezy/stormy)",
    "particles": ["array of strings from: falling_leaves, rain, snow, dust, marine_snow"],
    "other_effects": ["array of strings from: wind, turbulence, fancy_clouds, glowing_rocks, rocks, boulders, simulated_river, tilted_river"]
  }
}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Instruction: {user_instruction}"}
                ],
                temperature=0.7,  
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            return content.strip()
            
        except Exception as e:
            print(f"API Error in Planner: {e}")
            return None

def main():
    import sys
    
    planner = EnvironmentPlanner(API_KEY, BASE_URL, MODEL_NAME)

    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = ""
    
    if not user_prompt:
        print("Warning: No user prompt provided. Using empty prompt.")
    else:
        print(f"User Prompt: {user_prompt}")
    
    json_result = planner.infer_manifest(user_prompt)

    if json_result:
        try:
            parsed_data = json.loads(json_result)

            output_path = Path(OUTPUT_MANIFEST)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=4, ensure_ascii=False)
            
            print("-" * 40)
            print(json.dumps(parsed_data, indent=4, ensure_ascii=False))
            print("-" * 40)
            
        except json.JSONDecodeError:
            print("Raw output:", json_result)
    else:
        print("Failed to generate manifest.")

if __name__ == "__main__":
    main()