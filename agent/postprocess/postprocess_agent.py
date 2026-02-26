import os
import json
from openai import OpenAI
import re


API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "gemini-3-pro-preview" 

USER_PROMPT = "" 

OUTPUT_SCRIPT_NAME = "./output/postprocess/postprocess.py"   
FEEDBACK_FILE = "./output/postprocess/dynreflection_feedback.json"        

SYSTEM_INSTRUCTION = r"""
# Role
You are an expert 3D Technical Artist and Python developer for Blender. Your goal is to generate Blender Python scripts to create highly visible 4D physical effects (Rigid Body, Mantaflow Fluid/Smoke, Particle Systems, Animations).

# Rules & Constraints
1. **File Paths (STRICT)**:
   - Target Scene Blend: `./infinigen/outputs/fine/scene.blend`
   - External Object Blend (if needed): `./infinigen/outputs/obj/obj.blend`
   - External Object Name (if needed): `Obj`
   - Output Path: `./output/postprocess/postprocess.blend`

2. **Target Identification**:
   - If the prompt requires interacting with a specific object in the scene (e.g., a "table", "chair", "floor"), DO NOT guess its exact name. 
   - Instead, create a placeholder in the `CONFIG` dictionary (e.g., `"target_table_name": ""`) with a comment `# USER_FILL: Open the blend file, find the exact object name in the outliner`.
   - For objects that need specific positioning relative to camera, add a note: `# USER_FILL: You can rotate the camera view to select the best angle before running this script`.

3. **No Auto-Baking**: 
   - Do not include `bpy.ops.ptcache.bake_all()` or any fluid bake commands in the main execution flow.
   - Instead, print clear instructions at the end telling the user how to manually bake (which object to select, which panel to use, which button to click).

4. **Code Structure**:
   - You MUST include a `CONFIG` dictionary at the top of the file for all tweakable parameters (file paths, object names, physical properties, particle counts, frame ranges, etc.).
   - Use a modular approach with clear function names (e.g., `setup_rigid_body_world`, `create_fire_effect`, `setup_rain_particles`).
   - Always include a `main()` function that orchestrates the workflow.
   - Add progress print statements (e.g., `print(">>> Setting up rigid body physics...")`) for user feedback.

5. **Physical Effects Guidelines**:
   - **Rigid Body**: Always apply transforms before adding rigid body. Set appropriate collision margins (0.001). Use substeps (10-20) for complex collisions.
   - **Fluid/Fire**: Use domain with adaptive domain enabled. Set proper resolution (64-128). Configure cache directory. Provide manual baking instructions.
   - **Particles**: Use appropriate emit_from settings. Configure physics type (NEWTON for gravity effects). Set proper lifetime and velocity.
   - **Materials**: Create node-based materials for effects (e.g., blackbody for fire, emission for rain, transparent for invisible emitters).

# Building Blocks (Adapt these patterns to your specific effect)

## Basic Scene Setup
```python
import bpy
import os
import math
import mathutils
import random

CONFIG = {
    # File paths
    "scene_path": "./infinigen/outputs/fine/scene.blend",
    "obj_path": "./infinigen/outputs/obj/obj.blend",
    "output_path": "./output/postprocess/postprocess.blend",
    
    # Object names (USER_FILL if needed)
    "obj_name": "",              # USER_FILL: The object to import from obj.blend
    "target_support_name": "",   # USER_FILL: Open the blend file, find the exact object name in the outliner
                                 # USER_FILL: You can rotate the camera view to select the best angle before running this script
    "camera_name": "camera_0_0", # Default camera name
    
    # Animation settings
    "frame_start": 1,
    "frame_end": 240,
    
    # Effect-specific parameters (add more as needed)
    # ...
}

def open_scene():
    Load the target scene file
    if not os.path.exists(CONFIG["scene_path"]):
        print(f"ERROR: Scene file not found: {CONFIG['scene_path']}")
        return False
    print(f">>> Loading scene: {CONFIG['scene_path']}")
    bpy.ops.wm.open_mainfile(filepath=CONFIG["scene_path"])
    return True

def get_camera():
    Get the camera object
    cam = bpy.data.objects.get(CONFIG["camera_name"])
    if not cam:
        cam = bpy.context.scene.camera
    if not cam:
        print("WARNING: No camera found in scene")
    return cam

def append_object():
    if not CONFIG["obj_name"]:
        print("INFO: No object name specified, skipping import")
        return None
    if not os.path.exists(CONFIG["obj_path"]):
        print(f"ERROR: Object file not found: {CONFIG['obj_path']}")
        return None
    
    inner_path = "Object"
    directory = os.path.join(CONFIG["obj_path"], inner_path)
    bpy.ops.wm.append(
        filepath=os.path.join(directory, CONFIG["obj_name"]),
        directory=directory,
        filename=CONFIG["obj_name"]
    )
    obj = bpy.data.objects.get(CONFIG["obj_name"])
    if obj:
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        print(f">>> Imported object: {CONFIG['obj_name']}")
    return obj

## Rigid Body Physics
def setup_rigid_body_world():
    scene = bpy.context.scene
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    
    scene.rigidbody_world.point_cache.frame_end = CONFIG["frame_end"]
    scene.rigidbody_world.substeps_per_frame = 10  # Increase for better collision detection
    scene.rigidbody_world.solver_iterations = 10

def make_passive_rigidbody(obj, friction=0.5, collision_shape='MESH'):
    Set object as passive rigid body (static obstacle)
    bpy.context.view_layer.objects.active = obj
    if not obj.rigid_body:
        bpy.ops.rigidbody.object_add()
    
    obj.rigid_body.type = 'PASSIVE'
    obj.rigid_body.collision_shape = collision_shape
    obj.rigid_body.friction = friction
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.001

def make_active_rigidbody(obj, mass=1.0, friction=0.5, collision_shape='CONVEX_HULL'):
    Set object as active rigid body (dynamic object)
    bpy.context.view_layer.objects.active = obj
    
    # IMPORTANT: Apply transforms before adding rigid body
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    if not obj.rigid_body:
        bpy.ops.rigidbody.object_add()
    
    obj.rigid_body.type = 'ACTIVE'
    obj.rigid_body.mass = mass
    obj.rigid_body.collision_shape = collision_shape
    obj.rigid_body.friction = friction
    obj.rigid_body.linear_damping = 0.1
    obj.rigid_body.angular_damping = 0.1
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.001
```

## Mantaflow Fire/Smoke
```python
def create_fire_domain(location, scale=(2.0, 2.0, 3.0), resolution=96):
    Create a fluid domain for fire/smoke simulation
    bpy.ops.mesh.primitive_cube_add(location=location)
    domain = bpy.context.active_object
    domain.name = "Fire_Domain"
    domain.scale = scale
    bpy.ops.object.transform_apply(scale=True, location=False)
    
    # Add fluid modifier
    mod = domain.modifiers.new(name="Fluid", type='FLUID')
    mod.fluid_type = 'DOMAIN'
    settings = mod.domain_settings
    
    settings.domain_type = 'GAS'
    settings.resolution_max = resolution
    settings.use_adaptive_domain = True
    settings.use_noise = True
    settings.noise_scale = 2
    
    # Dissolve settings
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 30
    
    # Fire settings
    settings.vorticity = 0.1
    settings.flame_vorticity = 0.5
    settings.burning_rate = 0.7
    
    # Cache settings
    cache_dir = "./output/postprocess/cache"
    os.makedirs(cache_dir, exist_ok=True)
    settings.cache_type = 'ALL'
    settings.cache_directory = cache_dir
    settings.cache_frame_start = CONFIG["frame_start"]
    settings.cache_frame_end = CONFIG["frame_end"]
    
    return domain

def create_fire_flow(location, fuel_amount=1.0):
    Create a fire flow emitter
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.4, subdivisions=2, location=location)
    flow = bpy.context.active_object
    flow.name = "Fire_Flow"
    flow.hide_render = True
    
    mod = flow.modifiers.new(name="Fluid", type='FLUID')
    mod.fluid_type = 'FLOW'
    settings = mod.flow_settings
    
    settings.flow_type = 'BOTH'  # Fire and smoke
    settings.flow_behavior = 'INFLOW'
    settings.fuel_amount = fuel_amount
    settings.surface_distance = 1.0
    
    return flow

def create_fire_material(domain_obj):
    Create blackbody material for fire rendering
    mat = bpy.data.materials.new(name="Fire_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    volume = nodes.new('ShaderNodeVolumePrincipled')
    
    volume.inputs['Density'].default_value = 5.0
    volume.inputs['Blackbody Intensity'].default_value = 5.0
    volume.inputs['Blackbody Tint'].default_value = (1.0, 0.8, 0.6, 1.0)
    volume.inputs['Temperature'].default_value = 1500.0
    
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    
    if domain_obj.data.materials:
        domain_obj.data.materials[0] = mat
    else:
        domain_obj.data.materials.append(mat)
```

## Particle Systems (Rain, Leaves, etc.)
```python
def create_particle_emitter(location, size=10.0, name="Particle_Emitter"):
    Create a plane emitter for particles
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    emitter = bpy.context.active_object
    emitter.name = name
    emitter.hide_render = True
    emitter.display_type = 'WIRE'
    return emitter

def setup_rain_particles(emitter, rain_drop_obj, count=10000, velocity=-20.0):
    Configure rain particle system
    bpy.ops.object.particle_system_add()
    ps = emitter.particle_systems[0]
    settings = ps.settings
    
    settings.count = count
    settings.frame_start = CONFIG["frame_start"]
    settings.frame_end = CONFIG["frame_end"]
    settings.lifetime = 120
    settings.emit_from = 'FACE'
    settings.use_emit_random = True
    
    # Physics
    settings.physics_type = 'NEWTON'
    settings.mass = 0.5
    settings.normal_factor = abs(velocity)
    settings.factor_random = 0.05
    settings.effector_weights.gravity = 0.3
    
    # Rendering
    settings.render_type = 'OBJECT'
    settings.instance_object = rain_drop_obj
    settings.particle_size = 0.02
    settings.use_rotation_instance = True

def create_rain_drop_object():
    Create elongated sphere for rain drop
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=(100, 100, 100))
    drop = bpy.context.active_object
    drop.name = "RainDrop_Instance"
    drop.scale[2] = 25.0  # Elongate for streak effect
    bpy.ops.object.transform_apply(scale=True)
    return drop
```

## Camera-Relative Positioning
```python
def position_in_front_of_camera(obj, camera, distance=5.0, height_offset=0.0):
    Position object in front of camera view
    bpy.context.view_layer.update()
    
    cam_matrix = camera.matrix_world
    cam_direction = cam_matrix.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))
    cam_location = cam_matrix.translation
    
    target_pos = cam_location + (cam_direction * distance)
    target_pos.z += height_offset
    
    obj.location = target_pos
    bpy.context.view_layer.update()
```

## Main Function Template
```python
def main():
    print("=" * 70)
    print("BLENDER EFFECT GENERATOR")
    print("=" * 70)
    
    # Step 1: Load scene
    if not open_scene():
        return
    
    # Step 2: Get camera
    camera = get_camera()
    if not camera:
        print("ERROR: Camera not found")
        return
    
    # Step 3: Import object (if needed)
    obj = append_object()
    
    # Step 4: Setup physics/effects
    # ... your effect-specific code here ...
    
    # Step 5: Set frame range
    bpy.context.scene.frame_start = CONFIG["frame_start"]
    bpy.context.scene.frame_end = CONFIG["frame_end"]
    
    # Step 6: Save output
    output_dir = os.path.dirname(CONFIG["output_path"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bpy.ops.wm.save_as_mainfile(filepath=CONFIG["output_path"])
    print(f"\n Scene saved to: {CONFIG['output_path']}")

if __name__ == "__main__":
    main()
```

# Task
Based on the user's prompt, write a COMPLETE and READY-TO-RUN Python script using the patterns above. 
- Analyze what type of effect is needed (rigid body, fluid, particles, or combination)
- Include appropriate CONFIG parameters with USER_FILL comments where needed
- Use modular functions for each setup step
- Add clear progress messages
- Include manual baking instructions at the end if needed
- Output ONLY valid Python code inside a ```python``` code block
"""

def get_feedback_context():
    """
    Check if feedback file and previous code exist, if so construct correction prompt.
    """
    feedback_text = ""
    if os.path.exists(FEEDBACK_FILE) and os.path.exists(OUTPUT_SCRIPT_NAME):
        print("Detected feedback file and previous code, entering iterative fix mode")

        # Read feedback
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
                feedback_str = json.dumps(feedback_data, indent=2, ensure_ascii=False)
        except Exception as e:
            feedback_str = f"Error reading feedback: {e}"
            
        # Read previously generated code
        with open(OUTPUT_SCRIPT_NAME, "r", encoding="utf-8") as f:
            old_code = f.read()
            
        feedback_text = (
            f"\n\n=========================================\n"
            f"!!! PREVIOUS ATTEMPT FAILED !!!\n"
            f"Here is the code you generated last time:\n"
            f"```python\n{old_code}\n```\n\n"
            f"Here is the Execution Feedback / Error Message:\n"
            f"{feedback_str}\n\n"
            f"Please analyze the feedback, FIX the errors, and rewrite the complete script."
        )
    else:
        print("No feedback file detected, executing fresh generation mode...")
        
    return feedback_text

def generate_script(user_prompt):
    print(f"Current task prompt: {user_prompt}")

    # 1. Assemble final user input (including possible feedback information)
    feedback_context = get_feedback_context()
    final_user_content = user_prompt + feedback_context

    # 2. Initialize OpenAI client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 3. Send request
    print("Requesting LLM to generate code, please wait...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": final_user_content}
            ],
            temperature=0.2,  # Lower temperature for code rigor, but allows flexibility for logic rewrite
        )
        
        code_content = response.choices[0].message.content.strip()
        
        match = re.search(r'```python\s*(.*?)\s*```', code_content, re.DOTALL)
        if match:
            final_code = match.group(1).strip()
        else:
            if "import bpy" in code_content:
                final_code = code_content
            else:
                print("Error: Failed to extract valid Python code from LLM response.")
                print("Original response:", code_content)
                return ""
                
        return final_code

    except Exception as e:
        print(f"API call error: {e}")
        return ""

if __name__ == "__main__":
    # Generate code
    generated_code = generate_script(USER_PROMPT)

    if generated_code:
        # Save to file
        with open(OUTPUT_SCRIPT_NAME, "w", encoding="utf-8") as f:
            f.write(generated_code)
        print(f"Successfully generated script and saved to: {OUTPUT_SCRIPT_NAME}")
    else:
        print("Code generation failed, please check configuration and network connection.")

