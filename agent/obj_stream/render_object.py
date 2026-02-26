import bpy
from mathutils import Vector
from pathlib import Path


BLEND_PATH = "./infinigen/outputs/obj/obj.blend"
OUT_DIR    = "./infinigen/outputs/obj/render"

RES_X, RES_Y = 1024, 1024
MARGIN = 1.15
USE_CYCLES = True
CAM_NAME = "AutoCam_Ortho"
OBJECT_NAME = ""  


bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)


def get_target_object(name: str = ""):
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    if bpy.context.active_object and bpy.context.active_object.type == 'MESH':
        return bpy.context.active_object
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            return o

def world_bbox(obj):
    pts = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    min_v = Vector((min(xs), min(ys), min(zs)))
    max_v = Vector((max(xs), max(ys), max(zs)))
    center = (min_v + max_v) * 0.5
    size = (max_v - min_v)
    return center, size

def ensure_camera(name: str):
    cam_obj = bpy.data.objects.get(name)
    if cam_obj and cam_obj.type == 'CAMERA':
        return cam_obj
    cam_data = bpy.data.cameras.new(name + "_DATA")
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj

def look_at(cam_obj, target: Vector):
    direction = (target - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def setup_render(scene):
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100

    if USE_CYCLES:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types, "SceneEEVEE") else 'BLENDER_EEVEE'

    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

def render_still(filepath_abs: Path):
    bpy.context.scene.render.filepath = str(filepath_abs)
    bpy.ops.render.render(write_still=True)


scene = bpy.context.scene  
setup_render(scene)

obj = get_target_object(OBJECT_NAME)

OUT_DIR.mkdir(parents=True, exist_ok=True)

center, size = world_bbox(obj)

cam = ensure_camera(CAM_NAME)
scene.camera = cam

cam.data.type = 'ORTHO'

max_dim = max(size.x, size.y, size.z)
dist = max_dim * 2.0

cam.location = center + Vector((0.0, -dist, 0.0))
look_at(cam, center)
cam.data.ortho_scale = max(size.x, size.z) * MARGIN
front_path = OUT_DIR / "front.png"
render_still(front_path)

cam.location = center + Vector((dist, 0.0, 0.0))
look_at(cam, center)

cam.data.ortho_scale = max(size.y, size.z) * MARGIN

side_path = OUT_DIR / "side.png"
render_still(side_path)

