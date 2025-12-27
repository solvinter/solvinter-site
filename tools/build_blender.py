#!/usr/bin/env python3
import argparse
import os
import subprocess
import textwrap
import tempfile
from shutil import which

import yaml


def find_blender() -> str:
    b = which("blender")
    if b:
        return b

    app_bin = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(app_bin):
        return app_bin

    raise FileNotFoundError("Blender not found. Install Blender or add it to PATH.")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path", help="Path to building YAML (e.g. buildings/atelje0_1.yaml)")
    ap.add_argument("--outdir", default="outputs/atelje0_1", help="Output directory")
    ap.add_argument("--export-glb", action="store_true", help="Export GLB")
    ap.add_argument("--render", action="store_true", help="Render a preview PNG")
    args = ap.parse_args()

    data = load_yaml(args.yaml_path)
    b = data.get("building", data)  # supports nested or flat YAML
    bid = b.get("id", "building")

    # Skip YAMLs that are metadata-only (no geometry yet)
    if 'footprint' not in b or 'outer' not in b.get('footprint', {}):
        print(f"SKIP: {bid} (no footprint in YAML)")
        return

    footprint = b["footprint"]["outer"]
    base_h = float(b["base"]["height"])
    wall_h = float(b["walls"]["height"])
    wall_t = float(b["walls"].get("thickness", 0.15))

    roof = b.get("roof", {})
    roof_type = roof.get("type", "flat")
    overhang = float(roof.get("overhang", 0.0))
    ridge_above = float(roof.get("ridge_height_above_walls", 0.0))

    x = float(footprint["x"])
    y = float(footprint["y"])

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    script = textwrap.dedent(f"""
    import bpy
    import math

    # Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    def add_box(name, sx, sy, sz, z0=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, z0 + sz/2))
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (sx/2, sy/2, sz/2)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return obj

    # Base
    base = add_box("{bid}_base", {x}, {y}, {base_h}, z0=0.0)

    # Walls (outer cube - inner cube boolean)
    outer = add_box("{bid}_walls_outer", {x}, {y}, {wall_h}, z0={base_h})
    inner_x = max({x} - 2*{wall_t}, 0.1)
    inner_y = max({y} - 2*{wall_t}, 0.1)
    inner = add_box("{bid}_walls_inner_cut", inner_x, inner_y, {wall_h}+0.02, z0={base_h}-0.01)

    mod = outer.modifiers.new(name="WallHollow", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = inner
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(inner, do_unlink=True)
    outer.name = "{bid}_walls"

    # Roof (placeholder box for now)
    if "{roof_type}" == "gable":
        roof_x = {x} + 2*{overhang}
        roof_y = {y} + 2*{overhang}
        roof_h = max({ridge_above}, 0.01)
        roof = add_box("{bid}_roof", roof_x, roof_y, roof_h, z0={base_h}+{wall_h})
    else:
        roof = add_box("{bid}_roof", {x}, {y}, 0.05, z0={base_h}+{wall_h})

    # Light + Camera for render
    bpy.ops.object.light_add(type='SUN', location=(10, -10, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    bpy.ops.object.camera_add(location=(10, -10, 8))
    cam = bpy.context.active_object
    cam.rotation_euler = (math.radians(60), 0, math.radians(45))
    scene.camera = cam

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000

    # Save .blend
    bpy.ops.wm.save_as_mainfile(filepath=r"{outdir}/{bid}.blend")
    """).strip()

    if args.export_glb:
        script += textwrap.dedent(f"""

        bpy.ops.export_scene.gltf(
            filepath=r"{outdir}/{bid}.glb",
            export_format='GLB',
            export_yup=True
        )
        """)

    if args.render:
        script += textwrap.dedent(f"""

        scene.render.filepath = r"{outdir}/{bid}_preview.png"
        bpy.ops.render.render(write_still=True)
        """)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        temp_script_path = tf.name

    blender = find_blender()
    cmd = [blender, "--background", "--python", temp_script_path]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["HOME"] = os.environ.get("HOME", "")
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/Applications/Blender.app/Contents/MacOS"

    subprocess.check_call(cmd, env=env)
    os.unlink(temp_script_path)
    print(f"Done. Outputs in: {outdir}")

if __name__ == "__main__":
    main()
