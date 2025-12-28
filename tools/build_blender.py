#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
from shutil import which

import yaml


BLENDER_APP_BIN = "/Applications/Blender.app/Contents/MacOS/Blender"
BLENDER_RESOURCES = "/Applications/Blender.app/Contents/Resources/5.0"


def find_blender() -> str:
    b = which("blender")
    if b:
        return b
    if os.path.exists(BLENDER_APP_BIN):
        return BLENDER_APP_BIN
    raise FileNotFoundError("Blender not found. Install Blender or add it to PATH.")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_blender_env() -> dict:
    """
    Blender on macOS can crash if inherited env points to other python installs/venvs.
    Keep minimal env.
    """
    env = {}
    env["HOME"] = os.environ.get("HOME", "")
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/Applications/Blender.app/Contents/MacOS"
    if os.path.isdir(BLENDER_RESOURCES):
        env["BLENDER_SYSTEM_SCRIPTS"] = os.path.join(BLENDER_RESOURCES, "scripts")
        env["BLENDER_USER_SCRIPTS"] = os.path.join(BLENDER_RESOURCES, "scripts")
    return env


def build_blender_script(
    bid: str,
    outdir: str,
    base_pts2d: list[tuple[float, float]] | None,
    walls_pts2d: list[tuple[float, float]] | None,
    rect_xy: tuple[float, float] | None,
    base_h: float,
    wall_h: float,
    wall_t: float,
    overhang: float,
    ridge_above: float,
    export_glb: bool,
    render: bool,
) -> str:
    # camera helper
    if base_pts2d:
        xs = [p[0] for p in base_pts2d]
        ys = [p[1] for p in base_pts2d]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        size_x = maxx - minx
        size_y = maxy - miny
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
    else:
        assert rect_xy is not None
        size_x, size_y = rect_xy
        center_x, center_y = size_x / 2.0, size_y / 2.0

    cam_dist = max(size_x, size_y) * 2.2 + 8.0
    cam_z = base_h + wall_h + max(ridge_above, 1.0) + 6.0

    # literals for blender script
    base_literal = "None"
    if base_pts2d:
        base_literal = "[" + ", ".join(f"({x:.6f},{y:.6f})" for x, y in base_pts2d) + "]"

    walls_literal = "None"
    if walls_pts2d:
        walls_literal = "[" + ", ".join(f"({x:.6f},{y:.6f})" for x, y in walls_pts2d) + "]"

    rect_literal = "None"
    if rect_xy:
        rect_literal = f"({rect_xy[0]:.6f}, {rect_xy[1]:.6f})"

    script = f'''\
import bpy
import math
import os

# ---- reset scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def add_box(name, sx, sy, sz, x0=0.0, y0=0.0, z0=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0 + sx/2.0, y0 + sy/2.0, z0 + sz/2.0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx/2.0, sy/2.0, sz/2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj

def add_polygon_prism(name, pts2d, height, z0=0.0):
    verts_bottom = [(x, y, z0) for (x,y) in pts2d]
    verts_top = [(x, y, z0 + height) for (x,y) in pts2d]
    verts = verts_bottom + verts_top
    n = len(pts2d)

    faces = []
    faces.append(list(range(n-1, -1, -1)))       # bottom
    faces.append(list(range(n, 2*n)))            # top
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])

    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    tri = obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=tri.name)

    return obj

def _normalize(v):
    l = math.sqrt(v[0]*v[0] + v[1]*v[1])
    if l < 1e-9:
        return (0.0, 0.0)
    return (v[0]/l, v[1]/l)

def polygon_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n]
        a += (x1*y2 - x2*y1)
    return 0.5 * a

def _poly_area(pts):
    # signed area (shoelace). >0 => CCW, <0 => CW
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n]
        a += x1*y2 - x2*y1
    return 0.5 * a

def offset_polygon(pts, d):
    """
    Offset polygon outward by distance d, robust to CW/CCW winding.
    - Works best for simple mostly-convex polygons.
    - pts: [(x,y), ...] not closed
    """
    n = len(pts)
    if n < 3 or abs(d) < 1e-12:
        return list(pts)

    # Determine outward normal direction:
    # For CCW polygons, "left" normal points inward. For CW, left points outward.
    ccw = _poly_area(pts) > 0.0

    out = []
    for i in range(n):
        p0 = pts[(i-1) % n]
        p1 = pts[i]
        p2 = pts[(i+1) % n]

        # edge vectors
        e1 = (p1[0] - p0[0], p1[1] - p0[1])
        e2 = (p2[0] - p1[0], p2[1] - p1[1])

        # Choose outward normals:
        # left normal = (-y, x)
        # right normal = (y, -x)
        if ccw:
            n1 = _normalize(( e1[1], -e1[0]))  # RIGHT normal
            n2 = _normalize(( e2[1], -e2[0]))  # RIGHT normal
        else:
            n1 = _normalize((-e1[1],  e1[0]))  # LEFT normal
            n2 = _normalize((-e2[1],  e2[0]))  # LEFT normal

        # bisector (normalized)
        b = _normalize((n1[0] + n2[0], n1[1] + n2[1]))
        if b == (0.0, 0.0):
            b = n1

        # miter scale (clamped)
        dot = max(-0.999, min(0.999, b[0]*n1[0] + b[1]*n1[1]))
        scale = d / max(0.2, dot)
        out.append((p1[0] + b[0]*scale, p1[1] + b[1]*scale))
    return out
def add_pyramid_roof(name, pts2d, roof_h, z0):
    # roof slopes from all edges to a single apex (pyramid/hip roof)
    cx = sum(p[0] for p in pts2d) / len(pts2d)
    cy = sum(p[1] for p in pts2d) / len(pts2d)

    verts = [(x, y, z0) for (x, y) in pts2d]
    apex_i = len(verts)
    verts.append((cx, cy, z0 + roof_h))

    faces = []
    n = len(pts2d)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, apex_i])

    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    tri = obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=tri.name)

    return obj

BID = "{bid}"
OUTDIR = r"{outdir}"

BASE_PTS = {base_literal}
WALLS_PTS = {walls_literal}
RECT = {rect_literal}

BASE_H = {base_h:.6f}
WALL_H = {wall_h:.6f}
WALL_T = {wall_t:.6f}

OVERHANG = {overhang:.6f}
RIDGE_ABOVE = {ridge_above:.6f}

# ---- geometry ----
if BASE_PTS:
    base = add_polygon_prism(f"{{BID}}_base", BASE_PTS, BASE_H, z0=0.0)
else:
    sx, sy = RECT
    base = add_box(f"{{BID}}_base", sx, sy, BASE_H, z0=0.0)

if WALLS_PTS:
    walls = add_polygon_prism(f"{{BID}}_walls", WALLS_PTS, WALL_H, z0=BASE_H)
elif BASE_PTS:
    walls = add_polygon_prism(f"{{BID}}_walls", BASE_PTS, WALL_H, z0=BASE_H)
else:
    sx, sy = RECT
    walls = add_box(f"{{BID}}_walls", sx, sy, WALL_H, z0=BASE_H)

# ---- roof (pyramid/hip to apex) ----
roof_h = max(RIDGE_ABOVE, 0.25)
z0_roof = BASE_H + WALL_H

if BASE_PTS:
    d = OVERHANG if OVERHANG > 0 else 0.0
    roof_pts = offset_polygon(BASE_PTS, d) if d > 0 else BASE_PTS
    roof = add_pyramid_roof(f"{{BID}}_roof", roof_pts, roof_h, z0=z0_roof)
else:
    sx, sy = RECT
    rect_pts = [(0.0, 0.0), (sx, 0.0), (sx, sy), (0.0, sy)]
    d = OVERHANG if OVERHANG > 0 else 0.0
    roof_pts = offset_polygon(rect_pts, d) if d > 0 else rect_pts
    roof = add_pyramid_roof(f"{{BID}}_roof", roof_pts, roof_h, z0=z0_roof)

# ---- light & camera ----
bpy.ops.object.light_add(type='SUN', location=({center_x:.6f} + {cam_dist:.6f}, {center_y:.6f} - {cam_dist:.6f}, {cam_z:.6f}))
sun = bpy.context.active_object
sun.data.energy = 3.0

bpy.ops.object.camera_add(location=({center_x:.6f} + {cam_dist:.6f}, {center_y:.6f} - {cam_dist:.6f}, {cam_z:.6f}))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(60), 0, math.radians(45))
scene.camera = cam

scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.render.resolution_x = 1600
scene.render.resolution_y = 1000

# ---- save outputs ----
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTDIR, f"{{BID}}.blend"))
'''

    if export_glb:
        script += '''
bpy.ops.export_scene.gltf(
    filepath=os.path.join(OUTDIR, f"{BID}.glb"),
    export_format='GLB',
    export_yup=True
)
'''
    if render:
        script += '''
scene.render.filepath = os.path.join(OUTDIR, f"{BID}_preview.png")
bpy.ops.render.render(write_still=True)
'''
    return script


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path", help="Path to building YAML (e.g. buildings/atelje0_1.yaml)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--export-glb", action="store_true", help="Export GLB")
    ap.add_argument("--render", action="store_true", help="Render a preview PNG")
    args = ap.parse_args()

    data = load_yaml(args.yaml_path)
    b = data.get("building", data)
    bid = b.get("id", "building")

    fp = b.get("footprint", {})
    has_rect = isinstance(fp.get("outer"), dict) and ("x" in fp["outer"]) and ("y" in fp["outer"])
    has_base_poly = isinstance(fp.get("outer_polygon"), list) and len(fp["outer_polygon"]) >= 3
    has_walls_poly = isinstance(fp.get("walls_polygon"), list) and len(fp["walls_polygon"]) >= 3

    if not (has_rect or has_base_poly):
        print(f"SKIP: {bid} (no footprint in YAML)")
        return

    base_h = float(b.get("base", {}).get("height", 0.0))
    wall_h = float(b.get("walls", {}).get("height", 2.6))
    wall_t = float(b.get("walls", {}).get("thickness", 0.15))

    roof = b.get("roof", {})
    overhang = float(roof.get("overhang", 0.0))
    ridge_above = float(roof.get("ridge_height_above_walls", 0.5))

    base_pts2d = None
    walls_pts2d = None
    rect_xy = None

    if has_base_poly:
        base_pts2d = [(float(p[0]), float(p[1])) for p in fp["outer_polygon"]]
    if has_walls_poly:
        walls_pts2d = [(float(p[0]), float(p[1])) for p in fp["walls_polygon"]]
    if (not base_pts2d) and has_rect:
        rect_xy = (float(fp["outer"]["x"]), float(fp["outer"]["y"]))

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    script = build_blender_script(
        bid=bid,
        outdir=outdir,
        base_pts2d=base_pts2d,
        walls_pts2d=walls_pts2d,
        rect_xy=rect_xy,
        base_h=base_h,
        wall_h=wall_h,
        wall_t=wall_t,
        overhang=overhang,
        ridge_above=ridge_above,
        export_glb=args.export_glb,
        render=args.render,
    )

    debug_script_path = os.path.join(outdir, "_debug_last_blender_script.py")
    with open(debug_script_path, "w", encoding="utf-8") as df:
        df.write(script)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(script)
        temp_script_path = tf.name

    blender = find_blender()
    cmd = [blender, "--background", "--python", temp_script_path]

    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env=clean_blender_env())
    finally:
        try:
            os.unlink(temp_script_path)
        except OSError:
            pass

    print(f"Done. Outputs in: {outdir}", flush=True)


if __name__ == "__main__":
    main()
