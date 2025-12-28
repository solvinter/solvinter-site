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
    posts_per_segment: list[int] | None,
    post_size: float,
    corner_post_size: float,
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

    posts_per_segment_literal = 'None'
    if posts_per_segment:
        posts_per_segment_literal = str(list(posts_per_segment))

    rect_literal = "None"
    if rect_xy:
        rect_literal = f"({rect_xy[0]:.6f}, {rect_xy[1]:.6f})"

    script = f'''\
import bpy
import math
import mathutils
import os

# ---- reset scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ---- world lighting (never black) ----
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # white
    bg.inputs[1].default_value = 1.5                  # strength

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
POSTS_PER_SEGMENT = {posts_per_segment_literal}
POST_SIZE = {post_size:.6f}
CORNER_POST_SIZE = {corner_post_size:.6f}

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



# ---- posts (open south wall) ----
# POSTS_PER_SEGMENT: e.g. [4,3] => 4 posts along segment 0, 3 posts along segment 1
# POST_SIZE / CORNER_POST_SIZE are square side lengths in meters

def _roof_z_at_xy(x, y, base_pts2d, z0_roof, roof_h_in):
    """
    Approx: pyramid/hip roof to central apex.
    Height falls linearly with distance from centroid normalized by max centroid->vertex distance.
    Good enough visually for our use.
    """
    if not base_pts2d:
        return z0_roof + roof_h_in

    cx = sum(p[0] for p in base_pts2d) / len(base_pts2d)
    cy = sum(p[1] for p in base_pts2d) / len(base_pts2d)

    maxd = 0.0
    for (vx, vy) in base_pts2d:
        d = math.sqrt((vx - cx)**2 + (vy - cy)**2)
        if d > maxd:
            maxd = d
    if maxd < 1e-6:
        return z0_roof + roof_h_in

    dxy = math.sqrt((x - cx)**2 + (y - cy)**2)
    t = min(1.0, max(0.0, dxy / maxd))
    return z0_roof + roof_h_in * (1.0 - t)

def add_post(name, x, y, size, z0, h):
    sx = size
    sy = size
    sz = max(0.001, h)
    return add_box(name, sx, sy, sz, x0=x - sx/2.0, y0=y - sy/2.0, z0=z0)

def add_segment_posts(seg_name, p0, p1, count_total, size, z0, h, include_start=True, include_end=True):
    # count_total = totala stolpar längs segmentet (inkl endpoints)
    # include_start/include_end låter oss undvika dubbelstolpe vid knutpunkt (p1)

    out = []
    if count_total is None:
        return out

    try:
        count_total = int(count_total)
    except Exception:
        return out

    if count_total <= 0:
        return out

    # We want exactly count_total posts even if we skip start or end.
    # So we generate a slightly larger candidate set, filter, then trim.
    cand = count_total
    if not include_start:
        cand += 1
    if not include_end:
        cand += 1

    if cand <= 1:
        ts = [0.5]
    else:
        ts = [i / (cand - 1) for i in range(cand)]

    if not include_start:
        ts = [t for t in ts if t > 1e-9]
    if not include_end:
        ts = [t for t in ts if t < 1.0 - 1e-9]

    # Trim to exact count_total (should normally already match)
    if len(ts) > count_total:
        ts = ts[:count_total]

    for i, t in enumerate(ts):
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        out.append(add_post(f"{{seg_name}}_{{i:02d}}", x, y, size, z0=z0, h=h))
    return out

def add_open_south_posts():
    # south open wall posts (7 posts total: 4 on segment0, 3 on segment1)
    if not POSTS_PER_SEGMENT:
        return
    if not WALLS_PTS or len(WALLS_PTS) < 3:
        return

    roof_h_local = max(RIDGE_ABOVE, 0.25)
    z0_roof = BASE_H + WALL_H

    # Corner post at WALLS_PTS[0] (SW) - big one
    sw = WALLS_PTS[0]
    # corner post goes full wall height only
    add_post(f"{{BID}}_corner_SW", sw[0], sw[1], CORNER_POST_SIZE, z0=BASE_H, h=WALL_H)

    p0 = WALLS_PTS[0]
    p1 = WALLS_PTS[1]
    p2 = WALLS_PTS[2]

    c0 = POSTS_PER_SEGMENT[0] if len(POSTS_PER_SEGMENT) > 0 else 0
    c1 = POSTS_PER_SEGMENT[1] if len(POSTS_PER_SEGMENT) > 1 else 0

    # Segment 0: include both endpoints
    seg0_posts = add_segment_posts(f"{{BID}}_post_S0", p0, p1, c0, POST_SIZE, z0=BASE_H, h=WALL_H,
                                   include_start=True, include_end=True)

    # Segment 1: skip start endpoint (p1) to avoid duplicate with segment 0
    seg1_posts = add_segment_posts(f"{{BID}}_post_S1", p1, p2, c1, POST_SIZE, z0=BASE_H, h=WALL_H,
                                   include_start=False, include_end=True)
    print("DEBUG seg1 len:", len(seg1_posts))
    print("DEBUG seg1 len:", len(seg1_posts))

    # Extend each post to the roof as ONE solid post
    def extend_to_roof(objs):
        for obj in objs:
            x = obj.location.x
            y = obj.location.y
            z_roof = _roof_z_at_xy(x, y, BASE_PTS if BASE_PTS else [], z0_roof, roof_h_local)
            h_total = max(0.0, z_roof - BASE_H)
            name = obj.name
            bpy.data.objects.remove(obj, do_unlink=True)
            add_post(name, x, y, POST_SIZE, z0=BASE_H, h=h_total)

    extend_to_roof(seg0_posts)
    
    extend_to_roof(seg1_posts)
add_open_south_posts()

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
# Robust auto-frame camera + sun based on scene bounding box.
# This avoids black/white previews caused by wrong camera placement or clipping.

def _scene_bbox_mesh_only():
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        return None
    minx=miny=minz=1e18
    maxx=maxy=maxz=-1e18
    for o in meshes:
        for v in o.bound_box:
            w = o.matrix_world @ mathutils.Vector((v[0], v[1], v[2]))
            minx=min(minx, w.x); miny=min(miny, w.y); minz=min(minz, w.z)
            maxx=max(maxx, w.x); maxy=max(maxy, w.y); maxz=max(maxz, w.z)
    return (minx,miny,minz,maxx,maxy,maxz)

bb = _scene_bbox_mesh_only()
if bb:
    minx,miny,minz,maxx,maxy,maxz = bb
    cx = 0.5*(minx+maxx)
    cy = 0.5*(miny+maxy)
    cz = 0.5*(minz+maxz)
    dx = (maxx-minx)
    dy = (maxy-miny)
    dz = (maxz-minz)
    size = max(dx,dy,dz)
    if size < 0.001:
        size = 1.0
else:
    # fallback
    cx,cy,cz = (0.0,0.0,BASE_H + WALL_H*0.5)
    size = max(RECT[0], RECT[1], BASE_H + WALL_H + 1.0)

center = (cx, cy, cz)

# SUN (placed "behind" camera direction so it lights the building)
bpy.ops.object.light_add(type='SUN', location=(cx + 2.5*size, cy - 2.0*size, cz + 2.0*size))
sun = bpy.context.active_object
sun.data.energy = 3.0

# Camera position: diagonally above, pointing toward center
bpy.ops.object.camera_add(location=(cx + 2.2*size, cy - 2.2*size, cz + 1.2*size))
cam = bpy.context.active_object

# Ensure clipping doesn't cut the building
cam.data.clip_start = 0.01
cam.data.clip_end = max(1000.0, 20.0*size)

# Add tracking target
target = bpy.data.objects.new("CamTarget", None)
bpy.context.collection.objects.link(target)
target.location = center

con = cam.constraints.new(type='TRACK_TO')
con.target = target
con.track_axis = 'TRACK_NEGATIVE_Z'
con.up_axis = 'UP_Y'

scene.camera = cam

# Render settings
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.render.resolution_x = 1600
scene.render.resolution_y = 1000

# ---- save outputs ----

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTDIR, BID + ".blend"))
'''

    if export_glb:
        script += '''
bpy.ops.export_scene.gltf(
    filepath=os.path.join(OUTDIR, BID + ".glb"),
    export_format='GLB',
    export_yup=True
)
'''
    if render:
        script += '''
scene.render.filepath = os.path.join(OUTDIR, BID + "_preview.png")
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

    # optional open south wall posts (structure)
    south_cfg = (data.get("structure") or {}).get("south_open_wall") or {}
    posts_per_segment = south_cfg.get("posts_per_segment") or None
    post_size = float(south_cfg.get("post_size") or 0.09)
    corner_post_size = float(south_cfg.get("corner_post_size") or (post_size * 1.5))
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
        posts_per_segment=posts_per_segment,
        post_size=post_size,
        corner_post_size=corner_post_size,
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
