"""Shared helpers for the procedural model build (Blender 4.2, run headless).

Conventions (Blender side): metres, Z up, the model faces -Y, its left side is +X.
The exporter converts to the game's frame: X = left, Y = up, Z = forward
(x, y, z)_game = (x, z, -y)_blender — a proper rotation, so winding is kept.
"""
import math
import struct

import bmesh
import bpy
from mathutils import Matrix, Vector

# name: (linear albedo, metal, roughness, flags, emissive)
# flags: 1 = multiply by team colour, 2 = emissive, 4 = visor (strong env reflection)
MATS = {
    "fabric": ((0.13, 0.14, 0.13), 0.0, 0.85, 0, 0.0),
    "fabric_dark": ((0.07, 0.075, 0.08), 0.0, 0.8, 0, 0.0),
    "armor_team": ((0.80, 0.80, 0.80), 0.05, 0.38, 1, 0.0),
    "armor_white": ((0.78, 0.79, 0.80), 0.05, 0.35, 0, 0.0),
    "armor_grey": ((0.22, 0.235, 0.25), 0.25, 0.45, 0, 0.0),
    "metal": ((0.62, 0.62, 0.64), 1.0, 0.28, 0, 0.0),
    "gunmetal": ((0.045, 0.047, 0.05), 0.7, 0.42, 0, 0.0),
    "polymer": ((0.03, 0.032, 0.035), 0.0, 0.62, 0, 0.0),
    "polymer_tan": ((0.30, 0.25, 0.17), 0.0, 0.65, 0, 0.0),
    "rubber": ((0.02, 0.02, 0.02), 0.0, 0.92, 0, 0.0),
    "visor": ((1.0, 0.72, 0.32), 1.0, 0.06, 4, 0.0),
    "glass": ((0.10, 0.16, 0.22), 1.0, 0.04, 4, 0.0),
    "brass": ((0.85, 0.58, 0.25), 1.0, 0.3, 0, 0.0),
    "emit_team": ((1.0, 1.0, 1.0), 0.0, 0.5, 3, 4.0),
    "emit_cyan": ((0.25, 0.85, 1.0), 0.0, 0.5, 2, 5.0),
    "emit_red": ((1.0, 0.08, 0.04), 0.0, 0.5, 2, 12.0),
    "emit_green": ((0.3, 1.0, 0.35), 0.0, 0.5, 2, 6.0),
}

_mat_cache = {}


def mat(name):
    if name in _mat_cache:
        return _mat_cache[name]
    col, metal, rough, flags, em = MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    preview = (0.15, 0.45, 0.95) if flags & 1 else col
    bsdf.inputs["Base Color"].default_value = (*preview, 1.0)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    if flags & 2:
        bsdf.inputs["Emission Color"].default_value = (*(preview if flags & 1 else col), 1.0)
        bsdf.inputs["Emission Strength"].default_value = em
    _mat_cache[name] = m
    return m


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _mat_cache.clear()


def _finish(o, name, part, material, smooth_angle):
    o.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat(material))
    o["part"] = part
    o["smooth"] = smooth_angle
    return o


def prim(kind, name, part, material, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), smooth=35.0, **kw):
    """Primitive with unit size, then scaled (scale applied)."""
    bpy.ops.object.select_all(action="DESELECT")
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    elif kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(vertices=kw.get("verts", 24), radius=0.5, depth=1.0, location=loc, rotation=rot)
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=kw.get("verts", 24), radius1=kw["r1"], radius2=kw["r2"], depth=1.0, location=loc, rotation=rot)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=kw.get("seg", 32), ring_count=kw.get("rings", 16), radius=0.5, location=loc, rotation=rot)
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=kw["R"], minor_radius=kw["r"], major_segments=kw.get("seg", 32), minor_segments=kw.get("mseg", 10), location=loc, rotation=rot
        )
    else:
        raise ValueError(kind)
    o = bpy.context.active_object
    o.scale = scale
    return _finish(o, name, part, material, smooth)


def from_bmesh(bm, name, part, material, smooth=35.0):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    return _finish(o, name, part, material, smooth)


def loft(name, part, material, sections, smooth=35.0, cap=True):
    """Tube through rectangular sections [(centre, half_x, half_z, tilt_x_deg)].
    Each section is a rectangle in the local XZ plane, rotated about X by tilt
    (tilt 90 → horizontal rectangle, for vertical sweeps)."""
    bm = bmesh.new()
    rings = []
    for c, hx, hz, tilt in sections:
        r = Matrix.Rotation(math.radians(tilt), 3, "X")
        ring = []
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            ring.append(bm.verts.new(Vector(c) + r @ Vector((sx * hx, 0.0, sz * hz))))
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((a[i], a[j], b[j], b[i]))
    if cap:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return from_bmesh(bm, name, part, material, smooth)


def lathe(name, part, material, profile, axis="Z", seg=24, loc=(0, 0, 0), smooth=40.0):
    """Surface of revolution of [(radius, height)] around the local axis (Y: height grows towards -Y)."""
    bm = bmesh.new()
    rings = []
    for r, h in profile:
        ring = []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            ring.append(bm.verts.new(Vector((r * math.cos(a), r * math.sin(a), h))))
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for i in range(seg):
            j = (i + 1) % seg
            bm.faces.new((a[i], a[j], b[j], b[i]))
    if profile[0][0] > 1e-5:
        bm.faces.new(list(reversed(rings[0])))
    if profile[-1][0] > 1e-5:
        bm.faces.new(rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    rot = {"Z": Matrix.Identity(3), "Y": Matrix.Rotation(math.radians(90), 3, "X"), "X": Matrix.Rotation(math.radians(90), 3, "Y")}[axis]
    bmesh.ops.transform(bm, matrix=rot.to_4x4(), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=bm.verts)
    return from_bmesh(bm, name, part, material, smooth)


def bevel(o, width, seg=2, angle=40.0):
    m = o.modifiers.new("bevel", "BEVEL")
    m.width = width
    m.segments = seg
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(angle)
    m.harden_normals = False
    return o


def subsurf(o, levels=1):
    m = o.modifiers.new("subsurf", "SUBSURF")
    m.levels = levels
    m.render_levels = levels
    return o


def cut(o, cutter):
    """Boolean difference; the cutter is deleted after applying."""
    m = o.modifiers.new("cut", "BOOLEAN")
    m.operation = "DIFFERENCE"
    m.solver = "EXACT"
    m.object = cutter
    cutter.hide_render = True
    cutter.hide_viewport = True
    cutter["cutter"] = True
    return o


def array(o, count, offset):
    m = o.modifiers.new("array", "ARRAY")
    m.count = count
    m.use_relative_offset = False
    m.use_constant_offset = True
    m.constant_offset_displace = offset
    return o


def squash(o, sx=1.0, sy=1.0, sz=1.0):
    """Non-uniform scale about the object origin, applied."""
    o.scale = (sx, sy, sz)
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o


def socket(model, name, loc, fwd=(0, -1, 0), up=(0, 0, 1)):
    e = bpy.data.objects.new(f"socket_{model}_{name}", None)
    e.location = loc
    e["socket"] = name
    e["model"] = model
    e["fwd"] = fwd
    e["up"] = up
    bpy.context.collection.objects.link(e)
    return e


def apply_all():
    """Apply modifiers (booleans, bevels, …) on every mesh, drop cutters, set smoothing."""
    for o in list(bpy.data.objects):
        if o.get("cutter") or o.type != "MESH":
            continue
        bpy.ops.object.select_all(action="DESELECT")
        o.hide_viewport = False
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        for m in list(o.modifiers):
            bpy.ops.object.modifier_apply(modifier=m.name)
    for o in list(bpy.data.objects):
        if o.get("cutter"):
            bpy.data.objects.remove(o)
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.data.shade_smooth()
            o.data.set_sharp_from_angle(angle=math.radians(o.get("smooth", 35.0)))


def bake_ao(parts_of, samples=48, distance=0.06):
    """Bake ambient occlusion into an 'ao' colour attribute, part by part (so limbs
    hanging next to the torso in the rest pose do not darken each other)."""
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.world = scn.world or bpy.data.worlds.new("w")
    scn.world.light_settings.distance = distance
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for o in meshes:
        if "ao" not in o.data.color_attributes:
            o.data.color_attributes.new("ao", "BYTE_COLOR", "CORNER")
        o.data.color_attributes.active_color = o.data.color_attributes["ao"]
    groups = {}
    for o in meshes:
        groups.setdefault(parts_of(o), []).append(o)
    for objs in groups.values():
        for o in meshes:
            o.hide_render = o not in objs
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.bake(type="AO", target="VERTEX_COLORS")
    for o in meshes:
        o.hide_render = False


def _to_game(v):
    return (v[0], v[2], -v[1])


def export_omdl(path, pivots):
    """Writes every tagged mesh object, grouped by its 'part', relative to the part pivot.

    File: b"OMDL" u32 version, u32 parts, then per part:
      u16 len + name, f32 radius, u32 nv, u32 ni, nv × (pos 3f, normal 3f, rgba8 sqrt(albedo)+ao, rgba8 metal/rough/flags/emit×16), ni × u32
    then u32 sockets, each: u16 len + model, u16 len + name, pos 3f, fwd 3f, up 3f (model space, game axes).
    """
    dg = bpy.context.evaluated_depsgraph_get()
    parts = {}
    for o in bpy.data.objects:
        if o.type != "MESH" or o.get("cutter") or "part" not in o:
            continue
        parts.setdefault(o["part"], []).append(o)
    out = bytearray(b"OMDL") + struct.pack("<II", 1, len(parts))
    total = 0
    for part in sorted(parts):
        piv = Vector(pivots.get(part, (0, 0, 0)))
        verts = []
        index = {}
        idx = []
        for o in parts[part]:
            eo = o.evaluated_get(dg)
            me = eo.to_mesh()
            me.calc_loop_triangles()
            mw = o.matrix_world
            nm = mw.to_3x3().inverted().transposed()
            ao = me.color_attributes.get("ao")
            normals = me.corner_normals
            for tri in me.loop_triangles:
                mname = o.material_slots[tri.material_index].material.name if o.material_slots else "fabric"
                col, metal, rough, flags, em = MATS[mname.split(".")[0]]
                for li, vi in zip(tri.loops, tri.vertices):
                    p = mw @ me.vertices[vi].co - piv
                    n = (nm @ normals[li].vector).normalized()
                    a = 1.0 if ao is None else 0.25 + 0.75 * ao.data[li].color[0]
                    key = (round(p.x, 5), round(p.y, 5), round(p.z, 5), round(n.x, 3), round(n.y, 3), round(n.z, 3), mname, round(a, 2))
                    if key not in index:
                        index[key] = len(verts)
                        c8 = [min(255, int(math.sqrt(min(1.0, x)) * 255 + 0.5)) for x in col]
                        verts.append(
                            struct.pack("<3f3f4B4B", *_to_game(p), *_to_game(n), c8[0], c8[1], c8[2], int(a * 255), int(metal * 255), int(rough * 255), flags, min(255, int(em * 16)))
                        )
                    idx.append(index[key])
            eo.to_mesh_clear()
        radius = 0.0
        for v in verts:
            x, y, z = struct.unpack_from("<3f", v)
            radius = max(radius, math.sqrt(x * x + y * y + z * z))
        nb = part.encode()
        out += struct.pack("<H", len(nb)) + nb + struct.pack("<fII", radius, len(verts), len(idx))
        for v in verts:
            out += v
        out += struct.pack(f"<{len(idx)}I", *idx)
        total += len(idx) // 3
        print(f"  {part:16s} {len(verts):6d} v {len(idx) // 3:6d} tris  r={radius:.2f}")
    socks = [o for o in bpy.data.objects if o.get("socket")]
    out += struct.pack("<I", len(socks))
    for s in socks:
        for t in (s["model"], s["socket"]):
            b = t.encode()
            out += struct.pack("<H", len(b)) + b
        out += struct.pack("<3f3f3f", *_to_game(s.location), *_to_game(Vector(s["fwd"])), *_to_game(Vector(s["up"])))
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path}: {len(parts)} parts, {total} tris, {len(socks)} sockets, {len(out) / 1024:.0f} KiB")


def preview(path, target, dist, elev=12.0, azim=-35.0, size=(900, 900), lens=50.0, samples=48):
    """Quick Cycles still for checking the models (studio light, grey floor)."""
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    scn.render.resolution_x, scn.render.resolution_y = size
    world = scn.world or bpy.data.worlds.new("w")
    scn.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.45, 0.5, 0.58, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    t = Vector(target)
    a, e = math.radians(azim), math.radians(elev)
    cam_pos = t + Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cam = bpy.data.objects.new("cam", cd)
    scn.collection.objects.link(cam)
    cam.location = cam_pos
    cam.rotation_euler = (t - cam_pos).to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam
    for name, loc, energy, size_ in (("key", (2.5, -3.0, 4.0), 900, 2.0), ("rim", (-3.0, 3.0, 3.0), 600, 2.0), ("fill", (-3.5, -2.0, 1.5), 250, 3.0)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy * dist * dist / 9.0
        ld.size = size_
        lo = bpy.data.objects.new(name, ld)
        scn.collection.objects.link(lo)
        lo.location = t + Vector(loc) * (dist / 3.0)
        lo.rotation_euler = (t - lo.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.mesh.primitive_plane_add(size=40, location=(t.x, t.y, 0.0))
    fl = bpy.context.active_object
    fm = bpy.data.materials.new("floor")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.31, 0.33, 1)
    fl.data.materials.append(fm)
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
