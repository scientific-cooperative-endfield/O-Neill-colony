"""Pistol, assault rifle, .338 sniper rifle (+ magazines, spent casing).

Model space (Blender): forward -Y, up +Z, the right-hand grip centre at the origin.
Sockets: grip_l (support hand), muzzle, sight (eye point for aiming down sights), eject.
Run: blender -b -P tools/blender/weapons.py -- <out.omdl> [preview.png]
"""
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(__file__))
from common import apply_all, array, bake_ao, bevel, cut, export_omdl, lathe, loft, preview, prim, reset, socket  # noqa: E402

R = math.radians


def slots(part, name, y0, z, count, step, w=0.09, h=0.009, ln=0.02):
    c = prim("cube", name, part, "polymer", loc=(0, y0, z), scale=(w, ln, h))
    array(c, count, (0, -step, 0))
    return c


def pistol():
    p = "pistol"
    slide = loft(p + "_slide", p, "gunmetal", [((0, 0.055, 0.047), 0.0145, 0.017, 0), ((0, -0.128, 0.047), 0.0145, 0.017, 0)])
    bevel(slide, 0.003, 2)
    ser = prim("cube", p + "_ser", p, "gunmetal", loc=(0, 0.045, 0.05), scale=(0.04, 0.002, 0.022))
    array(ser, 7, (0, -0.005, 0))
    cut(slide, ser)
    cut(slide, prim("cube", p + "_portcut", p, "gunmetal", loc=(-0.012, -0.02, 0.054), scale=(0.02, 0.04, 0.02)))
    prim("cyl", p + "_barrel", p, "metal", loc=(0, -0.02, 0.05), rot=(R(90), 0, 0), scale=(0.019, 0.019, 0.21), verts=16)
    bevel(loft(p + "_frame", p, "polymer", [((0, 0.03, 0.02), 0.0135, 0.012, 0), ((0, -0.125, 0.02), 0.0135, 0.012, 0)]), 0.003, 2)
    grip = loft(p + "_grip", p, "polymer", [((0, 0.012, 0.02), 0.0145, 0.026, 72), ((0, 0.022, -0.03), 0.0155, 0.028, 72), ((0, 0.034, -0.078), 0.015, 0.027, 72)])
    bevel(grip, 0.005, 2)
    tex = prim("cube", p + "_gripcut", p, "polymer", loc=(0, 0.0, -0.01), rot=(R(-18), 0, 0), scale=(0.04, 0.03, 0.0025))
    array(tex, 7, (0, 0.002, -0.009))
    cut(grip, tex)
    prim("cube", p + "_guard", p, "polymer", loc=(0, -0.03, -0.006), scale=(0.008, 0.05, 0.005))
    prim("cube", p + "_guardf", p, "polymer", loc=(0, -0.054, 0.006), rot=(R(-20), 0, 0), scale=(0.008, 0.005, 0.026))
    prim("cube", p + "_trigger", p, "metal", loc=(0, -0.025, 0.006), rot=(R(15), 0, 0), scale=(0.005, 0.005, 0.02))
    prim("cube", p + "_rsight", p, "gunmetal", loc=(0, 0.045, 0.067), scale=(0.02, 0.008, 0.008))
    prim("cube", p + "_fsight", p, "gunmetal", loc=(0, -0.12, 0.067), scale=(0.005, 0.006, 0.008))
    for dx in (-0.006, 0.006):
        prim("sphere", p + "_dot", p, "emit_green", loc=(dx, 0.0491, 0.069), scale=(0.0025, 0.0025, 0.0025), seg=8, rings=4)
    prim("sphere", p + "_fdot", p, "emit_green", loc=(0, -0.1235, 0.069), scale=(0.0025, 0.0025, 0.0025), seg=8, rings=4)
    m = "pistol_mag"
    prim("cube", m + "_base", m, "polymer", loc=(0, 0.036, -0.085), rot=(R(-18), 0, 0), scale=(0.034, 0.06, 0.012))
    loft(m + "_body", m, "metal", [((0, 0.012, 0.015), 0.011, 0.02, 72), ((0, 0.033, -0.08), 0.011, 0.02, 72)])
    socket(p, "grip_l", (0.012, 0.02, -0.035))
    socket(p, "muzzle", (0, -0.132, 0.05))
    socket(p, "sight", (0, 0.06, 0.068))
    socket(p, "eject", (-0.016, -0.02, 0.055), fwd=(-1, 0, 0))


def rifle():
    p = "rifle"
    z = 0.085
    up = loft(p + "_upper", p, "gunmetal", [((0, 0.17, z), 0.024, 0.028, 0), ((0, -0.125, z), 0.024, 0.028, 0)])
    bevel(up, 0.004, 2)
    cut(up, prim("cube", p + "_portcut", p, "gunmetal", loc=(-0.024, 0.03, z), scale=(0.012, 0.065, 0.02)))
    bevel(loft(p + "_lower", p, "gunmetal", [((0, 0.13, 0.04), 0.022, 0.026, 0), ((0, -0.105, 0.04), 0.022, 0.026, 0)]), 0.004, 2)
    bevel(prim("cube", p + "_well", p, "gunmetal", loc=(0, -0.075, 0.005), scale=(0.042, 0.075, 0.05)), 0.004, 2)
    grip = loft(p + "_grip", p, "polymer", [((0, -0.005, 0.03), 0.016, 0.024, 70), ((0, 0.012, -0.03), 0.017, 0.026, 70), ((0, 0.032, -0.085), 0.016, 0.025, 68)])
    bevel(grip, 0.006, 2)
    prim("cube", p + "_guard", p, "gunmetal", loc=(0, -0.04, -0.012), scale=(0.01, 0.065, 0.005))
    prim("cube", p + "_trigger", p, "metal", loc=(0, -0.03, 0.0), rot=(R(15), 0, 0), scale=(0.005, 0.005, 0.022))
    hg = prim("cyl", p + "_handguard", p, "polymer_tan", loc=(0, -0.265, z), rot=(R(90), 0, 0), scale=(0.064, 0.064, 0.29), verts=8)
    bevel(hg, 0.003, 1)
    cut(hg, slots(p, p + "_slots", -0.15, z, 7, 0.035))
    cut(hg, slots(p, p + "_slots2", -0.15, z - 0.02, 7, 0.035, w=0.014, h=0.09))
    prim("cube", p + "_rail", p, "gunmetal", loc=(0, -0.12, 0.122), scale=(0.022, 0.56, 0.008))
    array(prim("cube", p + "_ridges", p, "gunmetal", loc=(0, 0.15, 0.128), scale=(0.024, 0.005, 0.005)), 46, (0, -0.012, 0))
    prim("cyl", p + "_barrel", p, "gunmetal", loc=(0, -0.47, z), rot=(R(90), 0, 0), scale=(0.018, 0.018, 0.17), verts=16)
    bevel(prim("cube", p + "_gasblock", p, "gunmetal", loc=(0, -0.42, z + 0.008), scale=(0.024, 0.02, 0.03)), 0.003, 1)
    brake = prim("cyl", p + "_brake", p, "gunmetal", loc=(0, -0.575, z), rot=(R(90), 0, 0), scale=(0.026, 0.026, 0.055), verts=16)
    bevel(brake, 0.002, 1)
    bs = prim("cube", p + "_brakecut", p, "gunmetal", loc=(0, -0.562, z + 0.012), scale=(0.04, 0.006, 0.012))
    array(bs, 3, (0, -0.012, 0))
    cut(brake, bs)
    prim("cyl", p + "_tube", p, "gunmetal", loc=(0, 0.26, 0.075), rot=(R(90), 0, 0), scale=(0.03, 0.03, 0.18), verts=16)
    stock = loft(p + "_stock", p, "polymer_tan", [((0, 0.19, 0.07), 0.021, 0.03, 0), ((0, 0.3, 0.06), 0.022, 0.045, 0), ((0, 0.395, 0.045), 0.022, 0.068, 0)])
    bevel(stock, 0.008, 3)
    cut(stock, prim("cube", p + "_stockcut", p, "polymer", loc=(0, 0.31, 0.02), scale=(0.06, 0.07, 0.03)))
    bevel(prim("cube", p + "_buttpad", p, "rubber", loc=(0, 0.405, 0.045), scale=(0.046, 0.016, 0.14)), 0.006, 2)
    prim("cube", p + "_charge", p, "gunmetal", loc=(0, 0.175, z + 0.02), scale=(0.05, 0.012, 0.008))
    # holographic sight
    bevel(prim("cube", p + "_holobase", p, "gunmetal", loc=(0, -0.01, 0.137), scale=(0.034, 0.085, 0.012)), 0.003, 1)
    hood = prim("cube", p + "_hood", p, "gunmetal", loc=(0, -0.015, 0.168), scale=(0.046, 0.065, 0.052))
    bevel(hood, 0.006, 2)
    cut(hood, prim("cube", p + "_hoodcut", p, "gunmetal", loc=(0, -0.015, 0.17), scale=(0.036, 0.09, 0.04)))
    prim("cube", p + "_glass", p, "glass", loc=(0, -0.035, 0.17), scale=(0.036, 0.002, 0.04))
    prim("torus", p + "_reticle", p, "emit_red", loc=(0, -0.037, 0.17), rot=(R(90), 0, 0), R=0.0045, r=0.0007, seg=16, mseg=4)
    prim("sphere", p + "_dot", p, "emit_red", loc=(0, -0.037, 0.17), scale=(0.0012, 0.0012, 0.0012), seg=8, rings=4)
    prim("cube", p + "_sling", p, "metal", loc=(0.026, 0.2, 0.05), scale=(0.004, 0.02, 0.012))
    m = "rifle_mag"
    mag = loft(
        m + "_body",
        m,
        "polymer",
        [((0, -0.075, 0.0), 0.0125, 0.034, 90), ((0, -0.082, -0.07), 0.0125, 0.034, 82), ((0, -0.096, -0.14), 0.0125, 0.034, 74), ((0, -0.116, -0.2), 0.0125, 0.034, 66)],
    )
    bevel(mag, 0.003, 1)
    ribs = prim("cube", m + "_ribs", m, "polymer", loc=(0, -0.082, -0.07), scale=(0.03, 0.004, 0.004))
    array(ribs, 5, (0, -0.004, -0.02))
    cut(mag, ribs)
    bevel(prim("cube", m + "_base", m, "polymer", loc=(0, -0.12, -0.205), rot=(R(-24), 0, 0), scale=(0.03, 0.074, 0.012)), 0.003, 1)
    socket(p, "grip_l", (0, -0.27, 0.05))
    socket(p, "muzzle", (0, -0.605, z))
    socket(p, "sight", (0, 0.06, 0.17))
    socket(p, "eject", (-0.028, 0.03, z), fwd=(-1, 0, 0))


def sniper():
    p = "sniper"
    z = 0.075
    bevel(prim("cyl", p + "_receiver", p, "gunmetal", loc=(0, -0.02, z), rot=(R(90), 0, 0), scale=(0.044, 0.044, 0.26), verts=24), 0.004, 2)
    barrel = lathe(p + "_barrel", p, "gunmetal", [(0.0, 0.14), (0.017, 0.14), (0.016, 0.3), (0.013, 0.6), (0.012, 0.76), (0.0, 0.76)], axis="Y", seg=20, loc=(0, 0, z))
    cut(barrel, prim("cube", p + "_flutes", p, "gunmetal", loc=(0, -0.42, z + 0.016), scale=(0.006, 0.26, 0.006)))
    brake = prim("cube", p + "_brake", p, "gunmetal", loc=(0, -0.8, z), scale=(0.04, 0.085, 0.036))
    bevel(brake, 0.006, 2)
    bc = prim("cube", p + "_brakecut", p, "gunmetal", loc=(0, -0.775, z), scale=(0.06, 0.01, 0.024))
    array(bc, 3, (0, -0.02, 0))
    cut(brake, bc)
    hg = prim("cyl", p + "_handguard", p, "gunmetal", loc=(0, -0.33, z - 0.004), rot=(R(90), 0, 0), scale=(0.07, 0.07, 0.38), verts=8)
    bevel(hg, 0.003, 1)
    cut(hg, slots(p, p + "_slots", -0.19, z - 0.004, 9, 0.034, w=0.1, h=0.012))
    bevel(loft(p + "_chassis", p, "polymer_tan", [((0, 0.13, 0.035), 0.025, 0.03, 0), ((0, -0.15, 0.035), 0.026, 0.03, 0)]), 0.006, 2)
    grip = loft(p + "_grip", p, "polymer_tan", [((0, -0.0, 0.02), 0.017, 0.025, 72), ((0, 0.015, -0.035), 0.018, 0.026, 72), ((0, 0.032, -0.085), 0.017, 0.025, 70)])
    bevel(grip, 0.006, 2)
    stock = loft(
        p + "_stock", p, "polymer_tan", [((0, 0.12, 0.05), 0.024, 0.04, 0), ((0, 0.26, 0.03), 0.024, 0.07, 0), ((0, 0.45, 0.02), 0.024, 0.085, 0), ((0, 0.48, 0.02), 0.024, 0.085, 0)]
    )
    bevel(stock, 0.009, 3)
    cut(stock, prim("cube", p + "_thumbhole", p, "polymer", loc=(0, 0.3, 0.0), rot=(R(-10), 0, 0), scale=(0.08, 0.13, 0.05)))
    bevel(prim("cube", p + "_cheek", p, "polymer", loc=(0, 0.33, 0.115), scale=(0.04, 0.17, 0.02)), 0.007, 2)
    bevel(prim("cube", p + "_buttpad", p, "rubber", loc=(0, 0.49, 0.02), scale=(0.05, 0.02, 0.18)), 0.007, 2)
    prim("cyl", p + "_monopod", p, "metal", loc=(0, 0.42, -0.07), scale=(0.012, 0.012, 0.07), verts=10)
    prim("cyl", p + "_bolt", p, "metal", loc=(-0.04, 0.05, z), rot=(0, R(90), 0), scale=(0.012, 0.012, 0.05), verts=10)
    prim("sphere", p + "_knob", p, "gunmetal", loc=(-0.068, 0.05, z - 0.008), scale=(0.024, 0.024, 0.024), seg=12, rings=6)
    prim("cube", p + "_guard", p, "gunmetal", loc=(0, -0.035, -0.008), scale=(0.01, 0.06, 0.005))
    prim("cube", p + "_trigger", p, "metal", loc=(0, -0.025, 0.004), rot=(R(15), 0, 0), scale=(0.005, 0.005, 0.02))
    for dx in (0.012, -0.012):
        prim("cyl", p + "_bipod", p, "gunmetal", loc=(dx, -0.5, z - 0.04), rot=(R(90), 0, 0), scale=(0.011, 0.011, 0.2), verts=8)
    zs = 0.145
    lathe(
        p + "_scope",
        p,
        "gunmetal",
        [(0.0, -0.12), (0.019, -0.12), (0.021, -0.1), (0.017, -0.07), (0.015, -0.05), (0.015, 0.14), (0.026, 0.19), (0.027, 0.24), (0.0, 0.24)],
        axis="Y",
        seg=32,
        loc=(0, 0, zs),
    )
    lathe(p + "_lens_r", p, "glass", [(0.0, -0.1205), (0.016, -0.1205), (0.0, -0.1203)], axis="Y", seg=24, loc=(0, 0, zs))
    lathe(p + "_lens_f", p, "glass", [(0.0, 0.2405), (0.024, 0.2405), (0.0, 0.2403)], axis="Y", seg=24, loc=(0, 0, zs))
    prim("cyl", p + "_turret_t", p, "gunmetal", loc=(0, -0.04, zs + 0.022), scale=(0.024, 0.024, 0.022), verts=16)
    prim("cyl", p + "_turret_s", p, "gunmetal", loc=(0.022, -0.04, zs), rot=(0, R(90), 0), scale=(0.022, 0.022, 0.02), verts=16)
    for y in (-0.08, 0.04):
        prim("torus", p + "_ring", p, "gunmetal", loc=(0, y, zs), rot=(R(90), 0, 0), R=0.0175, r=0.004, seg=24, mseg=6)
        prim("cube", p + "_ringbase", p, "gunmetal", loc=(0, y, zs - 0.033), scale=(0.022, 0.012, 0.03))
    prim("cube", p + "_rail", p, "gunmetal", loc=(0, -0.02, 0.11), scale=(0.022, 0.24, 0.008))
    bevel(prim("cube", "sniper_mag_body", "sniper_mag", "polymer", loc=(0, -0.09, -0.005), scale=(0.034, 0.085, 0.07)), 0.004, 2)
    socket(p, "grip_l", (0, -0.3, 0.035))
    socket(p, "muzzle", (0, -0.845, z))
    socket(p, "sight", (0, 0.14, zs))
    socket(p, "eject", (-0.04, -0.03, z + 0.01), fwd=(-1, 0, 0))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = argv[0] if argv else "assets/models/weapons.omdl"
    reset()
    pistol()
    rifle()
    sniper()
    lathe("casing", "casing", "brass", [(0.0, -0.022), (0.0048, -0.022), (0.0048, 0.01), (0.0032, 0.016), (0.003, 0.022), (0.0, 0.022)], seg=10)
    apply_all()
    bake_ao(lambda o: o["part"], distance=0.03)
    export_omdl(out, {})
    if len(argv) > 1:
        offs = {"pistol": 0.35, "pistol_mag": 0.35, "sniper": -0.35, "sniper_mag": -0.35, "casing": 0.6}
        for o in bpy.data.objects:
            o.location.x += offs.get(o.get("part", ""), 0.0)
            o.location.z += 0.2
        preview(argv[1], (0.1, -0.15, 0.25), 1.9, elev=25, azim=-70, lens=50, size=(1200, 800))


main()
