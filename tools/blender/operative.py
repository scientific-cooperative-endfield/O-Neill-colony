"""Operative in a pressurised combat suit: rigid parts, one per bone.

Rest pose (Blender coords: Z up, faces -Y, left = +X); legs and arms hang straight down.
Run: blender -b -P tools/blender/operative.py -- <out.omdl> [preview.png]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import apply_all, bake_ao, bevel, cut, export_omdl, lathe, loft, preview, prim, reset, squash, subsurf  # noqa: E402

HIP = (0.1, 0.0, 0.93)
KNEE_Z, ANKLE_Z = 0.50, 0.085
SH = (0.2, 0.0, 1.42)
ELBOW_Z, WRIST_Z = 1.13, 0.87

PIVOTS = {"pelvis": (0, 0, 0.95), "torso": (0, 0, 1.0), "head": (0, 0, 1.5)}
for s, n in ((1, "L"), (-1, "R")):
    PIVOTS[f"thigh_{n}"] = (s * HIP[0], 0, HIP[2])
    PIVOTS[f"shin_{n}"] = (s * HIP[0], 0, KNEE_Z)
    PIVOTS[f"foot_{n}"] = (s * HIP[0], 0, ANKLE_Z)
    PIVOTS[f"upperarm_{n}"] = (s * SH[0], 0, SH[2])
    PIVOTS[f"forearm_{n}"] = (s * SH[0], 0, ELBOW_Z)
    PIVOTS[f"hand_{n}"] = (s * SH[0], 0, WRIST_Z)


def leg(s, n):
    x = s * HIP[0]
    p = f"thigh_{n}"
    lathe(p + "_suit", p, "fabric", [(0.074, 0.96), (0.088, 0.9), (0.086, 0.78), (0.072, 0.62), (0.063, 0.52), (0.058, 0.48)], loc=(x, 0, 0))
    bevel(prim("cube", p + "_plate", p, "armor_team", loc=(x + s * 0.012, -0.062, 0.76), rot=(math.radians(-4), 0, 0), scale=(0.115, 0.04, 0.21)), 0.016, 3)
    bevel(prim("cube", p + "_pouch", p, "fabric_dark", loc=(x + s * 0.085, 0.0, 0.73), scale=(0.035, 0.1, 0.13)), 0.012, 2)
    prim("torus", p + "_strap", p, "polymer", loc=(x, 0, 0.84), scale=(1, 1, 1.6), R=0.087, r=0.007, mseg=6)
    prim("torus", p + "_strap2", p, "polymer", loc=(x, 0, 0.66), scale=(1, 1, 1.6), R=0.074, r=0.007, mseg=6)
    p = f"shin_{n}"
    lathe(p + "_suit", p, "fabric", [(0.06, 0.53), (0.066, 0.44), (0.069, 0.36), (0.057, 0.24), (0.05, 0.15), (0.052, 0.1)], loc=(x, 0, 0))
    bevel(prim("sphere", p + "_kneepad", p, "armor_team", loc=(x, -0.045, 0.5), scale=(0.12, 0.085, 0.13), seg=24, rings=12), 0.004, 1)
    bevel(loft(p + "_guard", p, "armor_grey", [((x, -0.05, 0.43), 0.045, 0.018, 90), ((x, -0.058, 0.32), 0.048, 0.02, 92), ((x, -0.05, 0.19), 0.04, 0.016, 95)]), 0.01, 2)
    p = f"foot_{n}"
    boot = loft(
        p + "_boot",
        p,
        "fabric_dark",
        [((x, 0.07, 0.085), 0.05, 0.07, 0), ((x, 0.0, 0.085), 0.058, 0.08, 0), ((x, -0.09, 0.06), 0.058, 0.05, 0), ((x, -0.165, 0.048), 0.048, 0.036, 0)],
    )
    bevel(boot, 0.02, 3, angle=30)
    bevel(prim("cube", p + "_sole", p, "rubber", loc=(x, -0.045, 0.012), scale=(0.12, 0.29, 0.024)), 0.01, 2)
    prim("cube", p + "_magplate", p, "metal", loc=(x, -0.03, 0.0), scale=(0.08, 0.16, 0.006))
    prim("cyl", p + "_cuff", p, "armor_grey", loc=(x, 0.0, 0.15), scale=(0.118, 0.112, 0.05), verts=20)
    bevel(prim("cube", p + "_toecap", p, "armor_team", loc=(x, -0.14, 0.06), scale=(0.095, 0.07, 0.05)), 0.014, 3)


def arm(s, n):
    x = s * SH[0]
    p = f"upperarm_{n}"
    lathe(p + "_suit", p, "fabric", [(0.055, 1.46), (0.064, 1.4), (0.06, 1.28), (0.052, 1.16), (0.049, 1.12)], loc=(x, 0, 0))
    pad = prim("sphere", p + "_pauldron", p, "armor_team", loc=(x + s * 0.02, 0, 1.415), scale=(0.15, 0.16, 0.12), seg=24, rings=12)
    cut(pad, prim("cube", p + "_padcut", p, "fabric", loc=(x + s * 0.02, 0, 1.33), scale=(0.3, 0.3, 0.1)))
    bevel(prim("cube", p + "_band", p, "armor_grey", loc=(x + s * 0.035, 0, 1.3), scale=(0.03, 0.08, 0.06)), 0.008, 2)
    p = f"forearm_{n}"
    lathe(p + "_suit", p, "fabric", [(0.05, 1.14), (0.054, 1.08), (0.047, 0.96), (0.04, 0.88), (0.038, 0.86)], loc=(x, 0, 0))
    bevel(prim("cyl", p + "_vambrace", p, "armor_grey", loc=(x, 0, 0.98), scale=(0.112, 0.112, 0.15), verts=20), 0.012, 2)
    prim("sphere", p + "_elbow", p, "armor_team", loc=(x, 0.03, 1.13), scale=(0.08, 0.07, 0.08), seg=16, rings=8)
    if s > 0:  # wrist computer on the left arm
        bevel(prim("cube", p + "_pad", p, "polymer", loc=(x, -0.05, 0.98), scale=(0.06, 0.03, 0.09)), 0.006, 2)
        prim("cube", p + "_screen", p, "emit_cyan", loc=(x, -0.066, 0.98), scale=(0.045, 0.004, 0.065))
    else:
        prim("cube", p + "_light", p, "emit_team", loc=(x - 0.05, 0, 0.98), scale=(0.006, 0.05, 0.1))
    p = f"hand_{n}"
    bevel(prim("cube", p + "_fist", p, "fabric_dark", loc=(x, -0.006, 0.815), scale=(0.05, 0.088, 0.095)), 0.02, 3)
    bevel(prim("cube", p + "_knuckles", p, "armor_grey", loc=(x, -0.05, 0.8), scale=(0.054, 0.02, 0.06)), 0.007, 2)
    bevel(prim("cube", p + "_thumb", p, "fabric_dark", loc=(x - s * 0.03, -0.03, 0.83), scale=(0.022, 0.03, 0.06)), 0.009, 2)
    prim("cyl", p + "_cuff", p, "polymer", loc=(x, 0, 0.865), scale=(0.088, 0.088, 0.03), verts=16)


def core():
    p = "pelvis"
    subsurf(prim("cube", "pelvis_suit", p, "fabric_dark", loc=(0, 0.005, 0.95), scale=(0.31, 0.2, 0.17)), 2)
    belt = prim("torus", "belt", p, "polymer", loc=(0, 0.0, 1.0), scale=(1.0, 0.68, 1.0), R=0.152, r=0.02, seg=32, mseg=8)
    squash(belt, 1, 1, 1.6)
    bevel(prim("cube", "buckle", p, "metal", loc=(0, -0.105, 1.0), scale=(0.055, 0.012, 0.035)), 0.004, 2)
    for i, (x, y) in enumerate(((0.11, -0.06), (-0.11, -0.06), (0.15, 0.03), (-0.15, 0.03), (0.07, 0.11), (-0.07, 0.11))):
        bevel(prim("cube", f"pouch{i}", p, "fabric_dark" if i % 2 else "polymer_tan", loc=(x, y, 0.975), scale=(0.055, 0.04, 0.07)), 0.01, 2)
    bevel(prim("cube", "codpiece", p, "armor_grey", loc=(0, -0.085, 0.9), rot=(math.radians(8), 0, 0), scale=(0.12, 0.03, 0.1)), 0.012, 2)

    p = "torso"
    body = lathe("torso_suit", p, "fabric", [(0.135, 0.98), (0.15, 1.08), (0.175, 1.25), (0.18, 1.36), (0.15, 1.45), (0.085, 1.5), (0.07, 1.52)])
    squash(body, 1.0, 0.7, 1.0)
    bevel(prim("cube", "chestplate", p, "armor_team", loc=(0, -0.09, 1.31), rot=(math.radians(-6), 0, 0), scale=(0.33, 0.08, 0.25)), 0.035, 4)
    prim("cube", "chest_light_l", p, "emit_team", loc=(0.12, -0.133, 1.37), scale=(0.05, 0.006, 0.012))
    prim("cube", "chest_light_r", p, "emit_team", loc=(-0.12, -0.133, 1.37), scale=(0.05, 0.006, 0.012))
    for i, z in enumerate((1.16, 1.09)):
        bevel(prim("cube", f"abs{i}", p, "armor_grey", loc=(0, -0.085, z), scale=(0.23 - 0.02 * i, 0.04, 0.06)), 0.012, 2)
    prim("torus", "collar", p, "metal", loc=(0, 0.0, 1.49), R=0.105, r=0.022, seg=32, mseg=10)
    for sx in (1, -1):
        bevel(prim("cube", f"strap{sx}", p, "polymer", loc=(sx * 0.1, -0.11, 1.42), rot=(math.radians(-20), 0, 0), scale=(0.035, 0.02, 0.12)), 0.006, 2)
    bevel(prim("cube", "pack", p, "armor_white", loc=(0, 0.17, 1.27), scale=(0.32, 0.15, 0.42)), 0.035, 3)
    for sx in (1, -1):
        bevel(prim("cyl", f"tank{sx}", p, "armor_grey", loc=(sx * 0.095, 0.265, 1.26), scale=(0.105, 0.105, 0.36), verts=20), 0.03, 3)
        prim("cyl", f"valve{sx}", p, "metal", loc=(sx * 0.095, 0.265, 1.46), scale=(0.035, 0.035, 0.04), verts=12)
        nz = prim("cone", f"nozzle{sx}", p, "gunmetal", loc=(sx * 0.1, 0.2, 1.02), r1=0.04, r2=0.022, scale=(1, 1, 0.07), verts=16)
        cut(nz, prim("cone", f"nozzlecut{sx}", p, "gunmetal", loc=(sx * 0.1, 0.2, 1.015), r1=0.032, r2=0.012, scale=(1, 1, 0.07), verts=16))
    prim("cube", "pack_light", p, "emit_team", loc=(0, 0.248, 1.42), scale=(0.1, 0.006, 0.014))
    prim("cyl", "antenna", p, "polymer", loc=(0.12, 0.16, 1.58), scale=(0.008, 0.008, 0.26), verts=8)
    prim("sphere", "antenna_tip", p, "emit_red", loc=(0.12, 0.16, 1.71), scale=(0.014, 0.014, 0.014), seg=8, rings=4)

    p = "head"
    shell = prim("sphere", "helmet", p, "armor_white", loc=(0, 0.012, 1.645), scale=(0.31, 0.33, 0.32), seg=40, rings=20)
    cut(shell, prim("cube", "visorcut", p, "fabric", loc=(0, -0.16, 1.63), scale=(0.25, 0.2, 0.13)))
    bevel(shell, 0.003, 1)
    prim("sphere", "visor", p, "visor", loc=(0, 0.0, 1.64), scale=(0.29, 0.31, 0.29), seg=40, rings=20)
    bevel(prim("cube", "crest", p, "armor_team", loc=(0, 0.02, 1.8), scale=(0.045, 0.27, 0.035)), 0.012, 3)
    for sx in (1, -1):
        prim("cyl", f"ear{sx}", p, "armor_grey", loc=(sx * 0.155, 0.015, 1.63), rot=(0, math.radians(90), 0), scale=(0.095, 0.095, 0.04), verts=20)
        prim("cyl", f"earlamp{sx}", p, "emit_team", loc=(sx * 0.176, 0.015, 1.63), rot=(0, math.radians(90), 0), scale=(0.04, 0.04, 0.008), verts=16)
    prim("cyl", "headlamp", p, "emit_cyan", loc=(0.1, -0.135, 1.735), rot=(math.radians(80), 0, 0), scale=(0.03, 0.03, 0.03), verts=12)
    bevel(prim("cube", "chin", p, "armor_grey", loc=(0, -0.14, 1.53), rot=(math.radians(15), 0, 0), scale=(0.15, 0.05, 0.04)), 0.012, 2)
    prim("cyl", "neckseal", p, "fabric_dark", loc=(0, 0.0, 1.52), scale=(0.16, 0.14, 0.06), verts=20)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = argv[0] if argv else "assets/models/operative.omdl"
    reset()
    core()
    for s, n in ((1, "L"), (-1, "R")):
        leg(s, n)
        arm(s, n)
    apply_all()
    bake_ao(lambda o: o["part"])
    export_omdl(out, PIVOTS)
    if len(argv) > 1:
        preview(argv[1], (0, 0, 0.95), 3.4, elev=10, azim=-30, lens=55)


main()
