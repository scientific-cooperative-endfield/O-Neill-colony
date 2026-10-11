//! Procedural animation of the Blender-built operative: a rigid-part skeleton
//! posed every frame from the simulation state (gait from distance walked,
//! two-bone IK for arms → weapon grips and legs → planted feet, aim-driven
//! spine and head, reload / recoil / weapon-switch, ragdoll-like fall), plus
//! the first-person viewmodel (arms + weapon in camera space).
//!
//! Frames are built in a local space with X = left, Y = up, Z = forward —
//! the same axes the parts were exported in (see tools/blender/common.py).

use std::f32::consts::PI;
use std::sync::OnceLock;

use glam::{Affine3A, Mat3, Vec3};

use crate::models::{self, Socket};
use crate::render::props::PropInst;

pub const WEAPONS: [&str; 3] = ["pistol", "rifle", "sniper"];

pub struct Ids {
    pub pelvis: u16,
    pub torso: u16,
    pub head: u16,
    pub thigh: [u16; 2],
    pub shin: [u16; 2],
    pub foot: [u16; 2],
    pub upper: [u16; 2],
    pub fore: [u16; 2],
    pub hand: [u16; 2],
    pub weapon: [u16; 3],
    pub mag: [u16; 3],
    pub casing: u16,
    pub sockets: [WeaponSockets; 3],
}

#[derive(Clone, Copy)]
pub struct WeaponSockets {
    pub grip_l: Socket,
    pub muzzle: Socket,
    pub sight: Socket,
    pub eject: Socket,
}

pub fn ids() -> &'static Ids {
    static IDS: OnceLock<Ids> = OnceLock::new();
    IDS.get_or_init(|| {
        let l = models::lib();
        let lr = |n: &str| [l.part(&format!("{n}_L")), l.part(&format!("{n}_R"))];
        let s =
            |w: &str| WeaponSockets { grip_l: l.socket(w, "grip_l"), muzzle: l.socket(w, "muzzle"), sight: l.socket(w, "sight"), eject: l.socket(w, "eject") };
        Ids {
            pelvis: l.part("pelvis"),
            torso: l.part("torso"),
            head: l.part("head"),
            thigh: lr("thigh"),
            shin: lr("shin"),
            foot: lr("foot"),
            upper: lr("upperarm"),
            fore: lr("forearm"),
            hand: lr("hand"),
            weapon: WEAPONS.map(|w| l.part(w)),
            mag: WEAPONS.map(|w| l.part(&format!("{w}_mag"))),
            casing: l.part("casing"),
            sockets: WEAPONS.map(s),
        }
    })
}

// rest-pose skeleton (metres, local axes)
const HIP_X: f32 = 0.1;
const THIGH: f32 = 0.43;
const SHIN: f32 = 0.415;
const ANKLE_H: f32 = 0.085;
const SHOULDER: Vec3 = Vec3::new(0.2, 0.42, 0.0); // relative to the torso pivot
const UPPER: f32 = 0.29;
const FORE: f32 = 0.26;
const FIST: f32 = 0.055; // wrist → fist centre

/// Orthonormal frame with local +Y along `y` and +Z as close as possible to `z_hint`.
pub fn frame_y(o: Vec3, y: Vec3, z_hint: Vec3) -> Affine3A {
    let y = y.normalize_or(Vec3::Y);
    let mut z = z_hint - y * z_hint.dot(y);
    if z.length_squared() < 1e-8 {
        z = y.any_orthonormal_vector();
    }
    let z = z.normalize();
    Affine3A::from_mat3_translation(Mat3::from_cols(y.cross(z), y, z), o)
}

/// Orthonormal frame with +Z along `z` and +Y as close as possible to `y_hint`.
pub fn frame_z(o: Vec3, z: Vec3, y_hint: Vec3) -> Affine3A {
    let z = z.normalize_or(Vec3::Z);
    let mut y = y_hint - z * y_hint.dot(z);
    if y.length_squared() < 1e-8 {
        y = z.any_orthonormal_vector();
    }
    let y = y.normalize();
    Affine3A::from_mat3_translation(Mat3::from_cols(y.cross(z), y, z), o)
}

/// Two-bone IK: returns (elbow/knee, reached end).
pub fn ik(s: Vec3, t: Vec3, a: f32, b: f32, pole: Vec3) -> (Vec3, Vec3) {
    let d = t - s;
    let dir = d.normalize_or(Vec3::NEG_Y);
    let len = d.length().clamp(0.05, (a + b) * 0.999);
    let ca = ((a * a + len * len - b * b) / (2.0 * a * len)).clamp(-1.0, 1.0);
    let sa = (1.0 - ca * ca).sqrt();
    let mut perp = pole - dir * pole.dot(dir);
    if perp.length_squared() < 1e-8 {
        perp = dir.any_orthonormal_vector();
    }
    let e = s + dir * (a * ca) + perp.normalize() * (a * sa);
    let end = e + (s + dir * len - e).normalize_or(dir) * b;
    (e, end)
}

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Magazine offset along the weapon's down axis during a reload (progress 0..1),
/// and whether the support hand is on the magazine.
pub fn reload_mag(p: f32) -> (f32, bool) {
    if p <= 0.0 || p >= 1.0 {
        return (0.0, false);
    }
    if p < 0.25 {
        (smooth(p / 0.25) * 0.22, false)
    } else if p < 0.55 {
        (0.6, true) // out of sight: the hand fetches a fresh one
    } else if p < 0.85 {
        (0.22 * (1.0 - smooth((p - 0.55) / 0.3)), true)
    } else {
        (0.0, false)
    }
}

pub struct PoseIn {
    pub feet: Vec3,
    pub up: Vec3,
    /// facing (perpendicular to up)
    pub fwd: Vec3,
    pub aim: Vec3,
    pub vel: Vec3,
    pub gait: f32,
    pub grounded: bool,
    pub crouch: f32,
    pub jet: bool,
    pub dead_t: Option<f32>,
    pub weapon: usize,
    pub reload: f32,
    pub recoil: f32,
    pub ads: f32,
    pub switching: f32,
    pub tint: [f32; 4],
    pub ext: [f32; 4],
}

pub struct Posed {
    pub weapon: Affine3A,
    pub muzzle: Vec3,
    pub eject: Vec3,
    pub nozzles: [Vec3; 2],
    pub head: Vec3,
}

fn push(out: &mut Vec<(u16, PropInst)>, part: u16, w: &Affine3A, local: Affine3A, tint: [f32; 4], ext: [f32; 4]) {
    out.push((part, PropInst::new(*w * local, tint, ext)));
}

/// Whole operative in the world. Returns weapon / muzzle placement for effects.
pub fn pose_body(p: &PoseIn, out: &mut Vec<(u16, PropInst)>) -> Posed {
    let id = ids();
    let left = p.up.cross(p.fwd).normalize_or(Vec3::X);
    let basis = Mat3::from_cols(left, p.up, p.fwd);
    let to_local = basis.transpose();
    let aim = (to_local * p.aim).normalize_or(Vec3::Z);
    let v = to_local * p.vel;
    let hv = Vec3::new(v.x, 0.0, v.z);
    let speed = hv.length();
    let dead = p.dead_t.is_some();
    let fall = p.dead_t.map_or(0.0, |t| smooth(t / 0.8));
    let world = Affine3A::from_mat3_translation(basis * Mat3::from_rotation_x(-fall * PI * 0.5), p.feet + p.up * (0.17 * fall));
    let (tint, ext) = (p.tint, p.ext);

    let c = if dead { 0.0 } else { p.crouch };
    let mov = (speed / 1.2).min(1.0) * if p.grounded && !dead { 1.0 } else { 0.0 };
    let bob = mov * 0.035 * (2.0 * p.gait).sin().abs();
    let pitch = aim.y.clamp(-1.0, 1.0).asin();
    let lean = 0.1 * (speed / 5.0).min(1.0) * mov + 0.32 * c - if dead { 0.0 } else { 0.3 * pitch };
    let pelvis_o = Vec3::new(0.0, 0.95 - 0.33 * c - bob + if p.jet { 0.05 } else { 0.0 }, 0.0);
    // long guns: the torso turns so the support shoulder comes forward
    let twist = if dead || p.weapon == 0 { 0.0 } else { -0.45 };
    let rp = Mat3::from_rotation_x(lean * 0.3) * Mat3::from_rotation_y(twist * 0.35);
    push(out, id.pelvis, &world, Affine3A::from_mat3_translation(rp, pelvis_o), tint, ext);
    let rt = Mat3::from_rotation_x(lean) * Mat3::from_rotation_y(twist);
    let torso_o = pelvis_o + Vec3::new(0.0, 0.05, 0.0);
    push(out, id.torso, &world, Affine3A::from_mat3_translation(rt, torso_o), tint, ext);
    let ty = rt * Vec3::Y;
    let neck = torso_o + ty * 0.5;
    let look = if dead { rt * Vec3::Z } else { aim };
    push(out, id.head, &world, frame_z(neck, look, ty), tint, ext);
    let shoulders = [torso_o + rt * SHOULDER, torso_o + rt * (SHOULDER * Vec3::new(-1.0, 1.0, 1.0))];

    // weapon in local space
    let wf = aim;
    let wu = (ty - wf * ty.dot(wf)).normalize_or(Vec3::Y);
    let wl = wu.cross(wf);
    let mut grip = if p.weapon == 0 { torso_o + ty * 0.3 + wf * 0.42 + wl * 0.02 } else { shoulders[1] + wf * 0.31 - wu * 0.075 + wl * 0.07 };
    grip += wu * (0.04 * p.ads) - wf * (0.05 * p.recoil);
    let sw = smooth(p.switching * 3.0);
    let (wf2, wu2) = if sw > 0.0 {
        let r = Mat3::from_axis_angle(wl, 0.9 * sw);
        (r * wf, r * wu)
    } else {
        (wf, wu)
    };
    grip -= wu * (0.15 * sw);
    let wl2 = wu2.cross(wf2);
    let wpn = Affine3A::from_mat3_translation(Mat3::from_cols(wl2, wu2, wf2), grip);
    let s = &id.sockets[p.weapon.min(2)];
    let (mag_drop, hand_on_mag) = reload_mag(p.reload);
    let wl_target = if hand_on_mag { grip - wu2 * (0.12 + mag_drop) + wf2 * 0.04 } else { wpn.transform_point3(s.grip_l.pos) };

    if !dead {
        push(out, id.weapon[p.weapon.min(2)], &world, wpn, tint, ext);
        if mag_drop < 0.5 {
            let m = Affine3A::from_mat3_translation(wpn.matrix3.into(), grip - wu2 * mag_drop);
            push(out, id.mag[p.weapon.min(2)], &world, m, tint, ext);
        }
    }

    // arms
    for (side, &sh) in shoulders.iter().enumerate() {
        let sg = if side == 0 { 1.0 } else { -1.0 };
        let (target, hand_y, hand_z) = if dead {
            (sh + Vec3::new(sg * 0.3, -0.42, 0.12), Vec3::new(sg * 0.5, 1.0, 0.0), Vec3::Z)
        } else if side == 1 {
            let back = (wu2 * 0.45 - wf2 * 0.75).normalize();
            (grip, back, -wl2)
        } else {
            let d = (-wu2 * 0.8 + wl2 * 0.3).normalize();
            (wl_target, d, wf2)
        };
        let wrist_t = target + hand_y.normalize() * FIST;
        let pole = Vec3::new(sg * 0.9, -1.0, -0.35);
        let (elbow, wrist) = ik(sh, wrist_t, UPPER, FORE, pole);
        push(out, id.upper[side], &world, frame_y(sh, sh - elbow, Vec3::Z), tint, ext);
        push(out, id.fore[side], &world, frame_y(elbow, elbow - wrist, Vec3::Z), tint, ext);
        push(out, id.hand[side], &world, frame_y(wrist, hand_y, hand_z), tint, ext);
    }

    // legs
    let dir = if speed > 0.2 { hv / speed } else { Vec3::Z };
    let amp = (0.28 + 0.04 * speed.min(7.0)) * mov;
    for side in 0..2 {
        let sg = if side == 0 { 1.0 } else { -1.0 };
        let hip = pelvis_o + rp * Vec3::new(sg * HIP_X, -0.02, 0.0);
        let ph = p.gait + PI * side as f32;
        let ankle = if dead {
            Vec3::new(sg * 0.14, ANKLE_H, 0.0)
        } else if p.jet {
            hip + Vec3::new(sg * 0.02, -0.76, 0.1 + 0.04 * sg)
        } else if !p.grounded {
            hip + Vec3::new(sg * 0.01, -0.68, 0.16 - 0.1 * sg)
        } else {
            let stance = Vec3::new(sg * 0.115, ANKLE_H, if side == 0 { 0.17 } else { -0.13 } * c);
            stance + dir * (ph.sin() * amp) + Vec3::Y * (ph.cos().max(0.0) * 0.12 * mov)
        };
        let (knee, ank) = ik(hip, ankle, THIGH, SHIN, Vec3::new(0.0, 0.0, 1.0) + dir * 0.2);
        push(out, id.thigh[side], &world, frame_y(hip, hip - knee, Vec3::Z), tint, ext);
        push(out, id.shin[side], &world, frame_y(knee, knee - ank, Vec3::Z), tint, ext);
        let toe = if p.grounded && !dead { -(ph.cos().max(0.0)) * 0.25 * mov } else { 0.35 };
        push(out, id.foot[side], &world, Affine3A::from_mat3_translation(Mat3::from_rotation_x(toe), ank), tint, ext);
    }
    let muzzle = world.transform_point3(wpn.transform_point3(s.muzzle.pos));
    let eject = world.transform_point3(wpn.transform_point3(s.eject.pos));
    let nozzles = [0.1f32, -0.1].map(|x| world.transform_point3(torso_o + rt * Vec3::new(x, 0.0, -0.2)));
    Posed { weapon: world * wpn, muzzle, eject, nozzles, head: world.transform_point3(neck + ty * 0.15) }
}

pub struct ViewIn {
    pub eye: Vec3,
    pub fwd: Vec3,
    pub up: Vec3,
    pub weapon: usize,
    pub ads: f32,
    pub reload: f32,
    pub kick: f32,
    pub switching: f32,
    pub sway: [f32; 2],
    pub gait: f32,
    pub moving: f32,
    pub sprint: f32,
    pub tint: [f32; 4],
}

/// First-person arms and weapon. Returns placement for muzzle effects.
pub fn pose_viewmodel(p: &ViewIn, out: &mut Vec<(u16, PropInst)>) -> Posed {
    let id = ids();
    let w = p.weapon.min(2);
    let s = &id.sockets[w];
    let fwd = p.fwd.normalize_or(Vec3::Z);
    let up = (p.up - fwd * p.up.dot(fwd)).normalize_or(Vec3::Y);
    let cam = Affine3A::from_mat3_translation(Mat3::from_cols(up.cross(fwd), up, fwd), p.eye);
    let ext = [0.0, 1.0, 0.0, 0.0];
    let a = smooth(p.ads);
    // hip position of the grip (camera space: +X = left)
    let hip = [Vec3::new(-0.12, -0.16, 0.36), Vec3::new(-0.135, -0.18, 0.33), Vec3::new(-0.14, -0.17, 0.3)][w];
    let eye_d = [0.36, 0.17, 0.085][w];
    let ads = Vec3::new(0.0, 0.0, eye_d) - s.sight.pos;
    let bob = Vec3::new((p.gait).sin() * 0.012, -(2.0 * p.gait).sin().abs() * 0.01, 0.0) * p.moving * (1.0 - 0.8 * a);
    let mut grip = hip.lerp(ads, a) + bob + Vec3::new(p.sway[0], p.sway[1], 0.0) * (1.0 - 0.7 * a) - Vec3::Z * (0.055 * p.kick);
    let (mag_drop, hand_on_mag) = reload_mag(p.reload);
    let reload_tilt = if p.reload > 0.0 && p.reload < 1.0 { smooth((p.reload * 6.0).min(1.0)) * smooth(((1.0 - p.reload) * 6.0).min(1.0)) } else { 0.0 };
    let sw = smooth(p.switching * 3.0);
    let sp = smooth(p.sprint) * (1.0 - a);
    grip += Vec3::new(0.02, -0.05, -0.02) * reload_tilt + Vec3::new(0.0, -0.2, 0.0) * sw + Vec3::new(0.05, -0.05, -0.04) * sp;
    let rot = Mat3::from_rotation_x(-0.09 * p.kick - 0.35 * reload_tilt * 0.3 + 0.9 * sw + 0.25 * sp)
        * Mat3::from_rotation_z(-0.55 * reload_tilt)
        * Mat3::from_rotation_y(0.5 * sp + 0.04 * (1.0 - a));
    let wpn = Affine3A::from_mat3_translation(rot, grip);
    push(out, id.weapon[w], &cam, wpn, p.tint, ext);
    if mag_drop < 0.5 {
        push(out, id.mag[w], &cam, Affine3A::from_mat3_translation(rot, grip - rot * Vec3::Y * mag_drop), p.tint, ext);
    }
    let (wl, wu, wf) = (rot * Vec3::X, rot * Vec3::Y, rot * Vec3::Z);
    // shoulders sit a little forward of reality so short rest-pose arms reach the handguard
    let shoulders = [Vec3::new(0.24, -0.36, 0.1), Vec3::new(-0.2, -0.33, 0.0)];
    for (side, &sh) in shoulders.iter().enumerate() {
        let sg = if side == 0 { 1.0 } else { -1.0 };
        let (target, hand_y, hand_z) = if side == 1 {
            let back = (wu * 0.45 - wf * 0.75).normalize();
            (grip, back, -wl)
        } else {
            let t = if hand_on_mag { grip - wu * (0.12 + mag_drop) + wf * 0.04 } else { wpn.transform_point3(s.grip_l.pos) };
            (t, (-wu * 0.8 + wl * 0.3).normalize(), wf)
        };
        let wrist_t = target + hand_y.normalize() * FIST;
        let (elbow, wrist) = ik(sh, wrist_t, UPPER, FORE, Vec3::new(sg * 1.0, -1.0, -0.4));
        // upper arms would fill the lower screen edge at this FOV; forearms + hands read better alone
        push(out, id.fore[side], &cam, frame_y(elbow, elbow - wrist, Vec3::Z), p.tint, ext);
        push(out, id.hand[side], &cam, frame_y(wrist, hand_y, hand_z), p.tint, ext);
    }
    let muzzle = cam.transform_point3(wpn.transform_point3(s.muzzle.pos));
    let eject = cam.transform_point3(wpn.transform_point3(s.eject.pos));
    Posed { weapon: cam * wpn, muzzle, eject, nozzles: [p.eye; 2], head: p.eye }
}

/// World transform of a spent casing.
pub fn casing(pos: Vec3, rot: glam::Quat) -> Affine3A {
    Affine3A::from_rotation_translation(rot, pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ik_reaches_and_keeps_lengths() {
        let s = Vec3::new(0.2, 1.4, 0.0);
        let t = Vec3::new(0.0, 1.2, 0.4);
        let (e, w) = ik(s, t, 0.29, 0.26, Vec3::new(1.0, -1.0, 0.0));
        assert!((e.distance(s) - 0.29).abs() < 1e-4);
        assert!((w.distance(e) - 0.26).abs() < 1e-4);
        assert!(w.distance(t) < 1e-3);
    }

    #[test]
    fn frames_are_proper_rotations() {
        let f = frame_y(Vec3::ZERO, Vec3::new(0.3, -1.0, 0.2), Vec3::Z);
        assert!((f.matrix3.determinant() - 1.0).abs() < 1e-5);
        let g = frame_z(Vec3::ZERO, Vec3::new(0.1, 0.2, 1.0), Vec3::Y);
        assert!((g.matrix3.determinant() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn body_pose_is_finite() {
        let p = PoseIn {
            feet: Vec3::new(0.0, -4000.0, 100.0),
            up: Vec3::Y,
            fwd: Vec3::Z,
            aim: Vec3::new(0.0, 0.2, 1.0).normalize(),
            vel: Vec3::new(0.0, 0.0, 3.0),
            gait: 1.0,
            grounded: true,
            crouch: 0.0,
            jet: false,
            dead_t: None,
            weapon: 1,
            reload: 0.3,
            recoil: 0.2,
            ads: 0.0,
            switching: 0.0,
            tint: [1.0; 4],
            ext: [0.0; 4],
        };
        let mut out = Vec::new();
        let r = pose_body(&p, &mut out);
        assert!(out.len() >= 16, "{} parts", out.len());
        assert!(out.iter().all(|(_, i)| i.rows.iter().flatten().all(|x| x.is_finite())));
        assert!(r.muzzle.distance(p.feet) < 2.5);
    }
}
