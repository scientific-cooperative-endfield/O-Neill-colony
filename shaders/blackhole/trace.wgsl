// Per-pixel backward null-geodesic tracer (see src/bin/blackhole/null_geo.rs
// for the CPU reference and DECISIONS.md for the approximations).
//
// Each ray stays in the plane spanned by the camera position and its initial
// direction; in that plane u = 1/r obeys u'' + u = 3u² (' = d/dφ). The ray is
// integrated with adaptive RK4 in φ. Crossings of the equatorial plane are
// found analytically (z ∝ a cos φ + b sin φ) and shaded as the thin accretion
// disk, so every disk image (primary, secondary, tertiary …) appears naturally.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var samp_rep: sampler;
@group(0) @binding(2) var disk_tex: texture_2d<f32>;
@group(0) @binding(3) var lut_tex: texture_2d<f32>;
@group(0) @binding(4) var sky_tex: texture_cube<f32>;
@group(0) @binding(5) var samp_clamp: sampler;
@group(0) @binding(6) var out_emit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_dir: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>, 4>;
@group(0) @binding(9) var<storage, read> snaps: array<Snap>;
@group(0) @binding(10) var<storage, read> nodes: array<Node>;

// hull panels per snapshot slot (structure.rs N_NODES)
const N_NODES_MAX: u32 = 396u;

var<workgroup> wg_steps: atomic<u32>;

const BASE_STEP: f32 = 0.07;
const U_CAPTURE: f32 = 0.49950050; // 1 / (2 · 1.001)

fn bb(t: f32) -> vec3<f32> {
    return textureSampleLevel(lut_tex, samp_clamp, bb_uv(t), 0.0).rgb;
}

fn step_size(u: f32, w: f32) -> f32 {
    let ps = abs(3.0 * u - 1.0);
    var h = P.misc.x * BASE_STEP * (0.25 + 0.75 * clamp(ps * 3.0, 0.0, 1.0));
    if (abs(w) > 1e-30) {
        h = min(h, 0.3 * abs(u) / abs(w));
    }
    return max(h, 1e-7);
}

fn accel(u: f32) -> f32 {
    return 3.0 * u * u - u;
}

// Novikov–Thorne-like effective temperature: T ∝ [r⁻³ (1 − √(r_in/r))]^¼,
// normalised so that the peak (at r = 49/36 r_in) equals T_peak.
fn disk_temp(r: f32) -> f32 {
    let rin = P.disk.x;
    if (r <= rin) {
        return 0.0;
    }
    let f = (1.0 - sqrt(rin / r)) / (r * r * r);
    let rp = rin * 49.0 / 36.0;
    let fp = (1.0 - sqrt(rin / rp)) / (rp * rp * rp);
    return P.disk.z * pow(f / fp, 0.25);
}

// Frequency ratio g = ν_obs / ν_emit for gas on a Keplerian circular orbit at r.
// The emitter has u = u^t (∂t + Ω ∂φ), u^t = 1/√(1 − 3/r); with the conserved
// photon energy E (normalised to ν_obs = 1) and L_z: ν_emit = u^t (E − Ω L_z).
fn disk_g(r: f32, e: f32, lz: f32) -> f32 {
    let om = pow(r, -1.5);
    let ut = inverseSqrt(max(1.0 - 3.0 / r, 1e-4));
    return 1.0 / max(ut * (e - om * lz), 1e-4);
}

// Thin disk hit: returns premultiplied emission (rgb) and opacity (a).
fn shade_disk(xc: vec3<f32>, e: f32, lz: f32, lod: f32) -> vec4<f32> {
    let r = length(xc.xy);
    let rin = P.disk.x;
    let rout = P.disk.y;
    let g = disk_g(r, e, lz);
    let az = atan2(xc.y, xc.x);
    let om = pow(r, -1.5);
    let v = log(r / rin) / log(rout / rin);
    let uv = vec2<f32>((az - om * P.cam_up.w) / TAU, v);
    let tx = textureSampleLevel(disk_tex, samp_rep, uv, lod);
    let dens = tx.r;
    let edge = smoothstep(0.0, 0.05, v) * (1.0 - smoothstep(0.55, 1.0, v));
    let t = disk_temp(r) * (0.82 + 0.36 * tx.g);
    let tobs = g * t;
    // bolometric intensity ∝ (g T)⁴, colour from the black-body LUT at g·T
    let k = tobs / P.disk.z;
    let inten = P.disk.w * k * k * k * k;
    let alpha = clamp(P.disk2.x * (0.25 + 0.95 * dens) * edge, 0.0, 1.0);
    let col = bb(tobs) * inten * (0.35 + 1.3 * dens);
    if ((P.flags.x & 8u) != 0u) {
        // debug: frequency shift g (red = blueshift g > 1, blue = redshift)
        return vec4<f32>(vec3<f32>(max(g - 1.0, 0.0), 0.05, max(1.0 - g, 0.0)) * 0.5, 1.0);
    }
    return vec4<f32>(col * alpha, alpha);
}

// Optically thin warm halo above/below the disk (visual extension, see DECISIONS.md).
fn halo(x: vec3<f32>, e: f32, lz: f32, ds: f32) -> vec3<f32> {
    let rc = length(x.xy);
    let rin = P.disk.x;
    let hh = P.disk2.z * rc;
    if (rc < rin * 0.75 || rc > P.disk.y * 1.1 || abs(x.z) > 3.0 * hh) {
        return vec3<f32>(0.0);
    }
    let zz = x.z / hh;
    let radial = smoothstep(rin * 0.75, rin * 1.3, rc) * (1.0 - smoothstep(P.disk.y * 0.5, P.disk.y * 1.1, rc));
    let dens = exp(-zz * zz) * radial;
    let r = max(rc, rin * 1.05);
    let g = disk_g(r, e, lz);
    let tobs = g * disk_temp(r) * 0.9;
    let k = tobs / P.disk.z;
    return bb(tobs) * (P.disk2.y * P.disk.w * k * k * k * k * dens * ds / (hh * 6.0));
}

// Covariant spatial photon momentum at radius r (PG chart, M = 1).
// pt = |L| / r is the tangential part; `outward` selects p^r > 0.
// p^r = −sE + (1 − s²) p_r = ±√(E² − (1 − s²) pt²).
fn photon_pr(r: f32, e: f32, pt: f32, outward: bool) -> f32 {
    let s = sqrt(2.0 / r);
    let q = 1.0 - s * s;
    let d = sqrt(max(e * e - q * pt * pt, 0.0));
    let pr_up = select(-d, d, outward);
    if (abs(q) > 0.05) {
        return (pr_up + s * e) / q;
    }
    // rationalised form, regular at the horizon
    return -(e * e - pt * pt) / (s * e - pr_up);
}

// ------------------------------------------------------------ station hull --
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn qrot_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return qrot(vec4<f32>(-q.xyz, q.w), v);
}

struct BoxHit {
    t: f32,
    nl: vec3<f32>,   // local normal
    pl: vec3<f32>,   // local hit point / half extents (−1 … 1)
}

// Ray (origin relative to the box centre) against an oriented box, t ∈ [0, tmax].
fn obb_hit(ro: vec3<f32>, rd: vec3<f32>, q: vec4<f32>, h: vec3<f32>, tmax: f32) -> BoxHit {
    var bh: BoxHit;
    bh.t = -1.0;
    let o = qrot_inv(q, ro);
    let d0 = qrot_inv(q, rd);
    let d = select(d0, vec3<f32>(1e-12), abs(d0) < vec3<f32>(1e-12));
    let inv = 1.0 / d;
    let t0 = (-h - o) * inv;
    let t1 = (h - o) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let tn = max(max(lo.x, lo.y), lo.z);
    let tf = min(min(hi.x, hi.y), hi.z);
    if (tn > tf || tf < 0.0 || tn > tmax) {
        return bh;
    }
    var nl = vec3<f32>(0.0, 0.0, -sign(d.z));
    if (tn == lo.x) {
        nl = vec3<f32>(-sign(d.x), 0.0, 0.0);
    } else if (tn == lo.y) {
        nl = vec3<f32>(0.0, -sign(d.y), 0.0);
    }
    let t = max(tn, 0.0);
    bh.t = t;
    bh.nl = nl;
    bh.pl = (o + d * t) / h;
    return bh;
}

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 127.1 + 3.7) * 43758.5453);
}

fn turbo(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);
    let r = 0.13572 + t * (4.6153 + t * (-42.66 + t * (132.13 + t * (-152.95 + t * 59.29))));
    let g = 0.0914 + t * (2.1942 + t * (4.84 + t * (-14.19 + t * (4.28 + t * 2.83))));
    let b = 0.1067 + t * (12.64 + t * (-60.58 + t * (110.36 + t * (-89.9 + t * 27.35))));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Hull panel radiance as seen by the camera: rest-frame shading (disk light
// from around the hole, sky reflections in the windows, interior glow), then
// the frequency shift g (colour from the black-body ratio, intensity ∝ g⁴).
fn shade_panel(nd: Node, idx: u32, hb: BoxHit, xw: vec3<f32>, rd: vec3<f32>, g: f32) -> vec3<f32> {
    let kind = nd.p.w;
    let nw = normalize(qrot(nd.q, hb.nl));
    let ldir = normalize(-xw);
    let ndl = max(dot(nw, ldir), 0.0);
    let lc = P.hull_light.rgb;
    let pl = hb.pl;
    let gx = abs(fract(pl.x * 2.0 + 0.5) - 0.5);
    let gy = abs(fract(pl.y * 4.0 + 0.5) - 0.5);
    let seam = 1.0 - smoothstep(0.0, 0.04, min(gx, gy));
    let outer = hb.nl.z > 0.5;
    let hsh = hash11(f32(idx));
    let refl = reflect(rd, nw);
    let spec = pow(max(dot(refl, ldir), 0.0), 60.0);
    var col: vec3<f32>;
    if (kind > 0.5 && kind < 1.5 && outer) {
        // window: dark glass, Fresnel sky reflection, lit land seen through it
        let fres = 0.04 + 0.96 * pow(1.0 - abs(dot(rd, nw)), 5.0);
        let sky = textureSampleLevel(sky_tex, samp_clamp, refl, 2.0).rgb * P.disk2.w;
        let mull = 1.0 - 0.8 * (1.0 - smoothstep(0.0, 0.02, abs(fract(pl.y * 10.0) - 0.5) - 0.46));
        let inside = vec3<f32>(0.30, 0.38, 0.22) * lc * (0.08 + 0.12 * hsh) * P.hull_light.w;
        col = (inside * mull + sky * fres + lc * spec * fres * 4.0) ;
    } else {
        let base = select(vec3<f32>(0.30, 0.31, 0.33), vec3<f32>(0.42, 0.40, 0.37), kind > 1.5) * (0.8 + 0.4 * hsh);
        let side = select(0.55, 1.0, outer);
        col = base * side * (lc * ndl + vec3<f32>(0.004)) * (1.0 - 0.45 * seam) + lc * spec * 0.25;
    }
    if ((P.flags.x & 16u) != 0u) {
        // debug: panel clock rate dτ/dt (blue slow → red fast)
        let span = max(P.heat.y - P.heat.x, 1e-30);
        col = turbo((nd.h.w - P.heat.x) / span) * (0.25 + 0.75 * max(ndl, 0.3));
        return col;
    }
    if ((P.flags.x & 32u) != 0u) {
        col = turbo(nd.e.y) * (0.25 + 0.75 * max(ndl, 0.3));
        return col;
    }
    let tint = bb(5800.0 * g) / bb(5800.0);
    return col * tint * (g * g * g * g);
}

// Snapshot pair bracketing the emission time te (relative to the camera's T).
fn snap_pick(te: f32) -> vec3<f32> {
    let n = u32(P.hull.x);
    if (P.hull.w < 0.5 || n < 2u || te >= snaps[0].c.w) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    for (var k = 0u; k + 1u < n; k = k + 1u) {
        let ta = snaps[k].c.w;
        let tb = snaps[k + 1u].c.w;
        if (tb <= te) {
            return vec3<f32>(f32(k), f32(k + 1u), clamp((ta - te) / max(ta - tb, 1e-9), 0.0, 1.0));
        }
    }
    return vec3<f32>(f32(n - 1u), f32(n - 1u), 0.0);
}

struct TraceOut {
    emit: vec3<f32>,
    trans: f32,
    dir: vec3<f32>,   // escape direction × g_sky, zero when captured / opaque
    key: f32,         // distance to the first opaque surface (−1 captured, 1e4 sky)
    steps: u32,
};

fn perp(v: vec3<f32>) -> vec3<f32> {
    if (abs(v.z) < 0.9) {
        return normalize(cross(v, vec3<f32>(0.0, 0.0, 1.0)));
    }
    return normalize(cross(v, vec3<f32>(1.0, 0.0, 0.0)));
}

fn hermite(u0: f32, m0: f32, u1: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * u0 + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * u1 + (t3 - t2) * m1;
}

fn zone_line(r: f32, rz: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(0.0, width, abs(r - rz));
}

fn trace(ndc: vec2<f32>) -> TraceOut {
    if (P.kerr.w > 0.5) { return trace_kerr(ndc); }
    var o: TraceOut;
    o.emit = vec3<f32>(0.0);
    o.trans = 1.0;
    o.dir = vec3<f32>(0.0);
    o.key = 1e4;
    o.steps = 0u;

    let th = P.cam_pos.w;
    let n_cam = normalize(P.cam_fwd.xyz + ndc.x * th * P.cam_right.w * P.cam_right.xyz + ndc.y * th * P.cam_up.xyz);
    // Photon arriving at the camera: k' = −n, energy 1 in the camera frame.
    // Boost to the local rain (free-fall from rest at infinity) frame.
    let beta = P.cam_beta.xyz;
    let gam = P.cam_fwd.w;
    let kc = -n_cam;
    let bk = dot(beta, kc);
    let b2 = dot(beta, beta);
    var p = kc;
    if (b2 > 1e-12) {
        p = kc + beta * ((gam - 1.0) * bk / b2 + gam);
    }
    let eps = gam * (1.0 + bk);
    let x0 = P.cam_pos.xyz;
    let r0 = length(x0);
    let rh = x0 / r0;
    let s0 = sqrt(2.0 / r0);
    // conserved quantities of the physical photon: E = −p_T = ε + v·p, L = x × p
    let e = eps - s0 * dot(rh, p);
    let lvec = cross(x0, p);
    let lz = lvec.z;
    let lmag = length(lvec);
    // backward ray in the rain frame and its orbital plane (rh, tv)
    let n = -p / eps;
    let cos_a = dot(n, rh);
    var tv = n - cos_a * rh;
    var sin_a = length(tv);
    if (sin_a < 1e-6) {
        tv = perp(rh);
        sin_a = 1e-6;
    } else {
        tv = tv / sin_a;
    }
    // --- hybrid lensing, near field: panels close to the camera (≪ r) are hit
    // by straight rays in the camera's local rain frame, camera-relative
    var near_skip = 0.0;
    let nsnap = u32(P.hull.x);
    let hull_on = nsnap > 0u && (P.flags.x & 64u) != 0u;
    if (hull_on && P.hull.y > 0.0) {
        let s0 = snaps[0];
        let dc = length(s0.c.xyz);
        if (dc - s0.info.x < P.hull.y) {
            near_skip = P.hull.y;
            let bq = dot(s0.c.xyz, n);
            let c2 = dc * dc - s0.info.x * s0.info.x;
            if (c2 < 0.0 || (bq > 0.0 && bq * bq > c2)) {
                let base = u32(s0.info.y) * N_NODES_MAX;
                let cnt = u32(s0.info.z);
                var best = 1e30;
                var bi = 0u;
                var bhb: BoxHit;
                for (var k = 0u; k < cnt; k = k + 1u) {
                    let nd = nodes[base + k];
                    if (nd.e.w < 0.5) {
                        continue;
                    }
                    let hb = obb_hit(-(s0.c.xyz + nd.p.xyz), n, nd.q, nd.h.xyz, best);
                    if (hb.t >= 0.0 && hb.t < best) {
                        best = hb.t;
                        bi = k;
                        bhb = hb;
                    }
                }
                if (best < 1e29) {
                    let nd = nodes[base + bi];
                    let ut = 1.0 / max(nd.w.w, 1e-6);
                    let nu = e * ut - dot(p, nd.w.xyz) * ut;
                    let g = clamp(1.0 / max(nu, 1e-6), 0.0, 50.0);
                    o.emit = shade_panel(nd, bi, bhb, x0 + n * best, n, g);
                    o.trans = 0.0;
                    o.key = best;
                    o.steps = 1u;
                    return o;
                }
            }
        }
    }
    let ibb = e / max(lmag, 1e-12);
    var dt_acc = 0.0;
    var u = 1.0 / r0;
    var w = -u * (cos_a + s0) / sin_a;
    var phi = 0.0;
    let az = rh.z;
    let bz = tv.z;
    // equatorial crossings: a cos φ + b sin φ = 0 at φ = δ + π/2 + kπ
    let delta = atan2(bz, az);
    var fz0 = az;
    var xprev = x0;
    var dist = 0.0;
    var crossings = 0.0;
    let u_esc = 1.0 / max(P.cam_beta.w, r0 * 1.01);
    let rin = P.disk.x;
    let rout = P.disk.y;
    let use_halo = (P.flags.x & 4u) != 0u && P.disk2.y > 0.0;
    let zones = (P.flags.x & 1u) != 0u;
    let br = P.beacon.w;
    // 0 running, 1 escaped, 2 captured, 3 opaque
    var status = 0u;
    if (u <= u_esc && w < 0.0) {
        status = 1u;
    }
    let max_steps = P.size.z;
    var i = 0u;
    loop {
        if (status != 0u || i >= max_steps) {
            break;
        }
        i = i + 1u;
        let h = step_size(u, w);
        // RK4 on (u, w)
        let k1u = w;
        let k1w = accel(u);
        let k2u = w + 0.5 * h * k1w;
        let k2w = accel(u + 0.5 * h * k1u);
        let k3u = w + 0.5 * h * k2w;
        let k3w = accel(u + 0.5 * h * k2u);
        let k4u = w + h * k3w;
        let k4w = accel(u + h * k3u);
        let u1 = u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        let w1 = w + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        let phi1 = phi + h;
        let c1 = cos(phi1);
        let sn1 = sin(phi1);
        let fz1 = az * c1 + bz * sn1;
        let um = max(0.5 * (u + u1), 1e-9);
        let wm = 0.5 * (w + w1);
        let ds = h / um * sqrt(1.0 + (wm / um) * (wm / um));
        let x1 = (1.0 / max(u1, 1e-9)) * (c1 * rh + sn1 * tv);

        // --- accretion disk (thin, z = 0) --------------------------------
        if (fz0 * fz1 <= 0.0 && fz0 != fz1) {
            let kk = ceil((phi - delta - 0.5 * PI) / PI);
            let phic = clamp(delta + 0.5 * PI + kk * PI, phi, phi1);
            let fr = (phic - phi) / h;
            let uc = hermite(u, w * h, u1, w1 * h, fr);
            if (uc > 0.0) {
                let rc = 1.0 / uc;
                let xc = rc * (cos(phic) * rh + sin(phic) * tv);
                if (rc >= rin && rc <= rout) {
                    // texture footprint: pixel cone width over the radial texel size,
                    // plus ~e^π demagnification per extra half-orbit (higher images)
                    let foot = (dist + fr * ds) * 2.0 * th / f32(P.size.y);
                    let texel = rc * log(rout / rin) / 256.0;
                    let lod = clamp(log2(max(foot / texel, 1e-3)) + 4.5 * crossings, 0.0, 10.0);
                    let c = shade_disk(xc, e, lz, lod);
                    o.emit += o.trans * c.rgb;
                    if (o.key > 9e3 && c.a > 0.3) {
                        o.key = dist + fr * ds;
                    }
                    o.trans *= 1.0 - c.a;
                    if (o.trans < 0.004) {
                        status = 3u;
                    }
                }
                crossings += 1.0;
                if (zones) {
                    let wdt = 0.02 * rc + 0.03;
                    let zl = zone_line(rc, 2.0, wdt) * vec3<f32>(1.0, 0.1, 0.1)
                        + zone_line(rc, 3.0, wdt) * vec3<f32>(1.0, 0.8, 0.1)
                        + zone_line(rc, 6.0, wdt) * vec3<f32>(0.1, 1.0, 0.3)
                        + zone_line(rc, P.misc.w, max(wdt, 0.01 * P.misc.w)) * vec3<f32>(0.3, 0.6, 1.0);
                    o.emit += o.trans * zl * 2.0;
                }
            }
        }
        // --- optically thin halo -----------------------------------------
        if (use_halo && status == 0u) {
            let xm = 0.5 * (xprev + x1);
            o.emit += o.trans * halo(xm, e, lz, ds);
        }
        // --- light-travel time: PG ΔT from the camera back to this segment.
        // dT/dφ = (1/b + √(2u) u′) / (u²(1 − 2u)); for ingoing light the
        // horizon factor cancels analytically (regular form)
        {
            let ub = max(um, 1e-6);
            var dtdphi = 0.0;
            if (wm < 0.0) {
                dtdphi = (ibb * ibb + 2.0 * ub * ub * ub) / (ub * ub * (ibb + sqrt(2.0 * ub) * abs(wm)));
            } else {
                dtdphi = (ibb + sqrt(2.0 * ub) * wm) / (ub * ub * max(1.0 - 2.0 * ub, 1e-4));
            }
            dt_acc += h * min(dtdphi, 1e6);
        }
        // --- station: hull panels or marker, sampled at the emission time ----
        if (nsnap > 0u && (hull_on || br > 0.0) && status == 0u && dist >= near_skip) {
            let seg = x1 - xprev;
            let sl = max(length(seg), 1e-12);
            let sd = seg / sl;
            let xr = xprev - x0;
            let rel = xr - snaps[0].c.xyz;
            let tcl = clamp(-dot(rel, sd), 0.0, sl);
            let reach = select(br + P.hull.z, P.hull.z, hull_on);
            if (length(rel + sd * tcl) < reach) {
                let pk = snap_pick(-dt_acc);
                let sa = snaps[u32(pk.x)];
                let sb = snaps[u32(pk.y)];
                let fr = pk.z;
                let cc = mix(sa.c.xyz, sb.c.xyz, fr);
                let uu = mix(sa.u, sb.u, fr);
                var tbest = 1e30;
                var nd_hit: Node;
                var hb_hit: BoxHit;
                var bi = 0u;
                var marker_hit = false;
                if (hull_on) {
                    let ba = u32(sa.info.y) * N_NODES_MAX;
                    let bb2 = u32(sb.info.y) * N_NODES_MAX;
                    let cnt = u32(sa.info.z);
                    let rr = xr - cc;
                    let rad = mix(sa.info.x, sb.info.x, fr);
                    let tcc = clamp(-dot(rr, sd), 0.0, sl);
                    if (length(rr + sd * tcc) < rad) {
                        for (var k = 0u; k < cnt; k = k + 1u) {
                            let na = nodes[ba + k];
                            let nb = nodes[bb2 + k];
                            if (na.e.w < 0.5) {
                                continue;
                            }
                            var nd = na;
                            nd.p = mix(na.p, nb.p, fr);
                            let qb = select(nb.q, -nb.q, dot(na.q, nb.q) < 0.0);
                            nd.q = normalize(mix(na.q, qb, fr));
                            let hb = obb_hit(rr - nd.p.xyz, sd, nd.q, nd.h.xyz, min(sl, tbest));
                            if (hb.t >= 0.0 && hb.t < tbest) {
                                tbest = hb.t;
                                nd_hit = nd;
                                hb_hit = hb;
                                bi = k;
                            }
                        }
                    }
                } else {
                    let bp = x0 + cc;
                    let tq = clamp(dot(bp - xprev, sd), 0.0, sl);
                    if (length(xprev + sd * tq - bp) < br) {
                        tbest = tq;
                        marker_hit = true;
                    }
                }
                if (tbest < 1e29) {
                    // exact redshift: ν_emit = E u^T − p_i u^i at the hit point
                    let t = tbest / sl;
                    let q = xprev + tbest * sd;
                    let rq = length(q);
                    let er = q / rq;
                    let ephi = -sin(phi + t * h) * rh + cos(phi + t * h) * tv;
                    let pt = lmag / rq;
                    let wq = mix(w, w1, t);
                    let pr = photon_pr(rq, e, pt, wq > 0.0);
                    let pcov = pr * er - pt * ephi;
                    if (marker_hit) {
                        let nu = e * uu.w - dot(pcov, uu.xyz);
                        let g = clamp(1.0 / max(nu, 1e-6), 0.0, 50.0);
                        let bp = x0 + cc;
                        let dd = length(q - bp);
                        let mu = sqrt(max(1.0 - (dd / br) * (dd / br), 0.0));
                        let tint = bb(P.beacon_col.w * g) / bb(P.beacon_col.w);
                        o.emit += o.trans * P.beacon_col.rgb * tint * (g * g * g * g) * (0.45 + 0.55 * mu);
                    } else {
                        let ut = 1.0 / max(nd_hit.w.w, 1e-6);
                        let nu = e * ut - dot(pcov, nd_hit.w.xyz) * ut;
                        let g = clamp(1.0 / max(nu, 1e-6), 0.0, 50.0);
                        o.emit += o.trans * shade_panel(nd_hit, bi, hb_hit, q, sd, g);
                    }
                    if (o.key > 9e3) {
                        o.key = dist + tbest;
                    }
                    o.trans = 0.0;
                    status = 3u;
                }
            }
        }
        u = u1;
        w = w1;
        phi = phi1;
        fz0 = fz1;
        xprev = x1;
        dist += ds;
        if (status == 0u) {
            if (u != u || u > 1e6 || (u > U_CAPTURE && w > 0.0)) {
                status = 2u;
            } else if (u < u_esc && w < 0.0) {
                status = 1u;
            }
        }
    }
    if (status == 0u) {
        status = select(1u, 2u, w > 0.0 || u > 0.3);
    }
    o.steps = i;
    if (status == 1u) {
        let phi_inf = phi + atan2(u, -w);
        let dir = cos(phi_inf) * rh + sin(phi_inf) * tv;
        // star light from infinity: g_sky = ν_obs / ν_∞ = 1 / E
        let g_sky = clamp(1.0 / max(e, 1e-3), 0.02, 30.0);
        o.dir = normalize(dir) * g_sky;
    } else if (status == 2u) {
        if (o.key > 9e3) {
            o.key = -1.0;
        }
    }
    return o;
}


// ------------------------------------------------------------------ Kerr --
// Spinning hole (a = P.kerr.x, M = 1) in Cartesian Kerr–Schild coordinates;
// see src/bh/kerr.rs (CPU reference with the tests). Photon: p_t = −1,
// Hamilton's equations for H = ½(−1 + |p|² − f (1 + l·p)²), integrated
// backwards from a static camera with RK4. Disk: prograde Keplerian ring
// from the ISCO (P.kerr.y) outwards.
struct KsGeom { r: f32, f: f32, l: vec3<f32>, };

fn ks_geom(x: vec3<f32>) -> KsGeom {
    let a = P.kerr.x;
    let a2 = a * a;
    let w = dot(x, x) - a2;
    let r2 = 0.5 * w + sqrt(0.25 * w * w + a2 * x.z * x.z);
    let r = sqrt(max(r2, 1e-12));
    var o: KsGeom;
    o.r = r;
    o.f = 2.0 * r2 * r / (r2 * r2 + a2 * x.z * x.z);
    let d = r2 + a2;
    o.l = vec3<f32>((r * x.x + a * x.y) / d, (r * x.y - a * x.x) / d, x.z / r);
    return o;
}

struct KsDer { dx: vec3<f32>, dp: vec3<f32>, };

fn ks_deriv(x: vec3<f32>, p: vec3<f32>) -> KsDer {
    let a = P.kerr.x;
    let a2 = a * a;
    let k = ks_geom(x);
    let r = k.r;
    let r2 = r * r;
    let s = 1.0 + dot(k.l, p);
    let ez = vec3<f32>(0.0, 0.0, 1.0);
    let dr = (x * r2 + ez * (a2 * x.z)) / (r * (2.0 * r2 - dot(x, x) + a2));
    let w = r2 * r2 + a2 * x.z * x.z;
    let df = (dr * (6.0 * r2 * w) - (dr * (4.0 * r2 * r) + ez * (2.0 * a2 * x.z)) * (2.0 * r2 * r)) / (w * w);
    let d = r2 + a2;
    let dlx = (dr * x.x + vec3<f32>(r, a, 0.0)) / d - dr * ((r * x.x + a * x.y) * 2.0 * r / (d * d));
    let dly = (dr * x.y + vec3<f32>(-a, r, 0.0)) / d - dr * ((r * x.y - a * x.x) * 2.0 * r / (d * d));
    let dlz = ez / r - dr * (x.z / r2);
    let ds = dlx * p.x + dly * p.y + dlz * p.z;
    var o: KsDer;
    o.dx = p - k.l * (k.f * s);
    o.dp = 0.5 * (df * (s * s) + ds * (2.0 * k.f * s));
    return o;
}

fn kerr_disk_g(r: f32, xi: f32) -> f32 {
    let a = P.kerr.x;
    let om = 1.0 / (pow(r, 1.5) + a);
    let n = (1.0 - 2.0 / r) + 4.0 * a * om / r - om * om * (r * r + a * a + 2.0 * a * a / r);
    if (n <= 0.0) { return 0.0; }
    return sqrt(n) / max(1.0 - om * xi, 1e-4);
}

fn shade_disk_kerr(xc: vec3<f32>, r: f32, g: f32) -> vec4<f32> {
    let rin = P.disk.x;
    let rout = P.disk.y;
    let az = atan2(xc.y, xc.x);
    let om = 1.0 / (pow(r, 1.5) + P.kerr.x);
    let v = log(r / rin) / log(rout / rin);
    let uv = vec2<f32>((az - om * P.cam_up.w) / TAU, v);
    let tx = textureSampleLevel(disk_tex, samp_rep, uv, 0.0);
    let dens = tx.r;
    let edge = smoothstep(0.0, 0.05, v) * (1.0 - smoothstep(0.55, 1.0, v));
    let tobs = g * disk_temp(r) * (0.82 + 0.36 * tx.g);
    let kk = tobs / P.disk.z;
    let inten = P.disk.w * kk * kk * kk * kk;
    let alpha = clamp(P.disk2.x * (0.25 + 0.95 * dens) * edge, 0.0, 1.0);
    if ((P.flags.x & 8u) != 0u) {
        return vec4<f32>(vec3<f32>(max(g - 1.0, 0.0), 0.05, max(1.0 - g, 0.0)) * 0.5, 1.0);
    }
    return vec4<f32>(bb(tobs) * inten * (0.35 + 1.3 * dens) * alpha, alpha);
}

fn trace_kerr(ndc: vec2<f32>) -> TraceOut {
    var o: TraceOut;
    o.emit = vec3<f32>(0.0);
    o.trans = 1.0;
    o.dir = vec3<f32>(0.0);
    o.key = 1e4;
    o.steps = 0u;
    let th = P.cam_pos.w;
    let n_cam = normalize(P.cam_fwd.xyz + ndc.x * th * P.cam_right.w * P.cam_right.xyz + ndc.y * th * P.cam_up.xyz);
    var x = P.cam_pos.xyz;
    let k0 = ks_geom(x);
    // photon arriving along −n_cam with p_t = −1: |p| from H = 0
    let nd = -n_cam;
    let c = dot(k0.l, nd);
    let qa = 1.0 - k0.f * c * c;
    let qb = -2.0 * k0.f * c;
    let qc = -(1.0 + k0.f);
    let kk = (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, 0.0))) / (2.0 * qa);
    var p = nd * kk;
    let xi = x.x * p.y - x.y * p.x;              // L_z / E (conserved)
    let g_cam = inverseSqrt(max(1.0 - k0.f, 0.02)); // static camera: ν_obs / ν_∞
    let rh = P.kerr.z;
    let r_esc = P.cam_beta.w;
    let nmax = min(P.size.z * 3u, 3000u);
    var status = 0u;
    var i = 0u;
    var dx_last = nd;
    for (; i < nmax; i = i + 1u) {
        let r = ks_geom(x).r;
        if (r < rh * 1.02) { status = 2u; break; }
        let h = -P.misc.x * 0.035 * clamp(r - rh * 0.97, 0.02, 0.6 * r);
        let d1 = ks_deriv(x, p);
        if (r > r_esc && dot(x, d1.dx) < 0.0) { dx_last = d1.dx; status = 1u; break; }
        let d2 = ks_deriv(x + d1.dx * (0.5 * h), p + d1.dp * (0.5 * h));
        let d3 = ks_deriv(x + d2.dx * (0.5 * h), p + d2.dp * (0.5 * h));
        let d4 = ks_deriv(x + d3.dx * h, p + d3.dp * h);
        let x1 = x + (d1.dx + 2.0 * d2.dx + 2.0 * d3.dx + d4.dx) * (h / 6.0);
        p = p + (d1.dp + 2.0 * d2.dp + 2.0 * d3.dp + d4.dp) * (h / 6.0);
        if (x.z * x1.z < 0.0) {
            let t = x.z / (x.z - x1.z);
            let xc = mix(x, x1, t);
            let rc = ks_geom(xc).r;
            if (rc >= P.disk.x && rc <= P.disk.y) {
                let g = g_cam * kerr_disk_g(rc, xi);
                let col = shade_disk_kerr(xc, rc, g);
                o.emit += o.trans * col.rgb;
                o.trans *= 1.0 - col.a;
                if (o.key > 9e3 && col.a > 0.5) { o.key = length(xc - P.cam_pos.xyz); }
                if (o.trans < 0.01) { status = 3u; x = x1; break; }
            }
        }
        x = x1;
        dx_last = d1.dx;
    }
    o.steps = i;
    if (status == 0u) { status = select(2u, 1u, length(x) > 20.0); }
    if (status == 1u) {
        o.dir = normalize(-dx_last) * clamp(g_cam, 0.02, 30.0);
    } else if (status == 2u && o.key > 9e3) {
        o.key = -1.0;
    }
    return o;
}

fn sky_color(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    let g = length(dir);
    let d = dir / g;
    var c = textureSampleLevel(sky_tex, samp_clamp, d, lod).rgb * P.disk2.w;
    // gravitational/kinematic blue- or redshift of the whole sky (∝ g⁴, colour shifted)
    let tint = bb(6500.0 * g) / bb(6500.0);
    return c * tint * (g * g * g * g);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        atomicStore(&wg_steps, 0u);
    }
    workgroupBarrier();
    let inside = gid.x < P.size.x && gid.y < P.size.y;
    var steps = 0u;
    if (inside) {
        let res = vec2<f32>(f32(P.size.x), f32(P.size.y));
        let spp = P.size.w;
        var emit = vec3<f32>(0.0);
        var trans = 0.0;
        var dir = vec3<f32>(0.0);
        var key = 1e4;
        if (spp <= 1u) {
            let ndc = vec2<f32>((f32(gid.x) + 0.5) / res.x * 2.0 - 1.0, 1.0 - (f32(gid.y) + 0.5) / res.y * 2.0);
            let t = trace(ndc);
            emit = t.emit;
            trans = t.trans;
            dir = t.dir;
            key = t.key;
            steps = t.steps;
        } else {
            // 4 rotated-grid samples; the sky is resolved here with a LOD
            // from the spread of the sample directions
            var offs = array<vec2<f32>, 4>(vec2<f32>(0.125, 0.375), vec2<f32>(0.625, 0.125), vec2<f32>(0.875, 0.625), vec2<f32>(0.375, 0.875));
            var dirs: array<vec3<f32>, 4>;
            var trs: array<f32, 4>;
            var mkey = 1e4;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let o = offs[k];
                let ndc = vec2<f32>((f32(gid.x) + o.x) / res.x * 2.0 - 1.0, 1.0 - (f32(gid.y) + o.y) / res.y * 2.0);
                let t = trace(ndc);
                emit += 0.25 * t.emit;
                dirs[k] = t.dir;
                trs[k] = t.trans;
                mkey = min(mkey, select(t.key, 1e5, t.key < 0.0));
                steps += t.steps;
            }
            var spread = 0.0;
            for (var k = 0u; k < 4u; k = k + 1u) {
                for (var j = k + 1u; j < 4u; j = j + 1u) {
                    if (dot(dirs[k], dirs[k]) > 0.0 && dot(dirs[j], dirs[j]) > 0.0) {
                        spread = max(spread, length(normalize(dirs[k]) - normalize(dirs[j])));
                    }
                }
            }
            let lod = clamp(log2(max(spread * 0.7, 1e-6) / P.misc.y), 0.0, P.misc.z - 1.0);
            for (var k = 0u; k < 4u; k = k + 1u) {
                if (dot(dirs[k], dirs[k]) > 0.0) {
                    emit += 0.25 * trs[k] * sky_color(dirs[k], lod);
                }
            }
            key = select(mkey, -1.0, mkey > 9e4);
            trans = 0.0;
            dir = vec3<f32>(0.0);
        }
        if ((P.flags.x & 2u) != 0u) {
            // debug: integrator steps per pixel (blue → red)
            let f = f32(steps) / f32(max(P.size.z * max(spp, 1u), 1u));
            emit = mix(vec3<f32>(0.0, 0.1, 1.0), vec3<f32>(1.0, 0.1, 0.0), f) * (0.2 + 4.0 * f);
            trans = 0.0;
            dir = vec3<f32>(0.0);
        }
        textureStore(out_emit, gid.xy, vec4<f32>(emit, trans));
        textureStore(out_dir, gid.xy, vec4<f32>(dir, key));
        atomicAdd(&wg_steps, steps);
    }
    workgroupBarrier();
    if (li == 0u) {
        atomicAdd(&stats[0], atomicLoad(&wg_steps));
    }
}
