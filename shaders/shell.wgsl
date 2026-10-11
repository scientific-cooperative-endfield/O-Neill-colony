// ============================================================================
//  shell.wgsl — the colony hull at canonical scale (R = 4 km, L = 32 km):
//  window strips with structural mullions, end caps with terraced habitats,
//  the axial spindle with station modules, and the exterior seen from space:
//  hinged mirrors, the land-strip hull, the sun-end agricultural ring and the
//  sky sphere (Milky Way, nebulae, star fields, Earth, Sun).
// ============================================================================

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) kind: f32,   // 0 window, 1 cap, 2 spindle, 3 mirror, 4 hull, 5 ring, 6 sky, 7 module
};
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) nrm: vec3<f32>,
    @location(3) fid: f32,
    @location(4) @interpolate(flat) fid_flat: f32,
};

fn cam_outside() -> bool {
    let c = g.cam_pos.xyz;
    return length(c.xy) > colony_r() + 1.0 || c.z < 0.0 || c.z > colony_len();
}

// Mirror opening angle: reflects the axial sunlight through the window at the
// elevation beta chosen by the day cycle, closes towards night.
fn mirror_alpha() -> f32 {
    return mix((PI * 0.5 - g.params2.z) * 0.5, 0.07, g.bounce_color.w);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var o: VsOut;
    let kind = u32(in.kind + 0.5);
    var p = in.pos;
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (kind == 3u) {
        // in.pos = (across 0..1, along 0..1, window index)
        let thc = (in.pos.z + 0.5) * STRIP_ANGLE;
        let rad = vec3<f32>(cos(thc), sin(thc), 0.0);
        let wh = STRIP_ANGLE * 0.5 - HALF_LAND;
        let th = thc + (in.pos.x - 0.5) * 2.0 * wh;
        let al = mirror_alpha();
        let dir = rad * sin(al) + vec3<f32>(0.0, 0.0, cos(al));
        p = vec3<f32>(cos(th), sin(th), 0.0) * (colony_r() + 25.0) + dir * (in.pos.y * colony_len());
        n = normalize(-rad * cos(al) + vec3<f32>(0.0, 0.0, sin(al)));
    } else if (kind == 6u) {
        p = g.cam_pos.xyz + in.pos * 1.0e5;
    }
    o.world_pos = p;
    o.kind = kind;
    o.nrm = n;
    if (kind == 6u) {
        o.clip = g.view_proj * vec4<f32>(p, 1.0);
        // black-hole mode: the sky lies behind everything, even debris
        // drifting farther than the sky sphere
        if (bh_on()) { o.clip.z = o.clip.w * 1e-7; }
        o.fid = 0.0;
    } else {
        o.clip = g.view_proj * vec4<f32>(bh_xf(p), 1.0);
        o.fid = bh_fid(p);
    }
    o.fid_flat = o.fid;
    return o;
}

fn rot_z(v: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

fn stars(d: vec3<f32>, scale: f32, density: f32) -> f32 {
    let q = d * scale;
    let cell = floor(q);
    let h = hash3(vec3<i32>(cell));
    if (u2f(h) > density) { return 0.0; }
    let h2 = pcg(h);
    let h3 = pcg(h2);
    let sp = cell + vec3<f32>(u2f(h2), u2f(h3), u2f(pcg(h3)));
    let dd = length(q - sp);
    let mag = pow(u2f(pcg(h3 + 7u)), 6.0) * 40.0 + 1.0;
    return mag * (1.0 - smoothstep(0.0, 0.07, dd));
}

fn star_tint(d: vec3<f32>) -> vec3<f32> {
    let t = u2f(hash3(vec3<i32>(floor(d * 90.0))));
    return mix(vec3<f32>(1.0, 0.78, 0.6), vec3<f32>(0.75, 0.85, 1.0), t);
}

fn earth_disc(di: vec3<f32>) -> vec4<f32> {
    let e = normalize(g.space.xyz);
    let ang = 0.03;
    let c = dot(di, e);
    if (c < cos(ang * 1.25)) { return vec4<f32>(0.0); }
    var t1 = normalize(cross(e, vec3<f32>(0.0, 1.0, 0.0)));
    let t2 = cross(e, t1);
    let x = dot(di, t1) / ang;
    let y = dot(di, t2) / ang;
    let r2 = x * x + y * y;
    if (r2 > 1.0) {
        let rim = 1.0 - smoothstep(1.0, 1.25, sqrt(r2));
        return vec4<f32>(vec3<f32>(0.2, 0.45, 1.0) * rim * 0.6, 0.0);
    }
    let nrm = normalize(-e * sqrt(1.0 - r2) + t1 * x + t2 * y);
    let sun_i = vec3<f32>(0.0, 0.0, 1.0);
    let lit = max(dot(nrm, sun_i), 0.0);
    let q = vec2<f32>(atan2(nrm.y, nrm.x) * 3.0 + g.cam_pos.w * 0.002, nrm.z * 4.0);
    let land = smoothstep(0.52, 0.58, fbm3(q * 1.3));
    let cloud = smoothstep(0.5, 0.75, fbm3(q * 2.1 + vec2<f32>(9.0, 3.0)));
    var base = mix(vec3<f32>(0.02, 0.07, 0.25), vec3<f32>(0.12, 0.2, 0.06), land);
    base = mix(base, vec3<f32>(0.9), cloud);
    let night_lights = vec3<f32>(1.0, 0.7, 0.3) * land * (1.0 - cloud) * smoothstep(0.1, 0.0, lit) * 0.15 * step(0.6, value_noise(q * 20.0));
    let rim = pow(1.0 - sqrt(1.0 - r2), 3.0) * vec3<f32>(0.3, 0.55, 1.0);
    return vec4<f32>((base * lit + rim * lit) * 4.0 + night_lights, 1.0);
}

// Inertial-frame deep sky: Milky Way with bulge, dust lanes, emission nebulae.
fn deep_space(di: vec3<f32>) -> vec3<f32> {
    var col = vec3<f32>(0.0004, 0.0005, 0.0009);
    let gal_n = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let h = dot(di, gal_n);
    let t1 = normalize(cross(gal_n, vec3<f32>(0.0, 0.0, 1.0)));
    let t2 = cross(gal_n, t1);
    let lon = atan2(dot(di, t2), dot(di, t1));
    let q = vec2<f32>(lon * 3.0, h * 9.0);
    let band = exp(-h * h * 28.0);
    let core = exp(-pow(lon - 0.6, 2.0) * 1.2) * exp(-h * h * 60.0);
    let dust = smoothstep(0.42, 0.68, fbm3(q * 2.3 + vec2<f32>(3.0, 1.0))) * exp(-h * h * 110.0);
    let neb = fbm3(q * 1.1 + vec2<f32>(11.0, 5.0));
    let fine = fbm3(q * 6.0 + vec2<f32>(-2.0, 7.0));
    col += vec3<f32>(0.05, 0.045, 0.055) * band * (0.35 + neb * 0.9 + fine * 0.3) * (1.0 - 0.85 * dust);
    col += vec3<f32>(0.11, 0.08, 0.05) * core * (1.0 - 0.9 * dust);
    col += vec3<f32>(0.07, 0.02, 0.04) * pow(max(neb - 0.55, 0.0), 2.0) * 10.0 * band;
    col += vec3<f32>(0.01, 0.025, 0.06) * pow(fbm3(q * 0.6 + vec2<f32>(-4.0, 9.0)), 3.0) * 0.8;
    let st = stars(di, 90.0, 0.25) + stars(di, 220.0, 0.15) * 0.6 + stars(di, 480.0, 0.35) * band * 0.5;
    col += star_tint(di) * st * 0.08;
    let e = earth_disc(di);
    col = mix(col, e.rgb, e.a) + e.rgb * (1.0 - e.a);
    return col;
}

// The real Sun on the colony axis (+Z), angular radius ~0.27°.
fn sun_disc(d: vec3<f32>) -> vec3<f32> {
    let mu = d.z;
    return vec3<f32>(1.0, 0.95, 0.88) * (smoothstep(0.999985, 0.99999, mu) * 20000.0 + pow(max(mu, 0.0), 6000.0) * 20.0 + pow(max(mu, 0.0), 1500.0) * 0.15);
}

fn space_radiance(d: vec3<f32>, window_index: i32) -> vec3<f32> {
    if (bh_on()) { return bh_env(d); }
    let di = rot_z(d, g.colony.w);
    var col = deep_space(di);
    // Sun image in the external mirror of this window.
    let j = (window_index + 2) % 3;
    let l = g.light_dir[j];
    let mu = dot(d, l.xyz);
    col += vec3<f32>(1.0, 0.95, 0.85) * l.w * (smoothstep(0.99994, 0.99997, mu) * 6000.0 + pow(max(mu, 0.0), 3000.0) * 60.0);
    // The mirror panel itself fills much of the view out of the window.
    col += vec3<f32>(0.02, 0.025, 0.03) * l.w * smoothstep(0.2, 0.9, mu);
    return col;
}

const SUN_DIR: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
const SUN_OUT: vec3<f32> = vec3<f32>(3.0, 2.9, 2.7);

// Exterior metal: panels, seams, sun specular, environment reflection.
fn hull_shade(p: vec3<f32>, n: vec3<f32>, d: vec3<f32>, pc: vec2<f32>, fw: f32) -> vec3<f32> {
    let cell = floor(pc / vec2<f32>(60.0, 90.0));
    let ph = hash3(vec3<i32>(i32(cell.x), i32(cell.y), 17));
    let f = fract(pc / vec2<f32>(60.0, 90.0));
    let seam_w = clamp(fw / 60.0, 0.004, 0.5);
    let seam = 1.0 - smoothstep(0.0, seam_w * 2.0, min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)));
    var albedo = vec3<f32>(0.2, 0.205, 0.215) * mix(0.75, 1.2, u2f(ph));
    albedo = mix(albedo, vec3<f32>(0.05), seam * 0.6);
    let r = reflect(d, n);
    let diff = max(dot(n, SUN_DIR), 0.0);
    let spec = pow(max(dot(r, SUN_DIR), 0.0), 120.0) * 4.0 + pow(max(dot(r, SUN_DIR), 0.0), 12.0) * 0.15;
    let env = deep_space(rot_z(r, g.colony.w)) * 0.25;
    let earthshine = vec3<f32>(0.006, 0.008, 0.012);
    // Light bounced off the big mirrors and leaking out of the neighbouring
    // windows (warm interior glow), plus a cool sky fill.
    let th = atan2(p.y, p.x);
    let wc = (floor(th / STRIP_ANGLE) + 0.5) * STRIP_ANGLE;
    let wn = vec3<f32>(cos(wc), sin(wc), 0.0);
    let wc2 = wc - sign(th - wc) * STRIP_ANGLE;
    let wn2 = vec3<f32>(cos(wc2), sin(wc2), 0.0);
    let mirror = SUN_OUT * 0.16 * (max(dot(n, wn), 0.0) + max(dot(n, wn2), 0.0));
    let glow = vec3<f32>(0.18, 0.2, 0.16) * g.sun_color.rgb * 0.05;
    var col = albedo * (SUN_OUT * diff + mirror + earthshine + glow + vec3<f32>(0.05, 0.055, 0.07)) + SUN_OUT * spec * 0.08 + env;
    // Navigation beacons.
    let bc = fract(pc / 600.0);
    // point-like lights, energy-conserving with distance (no bright blobs far away)
    let bsize = 1.5 / 600.0;
    let bw = max(fw / 600.0, bsize);
    let beacon = (1.0 - smoothstep(0.0, bw, length(bc - 0.5))) * pow(bsize / bw, 2.0) * step(0.5, fract(g.cam_pos.w * 0.5 + u2f(ph))) * step(0.7, u2f(pcg(ph)));
    col += vec3<f32>(1.0, 0.15, 0.05) * beacon * 20.0;
    return col;
}

fn cap_shade(p: vec3<f32>, fw: f32) -> vec3<f32> {
    let r = length(p.xy);
    let a = atan2(p.y, p.x);
    let night = g.bounce_color.w;
    let into = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0), p.z > colony_len() * 0.5);
    var albedo = vec3<f32>(0.3, 0.31, 0.33);
    var emissive = vec3<f32>(0.0);
    if (r < 260.0) {
        let spokes = step(0.9, fract(a / TAU * 24.0));
        albedo = mix(vec3<f32>(0.22, 0.23, 0.25), vec3<f32>(0.1), spokes);
        emissive += vec3<f32>(0.2, 0.7, 1.0) * (1.0 - smoothstep(0.0, 4.0, abs(r - 230.0))) * 4.0;
        emissive += vec3<f32>(1.0, 0.3, 0.1) * (1.0 - smoothstep(0.0, 3.0, abs(r - 120.0))) * (0.5 + 0.5 * sin(g.cam_pos.w * 3.0)) * 3.0;
    } else {
        // Terraced habitats; the fine pattern is pre-filtered with distance.
        let band = (colony_r() - r) / 120.0;
        let bi = floor(band);
        let bf = fract(band);
        let fine_fade = smoothstep(0.8, 3.0, fw);
        let band_fade = smoothstep(15.0, 60.0, fw);
        let garden = mix(vec3<f32>(0.05, 0.10, 0.035), vec3<f32>(0.10, 0.15, 0.06), value_noise(vec2<f32>(a * 600.0, bi)));
        let col_i = floor(a * r / 3.0);
        let row_i = floor(bf * 120.0 / 3.5);
        let wh = hash3(vec3<i32>(i32(col_i), i32(row_i), i32(bi)));
        let cellx = fract(a * r / 3.0);
        let celly = fract(bf * 120.0 / 3.5);
        let win = step(0.15, cellx) * step(cellx, 0.85) * step(0.2, celly) * step(celly, 0.8);
        let lit_frac = 0.25 + night * 0.4;
        let lit = step(u2f(wh), lit_frac);
        let wall = vec3<f32>(0.5, 0.48, 0.45);
        let fac_alb = mix(mix(wall, vec3<f32>(0.05, 0.07, 0.1), win), mix(wall, vec3<f32>(0.05, 0.07, 0.1), 0.39), fine_fade);
        let fac_em = vec3<f32>(1.0, 0.72, 0.42) * (0.1 + night * 2.5) * mix(win * lit, 0.39 * lit_frac, fine_fade);
        let is_garden = mix(step(bf, 0.4), 0.4, band_fade);
        albedo = mix(fac_alb, garden, is_garden);
        emissive += fac_em * (1.0 - is_garden);
    }
    let up = into;
    let l = g.light_dir[i32(g.cam_surf.w)];
    return albedo * (g.sun_color.rgb * l.w * max(dot(up, l.xyz), 0.0) * 0.6 + ambient(up, up)) + emissive;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let th = angle_of(p);
    let lv = vec2<f32>(th * colony_r(), p.z);
    let fw = fwidth(lv);
    let fr = fwidth(length(p.xy));
    let d = normalize(p - g.cam_pos.xyz);
    let outside = cam_outside();
    var col = vec3<f32>(0.0);
    if (bh_seam(in.fid, in.fid_flat)) { discard; }

    switch (in.kind) {
        case 0u: {
            // Window glass with structural mullions: 24 longerons x 640 rings.
            let along = abs(fract(lv.y / 50.0 + 0.5) - 0.5) * 50.0;
            let across = abs(fract(th / (TAU / 288.0) + 0.5) - 0.5) * (TAU / 288.0) * colony_r();
            let fa = max(fw.y, 0.01);
            let fc = max(fw.x, 0.01);
            let fa_line = (1.0 - smoothstep(0.8 - fa, 0.8 + fa, along)) * clamp(0.8 / fa, 0.1, 1.0);
            let fc_line = (1.0 - smoothstep(1.2 - fc, 1.2 + fc, across)) * clamp(1.2 / fc, 0.1, 1.0);
            let big = abs(fract(lv.y / 1000.0 + 0.5) - 0.5) * 1000.0;
            let big_line = 1.0 - smoothstep(4.0 - fa, 4.0 + fa, big);
            let frame = max(max(fa_line, fc_line), big_line);
            let n = up_at(p);
            if (outside) {
                // From space only the primary structure is resolved: 72
                // longerons around the hull and a ring frame every kilometre.
                let lon = abs(fract(th / (TAU / 72.0) + 0.5) - 0.5) * (TAU / 72.0) * colony_r();
                let lw = max(fw.x, 0.5);
                let bw = max(fw.y, 0.5);
                let cover = max(1.0 - smoothstep(6.0 - lw, 6.0 + lw, lon), 1.0 - smoothstep(4.0 - bw, 4.0 + bw, big));
                if (cover < 0.5) { discard; }
                col = hull_shade(p, -n, d, lv, max(fw.x, fw.y));
                return vec4<f32>(col, 1.0);
            }
            let wi = i32(floor(th / STRIP_ANGLE));
            var glass = space_radiance(d, wi);
            let fres = 0.04 + 0.96 * pow(1.0 - abs(dot(d, n)), 5.0);
            glass = mix(glass, g.sky_color.rgb * 0.5 + g.bounce_color.rgb * 0.3, fres * 0.3);
            let l = g.light_dir[i32(g.cam_surf.w)];
            let metal = vec3<f32>(0.12, 0.13, 0.15) * (ambient(n, n) + g.sun_color.rgb * l.w * 0.1);
            col = mix(glass, metal, frame);
        }
        case 1u: {
            let outer_side = (p.z < 1.0 && g.cam_pos.z < p.z) || (p.z > colony_len() - 1.0 && g.cam_pos.z > p.z);
            if (outer_side) {
                let nz = select(vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0), p.z > 1.0);
                let r = length(p.xy);
                col = hull_shade(p, nz, d, vec2<f32>(atan2(p.y, p.x) * r, r), max(fr, fw.x));
                col += vec3<f32>(0.3, 0.8, 1.0) * (1.0 - smoothstep(0.0, 6.0, abs(r - 230.0))) * 6.0;
                return vec4<f32>(col, 1.0);
            }
            col = cap_shade(p, max(fw.x, fr));
        }
        case 2u, 7u: {
            // Spindle tube / station modules: rings, windows, transit pulses.
            let ring = 1.0 - smoothstep(0.0, max(fw.y, 0.05) * 2.0, abs(fract(p.z / 100.0 + 0.5) - 0.5) * 100.0 - 1.5);
            let pulse = pow(1.0 - fract((p.z - g.cam_pos.w * 140.0) / 450.0), 12.0)
                      + pow(1.0 - fract((-p.z - g.cam_pos.w * 110.0) / 380.0), 12.0);
            let base = vec3<f32>(0.32, 0.33, 0.35) * (g.sky_color.rgb + g.bounce_color.rgb * 1.5 + vec3<f32>(0.01));
            if (in.kind == 7u) {
                let a = atan2(p.y, p.x) * 70.0;
                let wy = fract(p.z / 6.0);
                let wx = fract(a / 4.0);
                let win = step(0.3, wy) * step(wy, 0.7) * step(0.2, wx) * step(wx, 0.8) * (1.0 - smoothstep(1.0, 3.0, fw.y));
                col = mix(base * 1.3, vec3<f32>(1.0, 0.8, 0.55) * 1.5, win * 0.8) + vec3<f32>(0.2, 0.8, 1.0) * ring;
            } else {
                col = base + vec3<f32>(0.2, 0.8, 1.0) * ring * 2.0 + vec3<f32>(0.6, 0.9, 1.0) * pulse * 8.0;
            }
        }
        case 3u: {
            // Hinged mirror: aluminised front reflects space and the Sun,
            // the back is a dark truss.
            let n = normalize(in.nrm);
            let front = dot(n, -d) > 0.0;
            let nn = select(-n, n, front);
            let pc = vec2<f32>(lv.x, p.z);
            if (front) {
                let r = reflect(d, nn);
                let seam = 1.0 - smoothstep(0.0, max(fw.y, 1.0) * 2.0, abs(fract(length(p.z) / 400.0 + 0.5) - 0.5) * 400.0 - 1.0);
                var sky_r = deep_space(rot_z(r, g.colony.w)) + sun_disc(r);
                if (bh_on()) { sky_r = bh_env(r); }
                col = sky_r * vec3<f32>(0.86, 0.88, 0.9)
                    + vec3<f32>(0.05, 0.055, 0.06) * SUN_OUT * max(dot(nn, SUN_DIR), 0.0) * 0.25;
                col = mix(col, vec3<f32>(0.02), seam * 0.7);
            } else {
                col = hull_shade(p, nn, d, pc, max(fw.x, fw.y)) * 0.8;
            }
            return vec4<f32>(col, 1.0);
        }
        case 4u: {
            if (!outside) { discard; }
            col = hull_shade(p, up_at(p) * -1.0, d, lv, max(fw.x, fw.y));
            return vec4<f32>(col, 1.0);
        }
        case 5u: {
            // Agricultural ring at the sun end: glass pods with crops.
            let nz = select(vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0), g.cam_pos.z > p.z);
            let r = length(p.xy);
            let a = atan2(p.y, p.x);
            let pod = step(0.15, fract(a / TAU * 72.0)) * step(0.1, fract((r - colony_r() * 1.05) / 350.0)) * step(fract((r - colony_r() * 1.05) / 350.0), 0.9);
            let lit = max(dot(nz, SUN_DIR), 0.0);
            let crops = mix(vec3<f32>(0.04, 0.09, 0.03), vec3<f32>(0.1, 0.16, 0.05), value_noise(vec2<f32>(a * 300.0, r * 0.05)));
            col = mix(hull_shade(p, nz, d, vec2<f32>(a * r, r), max(fr, fw.x)), crops * SUN_OUT * (0.3 + lit) + vec3<f32>(0.6, 0.25, 0.9) * g.bounce_color.w * 0.3, pod);
            return vec4<f32>(col, 1.0);
        }
        default: {
            if (bh_on()) { return vec4<f32>(bh_env(d), 1.0); }
            if (!outside) { discard; }
            col = deep_space(rot_z(d, g.colony.w)) + sun_disc(d);
            return vec4<f32>(col, 1.0);
        }
    }
    return vec4<f32>(atmosphere(col, p), 1.0);
}
