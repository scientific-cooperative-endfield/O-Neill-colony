// ============================================================================
//  common.wgsl — prepended to every shader module.
//
//  O'Neill cylinder coordinate system (colony / rotating frame):
//    * the cylinder axis is +Z, from z = 0 to z = L,
//    * the floor is at radius R, "up" points toward the axis,
//    * surface coordinates: u = θ·R (arc length, wraps at 2πR), v = z,
//      h = height above the floor (towards the axis).
//  Three land valleys (centred at θ = 0°, 120°, 240°) alternate with three
//  window strips. Sunlight arrives along the axis and is bounced through each
//  window by an external mirror, lighting the valley opposite that window.
// ============================================================================

struct Globals {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    shadow_mat:    mat4x4<f32>,
    cam_pos:       vec4<f32>,               // xyz, w = time (s)
    cam_surf:      vec4<f32>,               // u, v, h, strip index of the camera
    light_dir:     array<vec4<f32>, 3>,     // per valley: dir TO light, w = strength
    sun_color:     vec4<f32>,               // rgb radiance, w = exposure
    sky_color:     vec4<f32>,               // in-scatter colour, w = fog density (1/m)
    bounce_color:  vec4<f32>,               // ambient bounce, w = night factor (0..1)
    colony:        vec4<f32>,               // R, L, omega (rad/s), spin angle (rad)
    wind:          vec4<f32>,               // dir (u, v), strength, speed (m/s)
    frustum:       array<vec4<f32>, 6>,
    grass:         vec4<f32>,               // spacing, radius, lod dist, thin start
    grid:          vec4<i32>,               // first cell u, v, grid size, seed
    field_base:    vec4<i32>,               // window base u, v; previous base u, v (cells)
    field_params:  vec4<f32>,               // cell size, res, imprint count, debug view
    balls:         array<vec4<f32>, 16>,    // imprints: u, v, clearance, radius
    ball_vel:      array<vec4<f32>, 16>,    // vu, vv, strength, -
    lights:        array<vec4<f32>, 4>,     // glowing orbs: xyz, intensity
    params:        vec4<f32>,               // shadows on, shadow texel, aircar count, frame
    params2:       vec4<f32>,               // render w, render h, mirror angle, tree dist
    space:         vec4<f32>,               // earth direction (inertial) xyz, w = sim dt
    bh:            vec4<f32>,               // black-hole mode: on, env gain, air pressure (0..1), hull split
    bh_rot:        mat4x4<f32>,             // colony direction -> black-hole frame (env cube)
    bh_misc:       vec4<f32>,               // haze boost, aerial-perspective strength, -, -
};

// Rigid transform of one hull panel (structure node) from its rest place to
// its current place in the frame of the player's fragment.
struct NodeXf {
    m:    mat4x4<f32>,
    info: vec4<f32>,    // fragment id, detached flag, -, -
};

// One grass blade produced by the culling compute shader (world space).
struct Blade {
    root:   vec3<f32>,
    width:  f32,
    ctrl:   vec3<f32>,
    packed: u32,        // unorm8 x4: hash, tint, dryness, kind
    tip:    vec3<f32>,
    facing: f32,        // width axis angle in the local (east, north) plane
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var height_tex: texture_2d<f32>;      // r32float, wraps in u
@group(0) @binding(2) var zone_tex: texture_2d<f32>;        // rgba8: grass, farm, city, forest
@group(0) @binding(3) var lin_sampler: sampler;             // repeat u, clamp v
@group(0) @binding(4) var shadow_tex: texture_depth_2d;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(0) @binding(6) var env_tex: texture_cube<f32>;           // black-hole sky (Rgba16Float, BH frame)
@group(0) @binding(7) var<storage, read> node_xf: array<NodeXf>;  // 396 structure nodes

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const STRIP_ANGLE: f32 = 2.09439510239;   // 120°
const HALF_LAND: f32 = 0.52359877559;     // 30°: land half-width (angle)
const BLOCK: f32 = 80.0;                  // city block pitch (m)
const ROAD_HALF: f32 = 7.0;
const FIELD_W: f32 = 56.0;                // farm field size across / along
const FIELD_L: f32 = 112.0;

// Grass kinds
const K_GRASS: u32 = 0u;
const K_WHEAT: u32 = 1u;
const K_YOUNG: u32 = 2u;
const K_LAVENDER: u32 = 3u;
const K_CANOLA: u32 = 4u;
const K_FLOWER: u32 = 5u;

// ---------------------------------------------------------------- hashing --
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash2(p: vec2<i32>) -> u32 { return pcg(pcg(bitcast<u32>(p.x)) + bitcast<u32>(p.y)); }
fn hash3(p: vec3<i32>) -> u32 { return pcg(hash2(p.xy) + bitcast<u32>(p.z)); }
fn u2f(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }

fn value_noise(p: vec2<f32>) -> f32 {
    let fl = floor(p);
    let i = vec2<i32>(fl);
    let f = p - fl;
    let u = f * f * (3.0 - 2.0 * f);
    let a = u2f(hash2(i));
    let b = u2f(hash2(i + vec2<i32>(1, 0)));
    let c = u2f(hash2(i + vec2<i32>(0, 1)));
    let d = u2f(hash2(i + vec2<i32>(1, 1)));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm3(p: vec2<f32>) -> f32 {
    var q = p;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 3; i++) {
        sum += amp * value_noise(q);
        q = vec2<f32>(q.x * 1.6 - q.y * 1.2, q.x * 1.2 + q.y * 1.6) + vec2<f32>(5.3, 1.7);
        amp *= 0.5;
    }
    return sum / 0.875;
}

// ----------------------------------------------------------- colony maths --
fn colony_r() -> f32 { return g.colony.x; }
fn colony_len() -> f32 { return g.colony.y; }
fn circumference() -> f32 { return TAU * g.colony.x; }

fn surf_to_world(u: f32, v: f32, h: f32) -> vec3<f32> {
    let th = u / g.colony.x;
    let r = g.colony.x - h;
    return vec3<f32>(r * cos(th), r * sin(th), v);
}

fn angle_of(p: vec3<f32>) -> f32 {
    var th = atan2(p.y, p.x);
    if (th < 0.0) { th += TAU; }
    return th;
}

// (u, v, h) of a world point.
fn world_to_surf(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(angle_of(p) * g.colony.x, p.z, g.colony.x - length(p.xy));
}

fn up_at(p: vec3<f32>) -> vec3<f32> {
    let l = max(length(p.xy), 1e-3);
    return vec3<f32>(-p.x / l, -p.y / l, 0.0);
}
fn east_of(th: f32) -> vec3<f32> { return vec3<f32>(-sin(th), cos(th), 0.0); }

// Nearest valley index and signed across-valley distance s (m) from its centre.
fn strip_of(th: f32) -> vec2<f32> {
    var k = round(th / STRIP_ANGLE);
    var d = th - k * STRIP_ANGLE;
    if (k > 2.5) { k = 0.0; }
    return vec2<f32>(k, d * g.colony.x);
}
fn half_land_width() -> f32 { return HALF_LAND * g.colony.x; }

// ---------------------------------------------------------- world textures --
fn sample_height(u: f32, v: f32) -> f32 {
    let dims = vec2<i32>(textureDimensions(height_tex));
    let du = circumference() / f32(dims.x);
    let dv = colony_len() / f32(dims.y - 1);
    let gx = u / du;
    let gy = clamp(v / dv, 0.0, f32(dims.y - 1) - 0.001);
    let fx = floor(gx);
    let fy = floor(gy);
    let x0 = ((i32(fx) % dims.x) + dims.x) % dims.x;
    let x1 = (x0 + 1) % dims.x;
    let y0 = i32(fy);
    let f = vec2<f32>(gx - fx, gy - fy);
    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), 0).x;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), 0).x;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y0 + 1), 0).x;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y0 + 1), 0).x;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Local terrain normal expressed in (east, up, north) components.
fn surf_normal_local(u: f32, v: f32, e: f32) -> vec3<f32> {
    let hl = sample_height(u - e, v);
    let hr = sample_height(u + e, v);
    let hd = sample_height(u, v - e);
    let hu = sample_height(u, v + e);
    return normalize(vec3<f32>(hl - hr, 2.0 * e, hd - hu));
}

fn local_to_world(n: vec3<f32>, th: f32) -> vec3<f32> {
    let e = east_of(th);
    let up = vec3<f32>(-cos(th), -sin(th), 0.0);
    return e * n.x + up * n.y + vec3<f32>(0.0, 0.0, 1.0) * n.z;
}

// rgba = (grass density, farm, city, forest)
fn sample_zone(u: f32, v: f32) -> vec4<f32> {
    return textureSampleLevel(zone_tex, lin_sampler, vec2<f32>(u / circumference(), v / colony_len()), 0.0);
}

// ---------------------------------------------------------- city & farms --
fn road_dist(s: f32, v: f32) -> f32 {
    let ds = abs(fract(s / BLOCK + 0.5) - 0.5) * BLOCK;
    let dv = abs(fract(v / BLOCK + 0.5) - 0.5) * BLOCK;
    return min(ds, dv);
}
fn road_mask(s: f32, v: f32, city: f32) -> f32 {
    if (city < 0.35) { return 0.0; }
    return 1.0 - smoothstep(ROAD_HALF - 0.6, ROAD_HALF + 0.4, road_dist(s, v));
}

// Farm field: x = crop type (0 wheat, 1 young, 2 lavender, 3 canola, 4 fallow,
// 5 barley), y = distance to field edge (m), z = hash.
fn farm_cell(s: f32, v: f32, strip: f32) -> vec3<f32> {
    let cs = floor(s / FIELD_W);
    let cv = floor(v / FIELD_L);
    let h = hash3(vec3<i32>(i32(cs), i32(cv), i32(strip) + 77));
    let edge = min(min(s - cs * FIELD_W, (cs + 1.0) * FIELD_W - s),
                   min(v - cv * FIELD_L, (cv + 1.0) * FIELD_L - v));
    return vec3<f32>(f32(h % 6u), edge, u2f(pcg(h)));
}

fn crop_albedo(kind: f32, n: f32) -> vec3<f32> {
    switch (u32(kind)) {
        // Muted, satellite-photo palette: ripe wheat, young crops, lucerne,
        // stubble, ploughed soil, fallow grass.
        case 0u: { return mix(vec3<f32>(0.30, 0.25, 0.14), vec3<f32>(0.38, 0.32, 0.19), n); }
        case 1u: { return mix(vec3<f32>(0.09, 0.14, 0.05), vec3<f32>(0.13, 0.18, 0.07), n); }
        case 2u: { return mix(vec3<f32>(0.07, 0.11, 0.045), vec3<f32>(0.10, 0.14, 0.06), n); }
        case 3u: { return mix(vec3<f32>(0.33, 0.30, 0.20), vec3<f32>(0.40, 0.36, 0.25), n); }
        case 4u: { return mix(vec3<f32>(0.12, 0.095, 0.07), vec3<f32>(0.17, 0.135, 0.10), n); }
        default: { return mix(vec3<f32>(0.17, 0.19, 0.10), vec3<f32>(0.22, 0.23, 0.13), n); }
    }
}

// ---------------------------------------------------------------- lighting --
// Light arriving at point p: (direction to light, strength). Each valley is lit
// by the mirror beam through the window opposite to it.
fn light_at(p: vec3<f32>) -> vec4<f32> {
    let k = i32(strip_of(angle_of(p)).x);
    return g.light_dir[k];
}

fn shadow_at(p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (g.params.x < 0.5) { return 1.0; }
    if (i32(strip_of(angle_of(p)).x) != i32(g.cam_surf.w)) { return 1.0; }
    let sp = g.shadow_mat * vec4<f32>(p + n * 0.12, 1.0);
    let uv = sp.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || sp.z <= 0.0 || sp.z >= 1.0) { return 1.0; }
    let t = g.params.y;
    let z = sp.z - 0.0008;
    var s = textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + vec2<f32>(-t, -t) * 0.6, z);
    s += textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + vec2<f32>(t, -t) * 0.6, z);
    s += textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + vec2<f32>(-t, t) * 0.6, z);
    s += textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + vec2<f32>(t, t) * 0.6, z);
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
    return mix(s * 0.25, 1.0, smoothstep(0.42, 0.5, edge));
}

// Slow cloud shadows: clouds float in the colony atmosphere, drifting with the wind.
fn cloud_shadow(u: f32, v: f32) -> f32 {
    let q = vec2<f32>(u, v) * 0.006 - g.wind.xy * g.cam_pos.w * (0.003 + g.wind.w * 0.0008);
    return mix(0.45, 1.0, smoothstep(0.40, 0.62, fbm3(q)));
}

// Hemispherical ambient: windows/far valleys above, bounce from the land below.
fn ambient(n: vec3<f32>, up: vec3<f32>) -> vec3<f32> {
    let k = dot(n, up) * 0.5 + 0.5;
    return mix(g.bounce_color.rgb * 0.5, g.sky_color.rgb * 0.55 + g.bounce_color.rgb * 0.35, k);
}

// Glowing orbs act as small point lights.
fn orb_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let l = g.lights[i];
        if (l.w == 0.0) { continue; }
        let d = l.xyz - p;
        let d2 = dot(d, d);
        if (d2 > 900.0) { continue; }
        let ndl = max(dot(n, d * inverseSqrt(d2)), 0.0) * 0.7 + 0.3;
        // w > 0: cyan glowing orb; w < 0: warm flash (muzzle, explosion)
        let col = select(vec3<f32>(0.35, 0.8, 1.0), vec3<f32>(1.0, 0.62, 0.3), l.w < 0.0);
        acc += col * abs(l.w) * ndl / (d2 + 0.6) * (1.0 - d2 / 900.0);
    }
    return acc;
}

// Atmospheric scattering inside the colony: the far valley, 1.6 km "above",
// fades into a luminous blue haze.
// Length of the view ray that runs through the colony air (the camera may be
// outside, in vacuum, looking in through the windows).
fn air_path(o: vec3<f32>, dir: vec3<f32>, d: f32) -> f32 {
    let rr = g.colony.x;
    if (dot(o.xy, o.xy) <= rr * rr && o.z >= 0.0 && o.z <= g.colony.y) { return d; }
    let a = dot(dir.xy, dir.xy);
    let b = dot(o.xy, dir.xy);
    let c = dot(o.xy, o.xy) - rr * rr;
    let disc = b * b - a * c;
    if (disc < 0.0 || a < 1e-8) { return 0.0; }
    let t0 = max((-b - sqrt(disc)) / a, 0.0);
    return max(d - t0, 0.0);
}

// Air column of the habitat: hydrostatic in the centrifugal field, the
// density falls toward the axis as exp(-k (1 - r²/R²)), k = ω²R²M/(2RT) ≈ 0.24.
fn air_density(p: vec3<f32>) -> f32 {
    let q = dot(p.xy, p.xy) / (g.colony.x * g.colony.x);
    return exp(-0.237 * (1.0 - min(q, 1.0)));
}

// Density-weighted path length (m of floor-density air) along the part of
// the ray that lies inside the colony; 4-point Gauss-Legendre.
fn air_column(o: vec3<f32>, dir: vec3<f32>, d: f32) -> f32 {
    let len = air_path(o, dir, d);
    if (len <= 0.0) { return 0.0; }
    let t0 = d - len;
    let h = len * 0.5;
    let m = t0 + h;
    let x1 = 0.3399810 * h;
    let x2 = 0.8611363 * h;
    let s = 0.6521452 * (air_density(o + dir * (m - x1)) + air_density(o + dir * (m + x1)))
          + 0.3478548 * (air_density(o + dir * (m - x2)) + air_density(o + dir * (m + x2)));
    return s * h;
}

// Rayleigh (λ⁻⁴: blue scatters 5.7x more than red) + Mie haze, sea-level
// values. Distant land loses contrast and turns blue, the far end cap of a
// 32 km cylinder dissolves in luminous haze — the cue that makes it look
// kilometres big instead of like a model.
const BETA_R: vec3<f32> = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6);
const BETA_M: f32 = 2.0e-5;

fn fog_k() -> f32 {
    var k = g.sky_color.w / 0.000042;
    if (bh_on()) { k *= g.bh.z; }
    return k;
}

fn aerial(col: vec3<f32>, o: vec3<f32>, dir: vec3<f32>, d: f32) -> vec3<f32> {
    let col_m = air_column(o, dir, d) * fog_k();
    if (col_m <= 0.0) { return col; }
    let ext = BETA_R + vec3<f32>(BETA_M);
    let tr = exp(-ext * col_m);
    let l = g.light_dir[i32(g.cam_surf.w)];
    let mu = dot(dir, l.xyz);
    // Rayleigh and Henyey-Greenstein (g = 0.76) phase functions, normalised
    // so that the in-scatter keeps the calibrated sky colour.
    let pr = 0.75 * (1.0 + mu * mu);
    let hg = 0.76;
    let pm = (1.0 - hg * hg) / pow(1.0 + hg * hg - 2.0 * hg * mu, 1.5) * 0.08;
    let sun = g.sun_color.rgb * l.w;
    let w_r = BETA_R / ext;
    let w_m = BETA_M / ext;
    let ins = g.sky_color.rgb * (0.55 + 0.45 * pr) * 1.15 + sun * (w_r * 0.035 * pr + w_m * 0.06 * pm);
    return col * tr + ins * (1.0 - tr);
}

// Cloud deck at 1.5 km above the floor (a cylindrical sheet r = R - 1500 m).
// Looking up, the view ray crosses it twice: the cloud bases above you and,
// 5 km further, the sunlit tops floating over the land on the other side.
const CLOUD_H: f32 = 1500.0;
fn cloud_at(p: vec3<f32>, from_below: bool) -> vec4<f32> {
    let cover = g.bh_misc.x;
    // angle measured from the middle of a window strip (noise seam hidden there)
    var th = atan2(p.y, p.x) - PI / 3.0;
    if (th < 0.0) { th += TAU; }
    let rc = g.colony.x - CLOUD_H;
    let drift = g.wind.xy * g.cam_pos.w * 4.0;
    let q = (vec2<f32>(th * rc, p.z) + drift) / 900.0;
    let n = fbm3(q) * 0.7 + fbm3(q * 3.1 + vec2<f32>(7.0, 2.0)) * 0.3;
    let a = smoothstep(1.0 - cover, 1.0 - cover + 0.22, n);
    if (a <= 0.0) { return vec4<f32>(0.0); }
    let lw = g.light_dir[i32(g.cam_surf.w)].w;
    let thick = smoothstep(1.0 - cover, 1.0, n);
    let lit = g.sun_color.rgb * lw * select(0.25, 0.08 + 0.2 * (1.0 - thick), from_below);
    let amb = g.sky_color.rgb * 1.4 + g.bounce_color.rgb * 0.5 + vec3<f32>(0.004);
    return vec4<f32>((lit + amb) * mix(1.0, 0.55, thick * select(0.0, 1.0, from_below)), a * 0.92);
}

fn atmosphere(col: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let o = g.cam_pos.xyz;
    let to = p - o;
    let d = length(to);
    let dir = to / max(d, 1e-3);
    var res = aerial(col, o, dir, d);
    // clouds (camera inside, below the deck)
    let rc = g.colony.x - CLOUD_H;
    let a = dot(dir.xy, dir.xy);
    let r2 = dot(o.xy, o.xy);
    if (g.bh_misc.x > 0.0 && r2 > rc * rc && r2 < g.colony.x * g.colony.x && a > 1e-6 && o.z > 0.0 && o.z < g.colony.y) {
        let b = dot(o.xy, dir.xy);
        let disc = b * b - a * (r2 - rc * rc);
        if (disc > 0.0 && b < 0.0) {
            let sq = sqrt(disc);
            let t1 = (-b - sq) / a;
            let t2 = (-b + sq) / a;
            if (t2 < d) {
                let c2 = cloud_at(o + dir * t2, false);
                let pz = o.z + dir.z * t2;
                if (c2.a > 0.0 && pz > 0.0 && pz < g.colony.y) {
                    res = mix(res, aerial(c2.rgb, o, dir, t2), c2.a);
                }
            }
            if (t1 < d) {
                let c1 = cloud_at(o + dir * t1, true);
                let pz = o.z + dir.z * t1;
                if (c1.a > 0.0 && pz > 0.0 && pz < g.colony.y) {
                    res = mix(res, aerial(c1.rgb, o, dir, t1), c1.a);
                }
            }
        }
    }
    return res;
}

// Target bend of the grass from wind gusts at surface point p = (u, v).
fn wind_goal(p: vec2<f32>, t: f32) -> vec2<f32> {
    let dir = g.wind.xy;
    let strength = g.wind.z;
    let perp = vec2<f32>(-dir.y, dir.x);
    let q = p - dir * t * g.wind.w;
    let gust = value_noise(q * 0.022);
    let gust2 = value_noise(q * 0.085 + vec2<f32>(13.7, 5.1));
    let turb = value_noise(q * 0.05 + vec2<f32>(-7.3, 21.0)) - 0.5;
    let mag = strength * (0.12 + 1.5 * gust * gust * gust + 0.3 * gust2);
    return (dir * mag + perp * turb * strength * 0.45) * 0.42;
}

fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

// ------------------------------------------------------- black-hole mode --
fn bh_on() -> bool { return g.bh.x > 0.5; }

// Lensed black-hole sky seen along colony-frame direction d.
fn bh_env(d: vec3<f32>) -> vec3<f32> {
    let b = (g.bh_rot * vec4<f32>(d, 0.0)).xyz;
    return textureSampleLevel(env_tex, lin_sampler, b, 0.0).rgb * g.bh.y;
}

// Structure node (hull panel) owning a colony-frame point: 16 rings x 24
// sectors of 15 degrees, plus 6 sectors per end cap.
fn bh_node(p: vec3<f32>) -> u32 {
    let len = colony_len();
    var th = atan2(p.y, p.x);
    if (th < 0.0) { th += TAU; }
    if ((p.z < 2.0 || p.z > len - 2.0) && length(p.xy) < colony_r() - 2.0) {
        let k = min(u32(th / (TAU / 6.0)), 5u);
        return 384u + select(0u, 6u, p.z > len * 0.5) + k;
    }
    let iz = u32(clamp(floor(p.z / (len / 16.0)), 0.0, 15.0));
    let j = min(u32(th / (TAU / 24.0)), 23u);
    return iz * 24u + j;
}

// Moves a point of the rest geometry with the panel that `anchor` belongs to.
fn bh_xf_at(p: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    if (g.bh.w < 0.5) { return p; }
    return (node_xf[bh_node(anchor)].m * vec4<f32>(p, 1.0)).xyz;
}
fn bh_xf(p: vec3<f32>) -> vec3<f32> { return bh_xf_at(p, p); }

// Fragment id of the panel at p (seams between fragments are discarded).
fn bh_fid(p: vec3<f32>) -> f32 {
    if (g.bh.w < 0.5) { return 0.0; }
    return node_xf[bh_node(p)].info.x;
}
fn bh_seam(fid: f32, fid_flat: f32) -> bool { return abs(fid - fid_flat) > 0.01; }
