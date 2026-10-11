// ============================================================================
//  balls.wgsl — every ball type: beach, soccer, basketball, bowling, glowing
//  orb, boulder, chrome, helium balloon.
// ============================================================================

struct BallGpu {
    pos:      vec3<f32>,
    radius:   f32,
    rot:      vec4<f32>,      // quaternion
    style:    u32,
    emissive: f32,
    ext:      vec2<f32>,      // x: stretch along local y (ellipsoid limbs), y: hue
};
@group(1) @binding(0) var<storage, read> balls_buf: array<BallGpu>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) obj: vec3<f32>,
    @location(3) @interpolate(flat) style: u32,
    @location(4) emissive: f32,
    @location(5) @interpolate(flat) hue: f32,
};

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @builtin(instance_index) ii: u32) -> VsOut {
    let b = balls_buf[ii];
    let s = select(1.0, b.ext.x, b.ext.x > 0.0);
    let local = vec3<f32>(pos.x, pos.y * s, pos.z);
    var o: VsOut;
    o.world_pos = b.pos + qrot(b.rot, local) * b.radius;
    // ellipsoid normal ∝ (x, y / s, z)
    o.normal = qrot(b.rot, normalize(vec3<f32>(pos.x, pos.y / s, pos.z)));
    o.obj = pos;
    o.style = b.style;
    o.emissive = b.emissive;
    o.hue = b.ext.y;
    o.clip = g.view_proj * vec4<f32>(o.world_pos, 1.0);
    return o;
}

@vertex
fn vs_shadow(@location(0) pos: vec3<f32>, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
    let b = balls_buf[ii];
    let s = select(1.0, b.ext.x, b.ext.x > 0.0);
    return g.shadow_mat * vec4<f32>(b.pos + qrot(b.rot, vec3<f32>(pos.x, pos.y * s, pos.z)) * b.radius, 1.0);
}

const PHI: f32 = 1.61803398875;

fn hue_rgb(h: f32) -> vec3<f32> {
    let k = vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0);
    return clamp(abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn soccer(o: vec3<f32>) -> vec3<f32> {
    // Nearest of 12 icosahedron (pentagons) and 20 dodecahedron (hexagons) points.
    var best = -2.0;
    var second = -2.0;
    var pent = false;
    let ip = 1.0 / PHI;
    for (var i = 0; i < 32; i++) {
        var p: vec3<f32>;
        let k = i % 8;
        let sx = select(-1.0, 1.0, (k & 1) == 0);
        let sy = select(-1.0, 1.0, (k & 2) == 0);
        let sz = select(-1.0, 1.0, (k & 4) == 0);
        if (i < 4) { p = vec3<f32>(0.0, sx, sy * PHI); }
        else if (i < 8) { p = vec3<f32>(sx, sy * PHI, 0.0); }
        else if (i < 12) { p = vec3<f32>(sx * PHI, 0.0, sy); }
        else if (i < 20) { p = vec3<f32>(sx, sy, sz); }
        else if (i < 24) { p = vec3<f32>(0.0, sx * ip, sy * PHI); }
        else if (i < 28) { p = vec3<f32>(sx * ip, sy * PHI, 0.0); }
        else { p = vec3<f32>(sx * PHI, 0.0, sy * ip); }
        let d = dot(o, normalize(p));
        if (d > best) { second = best; best = d; pent = i < 12; }
        else if (d > second) { second = d; }
    }
    let seam = 1.0 - smoothstep(0.0, 0.012, best - second);
    let base = select(vec3<f32>(0.85), vec3<f32>(0.02), pent);
    return mix(base, vec3<f32>(0.15), seam * 0.8);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    var n = normalize(in.normal);
    let o = normalize(in.obj);
    let v = normalize(g.cam_pos.xyz - p);
    let up = up_at(p);
    let sun = light_at(p);
    let surf = world_to_surf(p);
    let shadow = shadow_at(p, n) * cloud_shadow(surf.x, surf.y);
    let sun_rad = g.sun_color.rgb * sun.w * shadow;

    var albedo = vec3<f32>(0.8);
    var gloss = 0.3;
    var metal = 0.0;
    var emissive = vec3<f32>(0.0);
    switch (in.style) {
        case 0u: {   // beach ball gores
            let a = atan2(o.z, o.x) / TAU + 0.5;
            let sector = u32(floor(a * 6.0)) % 3u;
            albedo = select(select(vec3<f32>(0.05, 0.2, 0.8), vec3<f32>(0.9), sector == 1u), vec3<f32>(0.85, 0.06, 0.04), sector == 0u);
            if (abs(o.y) > 0.93) { albedo = vec3<f32>(0.9); }
            gloss = 0.6;
        }
        case 1u: { albedo = soccer(o); gloss = 0.4; }
        case 2u: {
            let line = min(min(abs(o.x), abs(o.y)), abs(abs(o.z) - 0.72 + 0.35 * o.x * o.x));
            albedo = mix(vec3<f32>(0.02), vec3<f32>(0.75, 0.28, 0.06) * mix(0.8, 1.0, value_noise(o.xy * 60.0)), smoothstep(0.012, 0.025, line));
            gloss = 0.25;
        }
        case 3u: {   // bowling ball: marbled, glossy, finger holes
            let m = fbm3(o.xy * 3.0 + o.zz * 2.0);
            albedo = mix(vec3<f32>(0.02, 0.03, 0.12), vec3<f32>(0.25, 0.05, 0.4), smoothstep(0.4, 0.7, m));
            let h1 = distance(o, normalize(vec3<f32>(0.0, 0.9, 0.3)));
            let h2 = distance(o, normalize(vec3<f32>(0.15, 0.9, 0.05)));
            let h3 = distance(o, normalize(vec3<f32>(-0.15, 0.9, 0.05)));
            if (min(h1, min(h2, h3)) < 0.09) { albedo = vec3<f32>(0.0); }
            gloss = 0.95;
        }
        case 4u: {   // glowing orb
            let rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
            let pulse = 0.85 + 0.15 * sin(g.cam_pos.w * 3.0 + p.x);
            emissive = vec3<f32>(0.35, 0.85, 1.0) * in.emissive * pulse * (1.0 + rim * 2.0);
            albedo = vec3<f32>(0.1);
        }
        case 5u: {   // boulder
            let r = fbm3(o.xy * 4.0 + o.zx * 3.0);
            albedo = mix(vec3<f32>(0.2, 0.18, 0.16), vec3<f32>(0.38, 0.35, 0.31), r);
            n = normalize(n + (vec3<f32>(value_noise(o.yz * 9.0), value_noise(o.zx * 9.0), value_noise(o.xy * 9.0)) - 0.5) * 0.5);
            gloss = 0.05;
        }
        case 6u: { albedo = vec3<f32>(0.95); metal = 1.0; gloss = 0.98; }
        case 8u: {   // pressure suit: team colour panels over light-grey fabric
            let panel = smoothstep(0.35, 0.45, abs(o.x)) + smoothstep(0.6, 0.7, o.y);
            let team = mix(vec3<f32>(0.18), hue_rgb(in.hue), 0.85);
            albedo = mix(vec3<f32>(0.72, 0.72, 0.70), team, clamp(1.0 - panel, 0.0, 1.0)) * mix(0.85, 1.0, value_noise(o.xz * 40.0));
            gloss = 0.25;
            emissive = hue_rgb(in.hue) * in.emissive;
        }
        case 9u: {   // helmet visor: gold-coated glass
            albedo = vec3<f32>(0.9, 0.7, 0.3);
            metal = 1.0;
            gloss = 0.97;
            emissive = vec3<f32>(0.2, 0.6, 1.0) * in.emissive;
        }
        case 10u: { albedo = vec3<f32>(0.12, 0.12, 0.13); metal = 0.7; gloss = 0.6; }   // weapon
        case 11u: { albedo = vec3<f32>(0.75, 0.88, 0.95); metal = 1.0; gloss = 0.95; }  // glass shard
        case 12u: { albedo = mix(vec3<f32>(0.35), hue_rgb(in.hue), 0.35); gloss = 0.15; } // debris
        case 13u: {  // micro black hole: nothing escapes; Hawking glow at the rim
            let rim = pow(1.0 - max(dot(n, v), 0.0), 4.0);
            albedo = vec3<f32>(0.0);
            gloss = 0.0;
            emissive = vec3<f32>(1.0, 0.55, 0.2) * rim * in.emissive;
        }
        default: {   // helium balloon
            albedo = vec3<f32>(0.85, 0.08, 0.12);
            gloss = 0.8;
            emissive = vec3<f32>(0.6, 0.05, 0.08) * pow(clamp(dot(v, -sun.xyz), 0.0, 1.0), 3.0) * sun.w;
        }
    }

    let ndv = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    let fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
    let r = reflect(-v, n);
    let env_k = dot(r, up) * 0.5 + 0.5;
    var env = mix(g.bounce_color.rgb * 1.2, g.sky_color.rgb * 1.8 + g.bounce_color.rgb * 0.3, env_k);
    env += g.sun_color.rgb * sun.w * shadow * pow(max(dot(r, sun.xyz), 0.0), mix(8.0, 3000.0, gloss)) * mix(1.0, 80.0, gloss);
    // Grass and ground occlude the lower half.
    let ao = mix(0.45, 1.0, smoothstep(-0.6, 0.3, dot(n, up)));
    let col = albedo * (1.0 - metal) * (sun_rad * max(dot(n, sun.xyz), 0.0) + ambient(n, up) + orb_light(p, n) * 0.3) * ao
            + env * fres * mix(0.2, 1.0, gloss) * ao + emissive;
    return vec4<f32>(atmosphere(col, p), 1.0);
}
