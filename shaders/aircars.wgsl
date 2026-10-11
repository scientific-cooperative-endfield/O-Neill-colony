// ============================================================================
//  aircars.wgsl — flying traffic. Positions are evaluated procedurally from
//  instance_index and time: no CPU work, no buffers. Lanes run along the
//  valleys at several altitudes and in rings around the axis.
//  Two pipelines: solid bodies, and additive light glows that stay visible
//  as streams of light across the whole cylinder at night.
// ============================================================================

struct Car {
    pos: vec3<f32>,
    fwd: vec3<f32>,
    up: vec3<f32>,
    hue: f32,
    scale: f32,
};

fn car_state(i: u32) -> Car {
    let t = g.cam_pos.w;
    var h = pcg(i * 7u + 13u);
    let r0 = u2f(h); h = pcg(h);
    let r1 = u2f(h); h = pcg(h);
    let r2 = u2f(h); h = pcg(h);
    let r3 = u2f(h); h = pcg(h);
    let lane = i % 10u;
    let dir = select(-1.0, 1.0, r3 > 0.5);
    var c: Car;
    c.hue = u2f(h);
    c.scale = 1.0;
    if (lane < 8u) {
        // Valley lanes: organised altitude layers.
        let k = f32(i % 3u);
        let hw = half_land_width() - 80.0;
        let s = (floor(r0 * 12.0) / 11.0 * 2.0 - 1.0) * hw + (r2 - 0.5) * 8.0;
        let alt = 30.0 + floor(r1 * 5.0) * 28.0 + select(0.0, 6.0, dir > 0.0);
        let speed = 40.0 + r2 * 45.0;
        let len = colony_len();
        let v = fract(r3 + dir * speed * t / len) * len;
        c.pos = surf_to_world(k * STRIP_ANGLE * colony_r() + s, v, alt);
        c.fwd = vec3<f32>(0.0, 0.0, dir);
        c.up = up_at(c.pos);
        c.scale = smoothstep(0.0, 200.0, min(v, len - v));
    } else {
        // Ring lanes circling the axis high above the land (and the windows).
        let rad = colony_r() - 260.0 - floor(r1 * 4.0) * 70.0;
        let z = 500.0 + floor(r0 * 8.0) * 760.0 + select(0.0, 8.0, dir > 0.0);
        let speed = 50.0 + r2 * 40.0;
        let th = r3 * TAU + dir * speed * t / rad;
        c.pos = vec3<f32>(rad * cos(th), rad * sin(th), z);
        c.fwd = east_of(th) * dir;
        c.up = up_at(c.pos);
    }
    return c;
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) obj: vec3<f32>,
    @location(3) hue: f32,
};

@vertex
fn vs_body(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) part: f32,
           @builtin(instance_index) ii: u32) -> VsOut {
    let c = car_state(ii);
    let right = normalize(cross(c.fwd, c.up));
    let size = vec3<f32>(2.2, 1.3, 5.0) * c.scale;
    let m = pos * size;
    var o: VsOut;
    o.world_pos = c.pos + right * m.x + c.up * m.y + c.fwd * m.z;
    o.normal = normalize(right * normal.x + c.up * normal.y + c.fwd * normal.z);
    o.obj = pos;
    o.hue = c.hue;
    o.clip = g.view_proj * vec4<f32>(bh_xf_at(o.world_pos, c.pos), 1.0);
    return o;
}

fn car_paint(h: f32) -> vec3<f32> {
    if (h < 0.3) { return vec3<f32>(0.85, 0.86, 0.88); }
    if (h < 0.5) { return vec3<f32>(0.03, 0.03, 0.035); }
    if (h < 0.7) { return vec3<f32>(0.6, 0.05, 0.04); }
    if (h < 0.85) { return vec3<f32>(0.05, 0.2, 0.6); }
    return vec3<f32>(0.9, 0.6, 0.1);
}

@fragment
fn fs_body(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let n = normalize(in.normal);
    let v = normalize(g.cam_pos.xyz - p);
    let up = up_at(p);
    let sun = light_at(p);
    let albedo = car_paint(in.hue);
    let canopy = step(0.2, in.obj.y) * step(in.obj.z, 0.25);
    let fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let r = reflect(-v, n);
    let env = mix(g.bounce_color.rgb, g.sky_color.rgb * 1.8, dot(r, up) * 0.5 + 0.5)
            + g.sun_color.rgb * sun.w * pow(max(dot(r, sun.xyz), 0.0), 200.0) * 20.0;
    var col = mix(albedo, vec3<f32>(0.02, 0.03, 0.05), canopy) * (g.sun_color.rgb * sun.w * max(dot(n, sun.xyz), 0.0) + ambient(n, up))
            + env * fres * 0.6;
    // under-glow strip
    col += vec3<f32>(0.1, 0.8, 1.0) * (1.0 - smoothstep(0.0, 0.08, abs(in.obj.y + 0.0))) * 3.0;
    return vec4<f32>(atmosphere(col, p), 1.0);
}

// ---- light glows ----------------------------------------------------------------
struct GlowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_glow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GlowOut {
    let c = car_state(ii / 2u);
    let head = (ii & 1u) == 0u;
    let lp = c.pos + c.fwd * select(-2.6, 2.6, head) * c.scale;
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
    let to_cam = g.cam_pos.xyz - lp;
    let dist = length(to_cam);
    let fwd = to_cam / dist;
    let right = normalize(cross(fwd, c.up));
    let upv = cross(right, fwd);
    let size = max(0.35, dist * 0.0018) * c.scale;
    var o: GlowOut;
    o.clip = g.view_proj * vec4<f32>(bh_xf_at(lp + (right * corner.x + upv * corner.y) * size, c.pos), 1.0);
    o.uv = corner;
    let night = g.bounce_color.w;
    let fade = exp(-dist * g.sky_color.w * 0.7);
    let base = select(vec3<f32>(1.0, 0.08, 0.04), vec3<f32>(0.85, 0.92, 1.0), head);
    // Keep perceived brightness roughly constant with distance (point light).
    o.color = base * (0.04 + night * 2.0) * fade * clamp(dist / 60.0, 1.0, mix(3.0, 10.0, night));
    return o;
}

@fragment
fn fs_glow(in: GlowOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    let a = exp(-r2 * 5.0) * (1.0 - smoothstep(0.7, 1.0, r2));
    return vec4<f32>(in.color * a, 0.0);
}
