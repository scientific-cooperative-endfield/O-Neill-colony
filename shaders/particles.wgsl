// ============================================================================
//  particles.wgsl — camera-facing sprites: smoke / dust / vapour (lit, alpha),
//  flashes (additive), sparks (velocity-stretched), muzzle stars, shock rings.
//  Premultiplied alpha: rgb is added, alpha darkens what is behind.
// ============================================================================

struct Particle {
    pos: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    axis: vec3<f32>,   // streak vector (sparks), world units
    kind: u32,         // 0 smoke, 1 glow, 2 spark, 3 muzzle star, 4 ring
};
@group(1) @binding(0) var<storage, read> parts: array<Particle>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) ii: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let pt = parts[ii];
    var corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    let c = corners[vi];
    let d = normalize(pt.pos - g.cam_pos.xyz);
    var wp: vec3<f32>;
    if (pt.kind == 2u) {
        let side = normalize(cross(d, pt.axis + vec3<f32>(1e-6, 0.0, 0.0)));
        wp = pt.pos + pt.axis * (c.y * 0.5 - 0.5) + side * c.x * pt.size;
    } else {
        var up = up_at(pt.pos);
        if (abs(dot(up, d)) > 0.98) { up = vec3<f32>(0.0, 0.0, 1.0); }
        let r = normalize(cross(d, up));
        let u = cross(r, d);
        let a = f32(ii % 97u) * 2.39996;
        let rc = vec2<f32>(c.x * cos(a) - c.y * sin(a), c.x * sin(a) + c.y * cos(a));
        let cc = select(rc, c, pt.kind == 3u);
        wp = pt.pos + (r * cc.x + u * cc.y) * pt.size;
    }
    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(wp, 1.0);
    o.uv = c;
    o.ii = ii;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pt = parts[in.ii];
    let r = length(in.uv);
    switch (pt.kind) {
        case 0u: {
            let n = value_noise(in.uv * 2.2 + vec2<f32>(f32(in.ii % 13u) * 3.7, f32(in.ii % 7u) * 1.3));
            let a = smoothstep(1.0, 0.15, r) * (0.55 + 0.45 * n) * pt.color.a;
            let up = up_at(pt.pos);
            let sun = light_at(pt.pos);
            let lit = g.sun_color.rgb * sun.w * (0.35 + 0.25 * max(dot(sun.xyz, up), 0.0)) + ambient(up, up) * 1.1;
            return vec4<f32>(pt.color.rgb * lit * a, a);
        }
        case 2u: {
            let a = (1.0 - abs(in.uv.x)) * smoothstep(-1.0, 0.6, in.uv.y);
            return vec4<f32>(pt.color.rgb * a * pt.color.a, 0.0);
        }
        case 3u: {
            let ang = atan2(in.uv.y, in.uv.x) + f32(in.ii % 17u);
            let spikes = pow(abs(cos(ang * 2.5)), 6.0) * smoothstep(1.0, 0.0, r);
            let core = exp(-r * r * 9.0);
            let a = (core * 1.6 + spikes * 0.9) * pt.color.a;
            return vec4<f32>(pt.color.rgb * a, 0.0);
        }
        case 4u: {
            let a = smoothstep(0.75, 0.92, r) * smoothstep(1.0, 0.94, r) * pt.color.a;
            return vec4<f32>(pt.color.rgb * a, 0.0);
        }
        default: {
            let a = exp(-r * r * 4.0) * smoothstep(1.0, 0.7, r) * pt.color.a;
            return vec4<f32>(pt.color.rgb * a, 0.0);
        }
    }
}
