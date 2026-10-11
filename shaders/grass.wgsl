// ============================================================================
//  grass.wgsl — blade rendering (no vertex buffers; triangle strip per blade).
//    near LOD: 6 segments = 13 vertices, far LOD: 2 segments = 5 vertices.
// ============================================================================

@group(1) @binding(0) var<storage, read> blades: array<Blade>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(perspective, centroid) world_pos: vec3<f32>,
    @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,
    @location(2) @interpolate(perspective, centroid) params: vec4<f32>,    // t, hash, tint, dryness
    @location(3) @interpolate(flat) kind: u32,
    @location(4) far_blend: f32,
};

fn width_profile(kind: u32, t: f32) -> f32 {
    let omt = 1.0 - t;
    switch (kind) {
        case 1u: {   // wheat: thin stem, plump ear on top
            return mix(0.8, 3.2, smoothstep(0.72, 0.8, t)) * (1.0 - smoothstep(0.93, 1.0, t)) + 0.15;
        }
        case 3u: { return mix(0.9, 2.4, smoothstep(0.55, 0.65, t)) * pow(omt, 0.3); }   // lavender spike
        case 4u: { return mix(1.0, 3.5, smoothstep(0.82, 0.9, t)) * pow(omt, 0.25); }   // canola flowers
        case 5u: { return mix(0.9, 7.0, smoothstep(0.86, 0.92, t)) * pow(omt, 0.2); }   // wild flower head
        default: { return pow(omt, 0.55); }
    }
}

fn blade_vertex(vi: u32, ii: u32, segs: u32) -> VsOut {
    let b = blades[ii];
    var t = f32(vi >> 1u) / f32(segs);
    var side = f32(vi & 1u) * 2.0 - 1.0;
    if (vi >= 2u * segs) { t = 1.0; side = 0.0; }
    let pk = unpack4x8unorm(b.packed);
    let kind = u32(round(pk.w * 7.0));

    let omt = 1.0 - t;
    let spine = 2.0 * omt * t * b.ctrl + t * t * b.tip;
    let tangent = normalize(2.0 * omt * b.ctrl + 2.0 * t * (b.tip - b.ctrl) + up_at(b.root) * 1e-4);
    let th = angle_of(b.root);
    let wdir = east_of(th) * cos(b.facing) + vec3<f32>(0.0, 0.0, sin(b.facing));
    let nc = cross(wdir, tangent);
    var n = select(up_at(b.root), nc * inverseSqrt(max(dot(nc, nc), 1e-12)), dot(nc, nc) > 1e-8);

    let centre = b.root + spine;
    let to_cam = g.cam_pos.xyz - centre;
    let dist = length(to_cam);
    let v = to_cam / max(dist, 1e-4);

    var w = b.width * width_profile(kind, t);
    let edge_on = 1.0 - abs(dot(n, v));
    w *= 1.0 + edge_on * edge_on * 0.9;
    let world = centre + wdir * (side * w * 0.5);

    let far_blend = smoothstep(g.grass.z * 0.5, g.grass.y, dist);
    n = normalize(mix(normalize(n + wdir * side * 0.5), up_at(b.root), far_blend * 0.6));

    var o: VsOut;
    o.clip = g.view_proj * vec4<f32>(bh_xf_at(world, b.root), 1.0);
    o.world_pos = world;
    o.normal = n;
    o.params = vec4<f32>(t, pk.x, pk.y, pk.z);
    o.kind = kind;
    o.far_blend = far_blend;
    return o;
}

@vertex
fn vs_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    return blade_vertex(vi, ii, 6u);
}
@vertex
fn vs_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    return blade_vertex(vi, ii, 2u);
}

fn flower_color(h: f32) -> vec3<f32> {
    if (h < 0.3) { return vec3<f32>(0.9, 0.85, 0.8); }        // daisy white
    if (h < 0.55) { return vec3<f32>(0.85, 0.65, 0.05); }     // buttercup
    if (h < 0.8) { return vec3<f32>(0.75, 0.06, 0.04); }      // poppy
    return vec3<f32>(0.15, 0.25, 0.8);                        // cornflower
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // MSAA may evaluate slightly outside the blade: keep everything finite.
    let nn = dot(in.normal, in.normal);
    var n = select(up_at(in.world_pos), in.normal * inverseSqrt(max(nn, 1e-12)), nn > 1e-8);
    if (!front) { n = -n; }
    let t = clamp(in.params.x, 0.0, 1.0);
    let hsh = in.params.y;
    let tint = in.params.z;
    let dry = in.params.w;
    let p = in.world_pos;

    var albedo: vec3<f32>;
    var translucency = 1.0;
    switch (in.kind) {
        case 1u: {   // wheat / barley
            let stem = mix(vec3<f32>(0.20, 0.22, 0.05), vec3<f32>(0.52, 0.42, 0.14), smoothstep(0.1, 0.6, t));
            let ear = mix(vec3<f32>(0.62, 0.48, 0.18), vec3<f32>(0.72, 0.64, 0.38), tint);
            albedo = mix(stem, ear, smoothstep(0.7, 0.8, t)) * mix(0.85, 1.1, hsh);
            translucency = 0.7;
        }
        case 2u: {
            albedo = mix(vec3<f32>(0.03, 0.08, 0.01), vec3<f32>(0.12, 0.32, 0.03), t) * mix(0.9, 1.15, hsh);
        }
        case 3u: {
            let stem = mix(vec3<f32>(0.05, 0.08, 0.04), vec3<f32>(0.16, 0.22, 0.12), t);
            albedo = mix(stem, vec3<f32>(0.24, 0.12, 0.45) * mix(0.8, 1.2, hsh), smoothstep(0.55, 0.65, t));
            translucency = 0.5;
        }
        case 4u: {
            let stem = mix(vec3<f32>(0.04, 0.10, 0.02), vec3<f32>(0.14, 0.30, 0.04), t);
            albedo = mix(stem, vec3<f32>(0.85, 0.72, 0.04), smoothstep(0.82, 0.9, t));
        }
        case 5u: {
            let stem = mix(vec3<f32>(0.03, 0.07, 0.01), vec3<f32>(0.12, 0.25, 0.04), t);
            albedo = mix(stem, flower_color(tint), smoothstep(0.86, 0.92, t));
        }
        case 7u: {   // debug view: red = trampled, blue = bend
            albedo = vec3<f32>(hsh, 0.04, tint) * mix(0.3, 1.0, t);
        }
        default: {
            let root_col = vec3<f32>(0.020, 0.045, 0.010);
            let mid_col = mix(vec3<f32>(0.075, 0.20, 0.025), vec3<f32>(0.11, 0.25, 0.03), hsh);
            let tip_col = mix(vec3<f32>(0.20, 0.36, 0.06), vec3<f32>(0.30, 0.40, 0.09), tint);
            albedo = mix(root_col, mid_col, smoothstep(0.0, 0.45, t));
            albedo = mix(albedo, tip_col, smoothstep(0.45, 1.0, t)) * mix(0.85, 1.12, tint);
            let dry_col = mix(vec3<f32>(0.26, 0.22, 0.09), vec3<f32>(0.48, 0.42, 0.20), t);
            albedo = mix(albedo, dry_col, dry * 0.5 * smoothstep(0.1, 0.6, t));
        }
    }

    let up = up_at(p);
    let sun = light_at(p);
    let l = sun.xyz;
    let v = normalize(g.cam_pos.xyz - p);
    let surf = world_to_surf(p);
    let shadow = shadow_at(p, up) * cloud_shadow(surf.x, surf.y);
    let sun_rad = g.sun_color.rgb * sun.w * shadow;

    let ao = mix(0.12, 1.0, pow(t, 1.3));
    let ndl = dot(n, l);
    let diffuse = clamp((ndl + 0.35) / 1.35, 0.0, 1.0);
    let back = pow(clamp(dot(v, -l), 0.0, 1.0), 3.0);
    let trans = (back * 0.85 + max(-ndl, 0.0) * 0.3) * smoothstep(0.15, 0.9, t) * translucency;
    let hv = normalize(l + v);
    let spec = pow(max(dot(n, hv), 0.0), 40.0) * 0.18 * t * (1.0 - in.far_blend);

    var col = albedo * (sun_rad * diffuse + ambient(n, up) + orb_light(p, n)) * ao
            + sun_rad * (albedo * vec3<f32>(1.25, 1.45, 0.55) * trans + vec3<f32>(spec)) * ao;
    col = atmosphere(col, p);
    col = select(min(col, vec3<f32>(64.0)), vec3<f32>(0.0), col != col);
    return vec4<f32>(col, 1.0);
}
