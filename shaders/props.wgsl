// ============================================================================
//  props.wgsl — rigid Blender-built meshes (operatives, weapons, casings).
//  One instance = one part: a 3×4 transform, team tint, effect flags.
// ============================================================================

struct PropInst {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    tint: vec4<f32>,   // rgb team colour, w: light strength (0 = dead suit)
    ext: vec4<f32>,    // x: shimmer (spawn protection), y: 0 world / 1 viewmodel / 2 shadow only, z: hit flash
};
@group(1) @binding(0) var<storage, read> props: array<PropInst>;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mat: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mat: vec4<f32>,
    @location(4) @interpolate(flat) ii: u32,
    @location(5) obj: vec3<f32>,
};

fn xf(i: PropInst, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(i.r0.xyz, p) + i.r0.w, dot(i.r1.xyz, p) + i.r1.w, dot(i.r2.xyz, p) + i.r2.w);
}

@vertex
fn vs_main(v: VsIn, @builtin(instance_index) ii: u32) -> VsOut {
    let inst = props[ii];
    var o: VsOut;
    o.world_pos = xf(inst, v.pos);
    o.normal = normalize(vec3<f32>(dot(inst.r0.xyz, v.normal), dot(inst.r1.xyz, v.normal), dot(inst.r2.xyz, v.normal)));
    o.color = v.color;
    o.mat = v.mat;
    o.ii = ii;
    o.obj = v.pos;
    var c = g.view_proj * vec4<f32>(o.world_pos, 1.0);
    if (inst.ext.y > 0.5 && inst.ext.y < 1.5) {
        // viewmodel: squeeze into the front half of the (reversed) depth range so it never clips into walls
        c.z = 0.5 * c.w + 0.5 * c.z;
    } else if (inst.ext.y > 1.5) {
        c = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    }
    o.clip = c;
    return o;
}

@vertex
fn vs_shadow(v: VsIn, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
    let inst = props[ii];
    if (inst.ext.y > 0.5 && inst.ext.y < 1.5) {
        return vec4<f32>(0.0, 0.0, -1.0, 1.0);
    }
    return g.shadow_mat * vec4<f32>(xf(inst, v.pos), 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let inst = props[in.ii];
    let p = in.world_pos;
    var n = normalize(in.normal);
    let v = normalize(g.cam_pos.xyz - p);
    if (dot(n, v) < -0.2) { n = -n; }
    let up = up_at(p);
    let sun = light_at(p);
    let surf = world_to_surf(p);
    let shadow = shadow_at(p, n) * cloud_shadow(surf.x, surf.y);
    let sun_rad = g.sun_color.rgb * sun.w * shadow;

    let flags = u32(in.mat.z * 255.0 + 0.5);
    var albedo = in.color.rgb * in.color.rgb;
    let ao = in.color.a;
    let metal = in.mat.x;
    let gloss = 1.0 - in.mat.y;
    var emissive = vec3<f32>(0.0);
    if ((flags & 1u) != 0u) { albedo *= inst.tint.rgb; }
    if ((flags & 2u) != 0u) {
        let ec = select(albedo, inst.tint.rgb, (flags & 1u) != 0u);
        emissive = ec * in.mat.w * 255.0 / 16.0 * inst.tint.w;
    }
    albedo *= mix(0.9, 1.06, value_noise(in.obj.xy * 300.0 + in.obj.zz * 170.0));

    let ndv = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    var fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
    if ((flags & 4u) != 0u) { fres = max(fres, vec3<f32>(0.25)); }
    let r = reflect(-v, n);
    let env_k = dot(r, up) * 0.5 + 0.5;
    var env = mix(g.bounce_color.rgb * 1.2, g.sky_color.rgb * 1.8 + g.bounce_color.rgb * 0.3, env_k);
    env += g.sun_color.rgb * sun.w * shadow * pow(max(dot(r, sun.xyz), 0.0), mix(8.0, 2500.0, gloss * gloss)) * mix(1.0, 60.0, gloss * gloss);
    let sky_ao = mix(0.55, 1.0, smoothstep(-0.7, 0.4, dot(n, up)));
    let diffuse = sun_rad * max(dot(n, sun.xyz), 0.0) + (ambient(n, up) + orb_light(p, n) * 0.3) * ao * sky_ao;
    var col = albedo * (1.0 - metal) * diffuse + env * fres * mix(0.15, 1.0, gloss) * ao * sky_ao + emissive;
    if (inst.ext.x > 0.0) {
        let scan = 0.5 + 0.5 * sin(dot(p, up) * 40.0 - g.cam_pos.w * 8.0);
        col += inst.tint.rgb * inst.ext.x * (0.3 + 1.5 * pow(1.0 - ndv, 3.0)) * scan;
    }
    col = mix(col, vec3<f32>(3.0, 2.6, 2.2), inst.ext.z * 0.6);
    return vec4<f32>(atmosphere(col, p), 1.0);
}
