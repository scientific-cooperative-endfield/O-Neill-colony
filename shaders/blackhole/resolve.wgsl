// Resolves the sky behind the traced rays: samples the HDR cubemap along the
// asymptotic escape direction with a mip level derived from the angular spread
// of neighbouring rays (strong lensing compresses the sky → higher mip, no sparkle).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var emit_tex: texture_2d<f32>;
@group(0) @binding(2) var dir_tex: texture_2d<f32>;
@group(0) @binding(3) var sky_tex: texture_cube<f32>;
@group(0) @binding(4) var samp_clamp: sampler;
@group(0) @binding(5) var lut_tex: texture_2d<f32>;
@group(0) @binding(6) var out_hdr: texture_storage_2d<rgba16float, write>;

fn bb(t: f32) -> vec3<f32> {
    return textureSampleLevel(lut_tex, samp_clamp, bb_uv(t), 0.0).rgb;
}

fn nb_angle(d: vec3<f32>, q: vec2<i32>) -> f32 {
    let sz = vec2<i32>(i32(P.size.x), i32(P.size.y));
    let qq = clamp(q, vec2<i32>(0), sz - 1);
    let nd = textureLoad(dir_tex, qq, 0).xyz;
    if (dot(nd, nd) <= 0.0) {
        return 0.0;
    }
    return length(normalize(nd) - d);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.size.x || gid.y >= P.size.y) {
        return;
    }
    let q = vec2<i32>(gid.xy);
    let e = textureLoad(emit_tex, q, 0);
    let dv = textureLoad(dir_tex, q, 0);
    var col = e.rgb;
    let g2 = dot(dv.xyz, dv.xyz);
    if (g2 > 0.0 && e.a > 0.0) {
        let g = sqrt(g2);
        let d = dv.xyz / g;
        let a = max(max(nb_angle(d, q + vec2<i32>(1, 0)), nb_angle(d, q - vec2<i32>(1, 0))),
                    max(nb_angle(d, q + vec2<i32>(0, 1)), nb_angle(d, q - vec2<i32>(0, 1))));
        let lod = clamp(log2(max(a, 1e-6) / P.misc.y), 0.0, P.misc.z - 1.0);
        var c = textureSampleLevel(sky_tex, samp_clamp, d, lod).rgb * P.disk2.w;
        let tint = bb(6500.0 * g) / bb(6500.0);
        c = c * tint * (g * g * g * g);
        col += e.a * c;
    }
    textureStore(out_hdr, q, vec4<f32>(col, dv.w));
}
