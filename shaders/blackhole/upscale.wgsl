// Edge-preserving (joint bilateral) upscale of the black-hole pass to the
// output resolution. The guide is the per-pixel depth key (distance to the
// first opaque surface: −1 for the shadow, 1e4 for the sky); taps whose key
// differs from the nearest tap are suppressed, so the shadow edge, the disk
// rim and the photon ring stay sharp instead of being smeared.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var low_tex: texture_2d<f32>;
@group(0) @binding(2) var out_hdr: texture_storage_2d<rgba16float, write>;

fn key_of(k: f32) -> f32 {
    return log2(2.0 + max(k, -1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full = vec2<u32>(P.flags.z, P.flags.w);
    if (gid.x >= full.x || gid.y >= full.y) {
        return;
    }
    let low = vec2<f32>(f32(P.size.x), f32(P.size.y));
    let fp = (vec2<f32>(gid.xy) + 0.5) * low / vec2<f32>(full) - 0.5;
    let base = vec2<i32>(floor(fp));
    let f = fp - floor(fp);
    let lim = vec2<i32>(P.size.xy) - 1;
    let near = clamp(vec2<i32>(round(fp)), vec2<i32>(0), lim);
    let kn = key_of(textureLoad(low_tex, near, 0).w);
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var j = 0; j < 2; j = j + 1) {
        for (var i = 0; i < 2; i = i + 1) {
            let q = clamp(base + vec2<i32>(i, j), vec2<i32>(0), lim);
            let t = textureLoad(low_tex, q, 0);
            let wb = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let wk = exp(-4.0 * abs(key_of(t.w) - kn));
            let wgt = wb * wk + 1e-5;
            acc += t.rgb * wgt;
            wsum += wgt;
        }
    }
    textureStore(out_hdr, gid.xy, vec4<f32>(acc / wsum, 1.0));
}
