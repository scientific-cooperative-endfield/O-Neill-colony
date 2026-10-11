// Post-processing of the black-hole frame (all compute):
//   meter  — centre- and highlight-weighted log-average luminance per 16×16 tile (shadow excluded)
//   expose — reduces the tiles, eye-adaptation in log space
//   down   — 13-tap bloom downsample (Karis average on the first level)
//   up     — 3×3 tent upsample + add (progressive, energy conserving)
//   final  — bloom mix, exposure, chromatic aberration, AgX/ACES, grain + dither
// The bloom is mixed (lerp, strength ≈ 4 %), never added on top, so it cannot
// wash out the black-hole shadow.

struct PostParams {
    size: vec4<u32>,  // output w, h, frame, tonemap (0 AgX, 1 ACES)
    a: vec4<f32>,     // dt (s), exposure compensation (EV), adaptation speed (1/s), instant (1 = snap)
    b: vec4<f32>,     // bloom strength, chromatic aberration, grain, vignette
    c: vec4<f32>,     // tile count x, tile count y, min exposure, max exposure
    d: vec4<f32>,     // x = 1 / bloom level count
};

@group(0) @binding(0) var<uniform> PP: PostParams;

fn lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ---- metering ------------------------------------------------------------------
@group(0) @binding(1) var meter_src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> tiles: array<vec2<f32>>;

var<workgroup> sh_sum: array<vec2<f32>, 256>;

@compute @workgroup_size(16, 16)
fn meter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
         @builtin(workgroup_id) wid: vec3<u32>) {
    // each thread meters one pixel of a 4× decimated grid
    let p = gid.xy * 4u + 2u;
    var v = vec2<f32>(0.0);
    if (p.x < PP.size.x && p.y < PP.size.y) {
        let l = lum(textureLoad(meter_src, vec2<i32>(p), 0).rgb);
        if (l > 1e-5) {
            let uv = vec2<f32>(p) / vec2<f32>(PP.size.xy) - 0.5;
            // centre-weighted and highlight-weighted (∝ √L): the dim sky must not
            // drive the exposure up until the disk saturates
            let wgt = (1.0 - 0.7 * min(dot(uv, uv) * 3.0, 1.0)) * sqrt(min(l, 1e4));
            v = vec2<f32>(log2(l) * wgt, wgt);
        }
    }
    sh_sum[li] = v;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) {
            break;
        }
        if (li < s) {
            sh_sum[li] = sh_sum[li] + sh_sum[li + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (li == 0u) {
        tiles[wid.y * u32(PP.c.x) + wid.x] = sh_sum[0];
    }
}

@group(0) @binding(3) var<storage, read_write> expo: array<f32, 4>;

@compute @workgroup_size(256)
fn expose(@builtin(local_invocation_index) li: u32) {
    let n = u32(PP.c.x) * u32(PP.c.y);
    var v = vec2<f32>(0.0);
    for (var i = li; i < n; i = i + 256u) {
        v += tiles[i];
    }
    sh_sum[li] = v;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) {
            break;
        }
        if (li < s) {
            sh_sum[li] = sh_sum[li] + sh_sum[li + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (li == 0u) {
        let t = sh_sum[0];
        var avg_ev = -4.0;
        if (t.y > 0.5) {
            avg_ev = t.x / t.y;
        }
        // target: map the average to middle grey (0.18) + compensation
        let target_ev = clamp(log2(0.18) - avg_ev + PP.a.y, log2(PP.c.z), log2(PP.c.w));
        var cur = expo[1];
        if (PP.a.w > 0.5 || expo[2] < 0.5) {
            cur = target_ev;
        } else {
            cur = cur + (target_ev - cur) * (1.0 - exp(-PP.a.x * PP.a.z));
        }
        expo[0] = exp2(cur);
        expo[1] = cur;
        expo[2] = 1.0;
        expo[3] = avg_ev;
    }
}

// ---- bloom ---------------------------------------------------------------------
@group(0) @binding(4) var b_src: texture_2d<f32>;
@group(0) @binding(5) var b_samp: sampler;
@group(0) @binding(6) var b_dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var b_base: texture_2d<f32>;

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(b_src, b_samp, uv, 0.0).rgb;
}
fn karis(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + lum(c));
}

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dsz = textureDimensions(b_dst);
    if (gid.x >= dsz.x || gid.y >= dsz.y) {
        return;
    }
    let ts = 1.0 / vec2<f32>(textureDimensions(b_src));
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dsz);
    let a = tap(uv + ts * vec2<f32>(-2.0, -2.0));
    let b = tap(uv + ts * vec2<f32>(0.0, -2.0));
    let c = tap(uv + ts * vec2<f32>(2.0, -2.0));
    let d = tap(uv + ts * vec2<f32>(-2.0, 0.0));
    let e = tap(uv);
    let f = tap(uv + ts * vec2<f32>(2.0, 0.0));
    let g = tap(uv + ts * vec2<f32>(-2.0, 2.0));
    let h = tap(uv + ts * vec2<f32>(0.0, 2.0));
    let i = tap(uv + ts * vec2<f32>(2.0, 2.0));
    let j = tap(uv + ts * vec2<f32>(-1.0, -1.0));
    let k = tap(uv + ts * vec2<f32>(1.0, -1.0));
    let l = tap(uv + ts * vec2<f32>(-1.0, 1.0));
    let m = tap(uv + ts * vec2<f32>(1.0, 1.0));
    var o: vec3<f32>;
    if (all(textureDimensions(b_src) == PP.size.xy)) {
        // first level: Karis average per 2×2 group suppresses firefly flicker
        let g0 = (a + b + d + e) * 0.25;
        let g1 = (b + c + e + f) * 0.25;
        let g2 = (d + e + g + h) * 0.25;
        let g3 = (e + f + h + i) * 0.25;
        let g4 = (j + k + l + m) * 0.25;
        let w0 = karis(g0) * 0.125;
        let w1 = karis(g1) * 0.125;
        let w2 = karis(g2) * 0.125;
        let w3 = karis(g3) * 0.125;
        let w4 = karis(g4) * 0.5;
        o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    } else {
        o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    textureStore(b_dst, gid.xy, vec4<f32>(max(o, vec3<f32>(0.0)), 1.0));
}

@compute @workgroup_size(8, 8)
fn up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dsz = textureDimensions(b_dst);
    if (gid.x >= dsz.x || gid.y >= dsz.y) {
        return;
    }
    let ts = 1.0 / vec2<f32>(textureDimensions(b_src));
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dsz);
    var s = tap(uv) * 4.0;
    s += (tap(uv + vec2<f32>(ts.x, 0.0)) + tap(uv - vec2<f32>(ts.x, 0.0)) + tap(uv + vec2<f32>(0.0, ts.y)) + tap(uv - vec2<f32>(0.0, ts.y))) * 2.0;
    s += tap(uv + ts) + tap(uv - ts) + tap(uv + vec2<f32>(ts.x, -ts.y)) + tap(uv + vec2<f32>(-ts.x, ts.y));
    let base = textureLoad(b_base, vec2<i32>(gid.xy), 0).rgb;
    textureStore(b_dst, gid.xy, vec4<f32>(base + s / 16.0, 1.0));
}

// ---- final ---------------------------------------------------------------------
@group(0) @binding(8) var f_hdr: texture_2d<f32>;
@group(0) @binding(9) var f_bloom: texture_2d<f32>;
@group(0) @binding(10) var f_samp: sampler;
@group(0) @binding(11) var<storage, read> f_expo: array<f32, 4>;
@group(0) @binding(12) var f_out: texture_storage_2d<rgba8unorm, write>;

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
// AgX (Benjamin Wrensch's fit); returns display-encoded values.
fn agx(c: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let mi = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var x = m * max(c, vec3<f32>(1e-10));
    x = clamp((log2(x) - min_ev) / (max_ev - min_ev), vec3<f32>(0.0), vec3<f32>(1.0));
    x = agx_contrast(x);
    // "punchy" look: a little more contrast and saturation than base AgX, so the
    // Doppler colour asymmetry of the disk survives the highlight desaturation
    x = pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.15));
    x = vec3<f32>(lum(x)) + 1.3 * (x - vec3<f32>(lum(x)));
    x = mi * x;
    return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn srgb_oetf(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}
fn hash(p: vec2<u32>, f: u32) -> f32 {
    var h = p.x * 1973u + p.y * 9277u + f * 26699u;
    h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12u)) * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffu) / 65535.0;
}

@compute @workgroup_size(8, 8)
fn final_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= PP.size.x || gid.y >= PP.size.y) {
        return;
    }
    let res = vec2<f32>(PP.size.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / res;
    let d = uv - 0.5;
    var hdr: vec3<f32>;
    if (PP.b.y > 0.0) {
        // lateral chromatic aberration, grows towards the frame edge
        let off = d * PP.b.y * dot(d, d) * 4.0;
        hdr = vec3<f32>(
            textureSampleLevel(f_hdr, f_samp, uv - off, 0.0).r,
            textureSampleLevel(f_hdr, f_samp, uv, 0.0).g,
            textureSampleLevel(f_hdr, f_samp, uv + off, 0.0).b);
    } else {
        hdr = textureLoad(f_hdr, vec2<i32>(gid.xy), 0).rgb;
    }
    let bloom = textureSampleLevel(f_bloom, f_samp, uv, 0.0).rgb;
    hdr = mix(hdr, bloom * PP.d.x, PP.b.x);
    hdr *= f_expo[0];
    hdr *= 1.0 - PP.b.w * dot(d, d) * 1.4;
    var c: vec3<f32>;
    if (PP.size.w == 0u) {
        c = agx(hdr);
    } else {
        c = srgb_oetf(aces(hdr));
    }
    // film grain (luminance-weighted, animated) + 1/255 triangular dither
    let n1 = hash(gid.xy, PP.size.z);
    let n2 = hash(gid.xy + vec2<u32>(17u, 59u), PP.size.z);
    let l = lum(c);
    c += (n1 - 0.5) * PP.b.z * 0.09 * (0.25 + l * (1.0 - l) * 3.0);
    c += (n1 + n2 - 1.0) / 255.0;
    textureStore(f_out, gid.xy, vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
