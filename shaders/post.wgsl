// ============================================================================
//  post.wgsl — HDR post-processing (standalone module, no common.wgsl).
//    bloom_down : 13-tap downsample (first pass: soft threshold + Karis average)
//    bloom_up   : 3x3 tent upsample, additively blended
//    composite  : exposure, bloom, AgX / ACES tonemapping, grading, vignette
//    final      : FXAA + upscale to the window + contrast-adaptive sharpening
// ============================================================================

struct PostParams {
    texel: vec4<f32>,     // src texel size xy, threshold, knee
    tone: vec4<f32>,      // exposure, bloom strength, tonemapper (0 AgX, 1 ACES), sharpen
    misc: vec4<f32>,      // fxaa on, vignette, time, saturation
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> pp: PostParams;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: VsOut;
    o.clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return o;
}

fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn tap(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(src_tex, src_samp, uv, 0.0).rgb;
    return select(min(c, vec3<f32>(1e4)), vec3<f32>(0.0), c != c);   // never let a NaN/Inf spread
}

fn karis(c: vec3<f32>) -> f32 { return 1.0 / (1.0 + lum(c)); }

fn downsample13(uv: vec2<f32>, first: bool) -> vec3<f32> {
    let t = pp.texel.xy;
    let a = tap(uv + t * vec2<f32>(-2.0, -2.0));
    let b = tap(uv + t * vec2<f32>(0.0, -2.0));
    let c = tap(uv + t * vec2<f32>(2.0, -2.0));
    let d = tap(uv + t * vec2<f32>(-2.0, 0.0));
    let e = tap(uv);
    let f = tap(uv + t * vec2<f32>(2.0, 0.0));
    let gg = tap(uv + t * vec2<f32>(-2.0, 2.0));
    let h = tap(uv + t * vec2<f32>(0.0, 2.0));
    let i = tap(uv + t * vec2<f32>(2.0, 2.0));
    let j = tap(uv + t * vec2<f32>(-1.0, -1.0));
    let k = tap(uv + t * vec2<f32>(1.0, -1.0));
    let l = tap(uv + t * vec2<f32>(-1.0, 1.0));
    let m = tap(uv + t * vec2<f32>(1.0, 1.0));
    if (first) {
        // Karis average per 2x2 group kills fireflies from tiny HDR highlights.
        let g0 = (a + b + d + e) * 0.25;
        let g1 = (b + c + e + f) * 0.25;
        let g2 = (d + e + gg + h) * 0.25;
        let g3 = (e + f + h + i) * 0.25;
        let g4 = (j + k + l + m) * 0.25;
        let w0 = karis(g0) * 0.125;
        let w1 = karis(g1) * 0.125;
        let w2 = karis(g2) * 0.125;
        let w3 = karis(g3) * 0.125;
        let w4 = karis(g4) * 0.5;
        return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    }
    return e * 0.125 + (a + c + gg + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

@fragment
fn fs_bloom_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let c = downsample13(in.uv, true);
    // Soft knee threshold.
    let br = max(c.r, max(c.g, c.b));
    let th = pp.texel.z;
    let knee = pp.texel.w;
    var soft = clamp(br - th + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    let w = max(soft, br - th) / max(br, 1e-5);
    return vec4<f32>(min(c * w, vec3<f32>(6e4)), 1.0);
}

@fragment
fn fs_bloom_down(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample13(in.uv, false), 1.0);
}

@fragment
fn fs_bloom_up(in: VsOut) -> @location(0) vec4<f32> {
    let t = pp.texel.xy;
    var c = tap(in.uv) * 4.0;
    c += (tap(in.uv + vec2<f32>(t.x, 0.0)) + tap(in.uv - vec2<f32>(t.x, 0.0))
        + tap(in.uv + vec2<f32>(0.0, t.y)) + tap(in.uv - vec2<f32>(0.0, t.y))) * 2.0;
    c += tap(in.uv + t) + tap(in.uv - t) + tap(in.uv + vec2<f32>(t.x, -t.y)) + tap(in.uv + vec2<f32>(-t.x, t.y));
    return vec4<f32>(c / 16.0, 1.0);
}

// ---- tonemapping ---------------------------------------------------------------
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Minimal AgX (Benjamin Wrensch's fit), punchy look.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
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
    // punchy look
    let l = lum(x);
    x = pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.25));
    x = l + 1.25 * (x - l);
    x = mi * x;
    // AgX outputs display-referred sRGB-ish; convert to linear for the sRGB target.
    return pow(clamp(x, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    var hdr = tap(in.uv);
    let bloom = textureSampleLevel(bloom_tex, src_samp, in.uv, 0.0).rgb;
    hdr = mix(hdr, bloom, pp.tone.y);
    hdr *= pp.tone.x;
    var c: vec3<f32>;
    if (pp.tone.z < 0.5) { c = agx(hdr); } else { c = aces(hdr); }
    let l = lum(c);
    c = max(mix(vec3<f32>(l), c, pp.misc.w), vec3<f32>(0.0));
    let d = in.uv - 0.5;
    c *= 1.0 - pp.misc.y * dot(d, d) * 1.6;
    return vec4<f32>(c, 1.0);
}

// ---- FXAA (3.11-style, quality tuned for speed) ---------------------------------
fn luma_s(uv: vec2<f32>) -> f32 { return sqrt(lum(tap(uv))); }

fn fxaa(uv: vec2<f32>) -> vec3<f32> {
    let t = pp.texel.xy;
    let rgbM = tap(uv);
    let lM = sqrt(lum(rgbM));
    let lN = luma_s(uv + vec2<f32>(0.0, -t.y));
    let lS = luma_s(uv + vec2<f32>(0.0, t.y));
    let lE = luma_s(uv + vec2<f32>(t.x, 0.0));
    let lW = luma_s(uv + vec2<f32>(-t.x, 0.0));
    let lmin = min(lM, min(min(lN, lS), min(lE, lW)));
    let lmax = max(lM, max(max(lN, lS), max(lE, lW)));
    let range = lmax - lmin;
    if (range < max(0.0312, lmax * 0.125)) { return rgbM; }
    let lNW = luma_s(uv + vec2<f32>(-t.x, -t.y));
    let lNE = luma_s(uv + vec2<f32>(t.x, -t.y));
    let lSW = luma_s(uv + vec2<f32>(-t.x, t.y));
    let lSE = luma_s(uv + vec2<f32>(t.x, t.y));
    let edgeH = abs(lNW + lNE - 2.0 * lN) + 2.0 * abs(lW + lE - 2.0 * lM) + abs(lSW + lSE - 2.0 * lS);
    let edgeV = abs(lNW + lSW - 2.0 * lW) + 2.0 * abs(lN + lS - 2.0 * lM) + abs(lNE + lSE - 2.0 * lE);
    let horizontal = edgeH >= edgeV;
    let l1 = select(lW, lN, horizontal);
    let l2 = select(lE, lS, horizontal);
    let g1 = abs(l1 - lM);
    let g2 = abs(l2 - lM);
    var step_len = select(t.x, t.y, horizontal);
    var lavg: f32;
    let gscaled = 0.25 * max(g1, g2);
    if (g1 >= g2) { step_len = -step_len; lavg = 0.5 * (l1 + lM); } else { lavg = 0.5 * (l2 + lM); }
    var cur = uv;
    if (horizontal) { cur.y += step_len * 0.5; } else { cur.x += step_len * 0.5; }
    let off = select(vec2<f32>(0.0, t.y), vec2<f32>(t.x, 0.0), horizontal);
    var u1 = cur - off;
    var u2 = cur + off;
    var e1 = luma_s(u1) - lavg;
    var e2 = luma_s(u2) - lavg;
    var done1 = abs(e1) >= gscaled;
    var done2 = abs(e2) >= gscaled;
    var steps = array<f32, 8>(1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0, 8.0);
    for (var i = 0; i < 8; i++) {
        if (done1 && done2) { break; }
        if (!done1) { u1 -= off * steps[i]; e1 = luma_s(u1) - lavg; done1 = abs(e1) >= gscaled; }
        if (!done2) { u2 += off * steps[i]; e2 = luma_s(u2) - lavg; done2 = abs(e2) >= gscaled; }
    }
    let d1 = select(uv.y - u1.y, uv.x - u1.x, horizontal);
    let d2 = select(u2.y - uv.y, u2.x - uv.x, horizontal);
    let near_d = min(d1, d2);
    let span = d1 + d2;
    let correct = ((select(e2, e1, d1 < d2)) < 0.0) != (lM < lavg);
    var px = select(0.0, 0.5 - near_d / span, correct);
    // sub-pixel aliasing
    let lavg_all = (2.0 * (lN + lS + lE + lW) + lNW + lNE + lSW + lSE) / 12.0;
    let sub1 = clamp(abs(lavg_all - lM) / range, 0.0, 1.0);
    let sub2 = (-2.0 * sub1 + 3.0) * sub1 * sub1;
    px = max(px, sub2 * sub2 * 0.75);
    var fuv = uv;
    if (horizontal) { fuv.y += px * step_len; } else { fuv.x += px * step_len; }
    return tap(fuv);
}

@fragment
fn fs_final(in: VsOut) -> @location(0) vec4<f32> {
    var c: vec3<f32>;
    if (pp.misc.x > 0.5) { c = fxaa(in.uv); } else { c = tap(in.uv); }
    // Contrast-adaptive sharpening (recovers detail lost to upscaling / FXAA).
    if (pp.tone.w > 0.0) {
        let t = pp.texel.xy;
        let n = tap(in.uv + vec2<f32>(0.0, -t.y));
        let s = tap(in.uv + vec2<f32>(0.0, t.y));
        let e = tap(in.uv + vec2<f32>(t.x, 0.0));
        let w = tap(in.uv + vec2<f32>(-t.x, 0.0));
        let mn = min(c, min(min(n, s), min(e, w)));
        let mx = max(c, max(max(n, s), max(e, w)));
        let amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0)));
        let wgt = -amp * mix(0.125, 0.2, pp.tone.w);
        c = clamp((c + (n + s + e + w) * wgt) / (1.0 + 4.0 * wgt), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    // Tiny animated dither: kills banding in the dark sky.
    let noise = fract(sin(dot(in.clip.xy + pp.misc.z, vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5;
    c += noise / 255.0;
    return vec4<f32>(c, 1.0);
}
