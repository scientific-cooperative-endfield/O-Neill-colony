// ============================================================================
//  grass_cull.wgsl — procedural blade generation + GPU culling (compute).
//
//  A camera-centred grid of candidates in surface coordinates (u, v), anchored
//  to the world so blades never swim. One workgroup = a 16x16 tile, rejected as
//  a whole first (distance / window strip / frustum sphere). Each surviving
//  candidate picks its vegetation from the zone map: meadow grass with wild
//  flowers, mown city lawns, or farm crops planted in rows (wheat, barley,
//  lavender, canola, young sprouts). Shaped by clumps and the dynamics field,
//  then appended to the near / far LOD buffers with per-workgroup atomics.
// ============================================================================

struct DrawArgs {
    vertex_count:   u32,
    instance_count: atomic<u32>,
    first_vertex:   u32,
    first_instance: u32,
};
struct Args {
    near: DrawArgs,
    far:  DrawArgs,
};

@group(1) @binding(0) var field_state: texture_2d<f32>;
@group(1) @binding(1) var field_trample: texture_2d<f32>;
@group(1) @binding(2) var field_sampler: sampler;
@group(1) @binding(3) var<storage, read_write> near_blades: array<Blade>;
@group(1) @binding(4) var<storage, read_write> far_blades: array<Blade>;
@group(1) @binding(5) var<storage, read_write> args: Args;

const TILE: u32 = 16u;
const CLUMP_SIZE: f32 = 1.1;

var<workgroup> wg_visible: u32;
var<workgroup> wg_near_count: atomic<u32>;
var<workgroup> wg_far_count: atomic<u32>;
var<workgroup> wg_near_base: u32;
var<workgroup> wg_far_base: u32;

fn sphere_visible(c: vec3<f32>, r: f32) -> bool {
    for (var i = 0; i < 6; i++) {
        let pl = g.frustum[i];
        if (dot(pl.xyz, c) + pl.w < -r) { return false; }
    }
    return true;
}

struct Candidate {
    blade: Blade,
    kind: u32,   // 0 rejected, 1 near, 2 far
};

fn make_blade(cell: vec2<i32>) -> Candidate {
    var out: Candidate;
    out.kind = 0u;

    let spacing = g.grass.x;
    let radius = g.grass.y;
    let cam = g.cam_pos.xyz;

    var h = hash2(cell) ^ bitcast<u32>(g.grid.w);
    h = pcg(h); let r0 = u2f(h);
    h = pcg(h); let r1 = u2f(h);
    h = pcg(h); let r2 = u2f(h);
    h = pcg(h); let r3 = u2f(h);
    h = pcg(h); let r4 = u2f(h);
    h = pcg(h); let r5 = u2f(h);
    h = pcg(h); let r6 = u2f(h);
    h = pcg(h); let r7 = u2f(h);

    var u = (f32(cell.x) + r0) * spacing;
    let v = (f32(cell.y) + r1) * spacing;
    if (v < 2.0 || v > colony_len() - 2.0) { return out; }

    // Cheap rejections first: distance thinning by camera distance estimate.
    let th0 = fract(u / circumference()) * TAU;
    let st = strip_of(th0);
    let strip = st.x;
    var s = st.y;
    if (abs(s) > half_land_width() - 3.0) { return out; }          // windows / edge promenade

    // Cheap distance estimate in surface coordinates before any texture fetch.
    let flat = vec2<f32>(u - g.cam_surf.x, v - g.cam_surf.y);
    let flat2 = dot(flat, flat);
    if (flat2 > radius * radius) { return out; }
    let est = sqrt(flat2 + max(g.cam_surf.z - 2.0, 0.0) * max(g.cam_surf.z - 2.0, 0.0) * 0.8);
    if (r2 > clamp(pow(g.grass.w / max(est, g.grass.w), 1.55), 0.05, 1.0) * 1.02) { return out; }

    let ground = sample_height(u, v);
    if (ground < 0.15) { return out; }                                 // water & wet shore
    let approx = surf_to_world(u, v, ground);
    let dist = distance(approx, cam);
    if (dist > radius) { return out; }
    let keep = clamp(pow(g.grass.w / max(dist, g.grass.w), 1.55), 0.05, 1.0);
    if (r2 > keep) { return out; }

    let zone = sample_zone(u, v);
    if (road_mask(s, v, zone.z) > 0.5 || (zone.z > 0.35 && road_dist(s, v) < ROAD_HALF + 2.7)) { return out; }

    // ---- vegetation type --------------------------------------------------------
    var vkind = K_GRASS;
    var height = 0.0;
    var width = 0.0;
    var tint = 0.0;
    var facing = 0.0;
    var aligned = false;
    var stiff = 1.0;
    let is_farm = zone.y > mix(0.35, 0.65, r6);
    if (is_farm) {
        let fc = farm_cell(s, v, strip);
        if (fc.y < 1.8 || (fc.x > 3.5 && fc.x < 4.5)) { return out; }   // tracks, fallow
        let crop = u32(fc.x);
        // Plant in rows running along the valley (v).
        var row = 0.42;
        if (crop == 2u) { row = 1.3; }                                  // lavender bushes
        let s_row = round(s / row) * row + (r0 - 0.5) * select(0.06, 0.45, crop == 2u);
        u += s_row - s;
        s = s_row;
        tint = fc.z;
        aligned = true;
        facing = (r5 - 0.5) * 0.6;
        switch (crop) {
            case 0u: { vkind = K_WHEAT; height = mix(0.8, 1.05, r3); width = 0.009; stiff = 0.9; }
            case 1u: { vkind = K_YOUNG; height = mix(0.18, 0.32, r3); width = 0.02; stiff = 1.6; }
            case 2u: { vkind = K_LAVENDER; height = mix(0.4, 0.62, r3); width = 0.011; stiff = 1.3; }
            case 3u: { vkind = K_CANOLA; height = mix(0.85, 1.2, r3); width = 0.014; stiff = 1.1; }
            default: { vkind = K_WHEAT; height = mix(0.65, 0.85, r3); width = 0.008; tint = 1.0; stiff = 0.9; }
        }
        // Thinning compensation stays, density is uniform in fields.
    } else {
        var density = zone.x;
        let n = surf_normal_local(u, v, 1.0);
        density *= 1.0 - smoothstep(0.25, 0.45, 1.0 - n.y);
        density *= mix(1.0, 0.5, zone.w);                               // forest floor
        if (r7 * 0.999 > density) { return out; }
        // Clumps (Voronoi)
        let pp = vec2<f32>(u, v);
        let ci = vec2<i32>(floor(pp / CLUMP_SIZE));
        var best_d = 1e9;
        var best_h = 0u;
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let cc = ci + vec2<i32>(x, y);
                let ch = hash2(cc + vec2<i32>(9137, -4721));
                let ch2 = pcg(ch);
                let centre = (vec2<f32>(cc) + vec2<f32>(u2f(ch), u2f(ch2))) * CLUMP_SIZE;
                let dd = dot(pp - centre, pp - centre);
                if (dd < best_d) { best_d = dd; best_h = pcg(ch2); }
            }
        }
        let cr0 = u2f(best_h);
        let cr1 = u2f(pcg(best_h));
        let cr2 = u2f(pcg(pcg(best_h)));
        let tall = smoothstep(0.35, 0.8, value_noise(pp * 0.028 + vec2<f32>(41.0, 3.0)));
        height = mix(0.22, 0.9, tall) * mix(0.75, 1.25, cr0) * mix(0.65, 1.15, r3) * mix(0.6, 1.0, density);
        height *= 1.0 - 0.35 * smoothstep(0.0, CLUMP_SIZE * 0.8, sqrt(best_d));
        height *= mix(1.0, 0.7, zone.w);
        if (zone.z > 0.5) { height = mix(0.07, 0.14, r3); }             // mown city lawn
        width = mix(0.010, 0.022, r4) * mix(0.85, 1.2, cr1);
        tint = cr1;
        let ra = r5 * TAU;
        let ca = cr2 * TAU;
        let fv = mix(vec2<f32>(cos(ra), sin(ra)), vec2<f32>(cos(ca), sin(ca)), 0.45) + vec2<f32>(1e-4, 0.0);
        facing = atan2(fv.y, fv.x);
        if (zone.z < 0.5 && r4 > 0.975 && tall > 0.2) {                // wild flowers
            vkind = K_FLOWER;
            height = mix(0.3, 0.6, r3);
            width = 0.007;
            tint = r6;
        }
    }
    height *= 1.0 - smoothstep(radius * 0.78, radius, dist);
    if (height < 0.03) { return out; }
    width *= min(inverseSqrt(keep), 3.5);

    let th = fract(u / circumference()) * TAU;
    let root = surf_to_world(u, v, ground - 0.02);
    let up = vec3<f32>(-cos(th), -sin(th), 0.0);
    if (!sphere_visible(root + up * (height * 0.5), height * 0.8 + width)) { return out; }

    // ---- dynamics ------------------------------------------------------------------
    let bend_axis = vec2<f32>(-sin(facing), cos(facing));
    var d = bend_axis * mix(0.06, 0.3, r6) * select(-1.0, 1.0, r7 > 0.5);
    if (aligned) { d *= 0.4; }

    let cs = g.field_params.x;
    let half_win = g.field_params.y * cs * 0.5 - 2.0;
    let rel = vec2<f32>(u - g.cam_surf.x, v - g.cam_surf.y);
    var bend = vec2<f32>(0.0);
    var tr = vec4<f32>(0.0);
    if (max(abs(rel.x), abs(rel.y)) < half_win) {
        let fuv = vec2<f32>(u, v) / (cs * g.field_params.y);
        bend = textureSampleLevel(field_state, field_sampler, fuv, 0.0).xy;
        tr = textureSampleLevel(field_trample, field_sampler, fuv, 0.0);
    } else {
        bend = wind_goal(vec2<f32>(u, v), g.cam_pos.w);
    }
    let flex = mix(0.7, 1.3, r4) * (0.55 + 0.6 * min(height, 1.0)) / stiff;
    let t = g.cam_pos.w;
    let flutter = sin(t * mix(4.0, 8.5, r3) + r0 * TAU + dot(vec2<f32>(u, v), g.wind.xy) * 1.3)
                  * 0.035 * (0.25 + g.wind.z) * height;
    d += bend * flex + bend_axis * flutter;
    let tl = length(tr.yz);
    let flat_dir = select(bend_axis, tr.yz / max(tl, 1e-4), tl > 1e-3);
    let fl = clamp(tr.x * mix(0.75, 1.15, r5), 0.0, 1.0);
    d = mix(d, flat_dir + d * 0.15, fl);
    let dl = length(d);
    if (dl > 0.96) { d *= 0.96 / dl; }

    // Bezier with length preservation, built in the local (east, up, north) frame.
    let horiz = d * height;
    let vert = height * sqrt(max(1.0 - dot(d, d), 0.0)) + 0.01;
    var p2 = vec3<f32>(horiz.x, vert, horiz.y);
    var p1 = vec3<f32>(horiz.x * 0.15, vert, horiz.y * 0.15);
    let l0 = length(p2);
    let l1 = length(p1) + length(p2 - p1);
    let ratio = height / max((2.0 * l0 + l1) / 3.0, 1e-4);
    p1 *= ratio;
    p2 = p1 + (p2 - p1) * ratio;

    let east = east_of(th);
    let north = vec3<f32>(0.0, 0.0, 1.0);
    let dry = clamp(smoothstep(0.55, 0.85, value_noise(vec2<f32>(u, v) * 0.05 + vec2<f32>(77.0, 9.0))) * 0.8 + r6 * 0.2, 0.0, 1.0);
    var pk = vec4<f32>(r3, tint, dry, f32(vkind) / 7.0);
    if (g.field_params.w > 0.5) {
        pk = vec4<f32>(fl, clamp(length(bend), 0.0, 1.0), 0.0, 7.0 / 7.0);   // debug view
    }

    out.blade.root = root;
    out.blade.width = width;
    out.blade.ctrl = east * p1.x + up * p1.y + north * p1.z;
    out.blade.packed = pack4x8unorm(pk);
    out.blade.tip = east * p2.x + up * p2.y + north * p2.z;
    out.blade.facing = facing;
    out.kind = select(2u, 1u, dist < g.grass.z);
    return out;
}

@compute @workgroup_size(16, 16)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if (lidx == 0u) {
        let sp = g.grass.x;
        let size = f32(TILE + 1u) * sp;
        let mn = vec2<f32>(g.grid.xy + vec2<i32>(wid.xy * TILE)) * sp;
        let c = mn + size * 0.5;
        let th = fract(c.x / circumference()) * TAU;
        let st = strip_of(th);
        let hc = sample_height(c.x, c.y);
        let centre = surf_to_world(c.x, c.y, hc + 1.0);
        let rad = size * 0.75 + 3.0;
        var vis = 1u;
        if (abs(st.y) > half_land_width() + size
            || distance(centre, g.cam_pos.xyz) - rad > g.grass.y
            || c.y < -size || c.y > colony_len() + size
            || !sphere_visible(centre, rad)) {
            vis = 0u;
        }
        wg_visible = vis;
    }
    if (workgroupUniformLoad(&wg_visible) == 0u) { return; }

    let cand = make_blade(g.grid.xy + vec2<i32>(gid.xy));
    var slot = 0u;
    if (cand.kind == 1u) {
        slot = atomicAdd(&wg_near_count, 1u);
    } else if (cand.kind == 2u) {
        slot = atomicAdd(&wg_far_count, 1u);
    }
    workgroupBarrier();
    if (lidx == 0u) {
        let nn = atomicLoad(&wg_near_count);
        let nf = atomicLoad(&wg_far_count);
        if (nn > 0u) { wg_near_base = atomicAdd(&args.near.instance_count, nn); }
        if (nf > 0u) { wg_far_base = atomicAdd(&args.far.instance_count, nf); }
    }
    workgroupBarrier();
    if (cand.kind == 1u) {
        let idx = wg_near_base + slot;
        if (idx < arrayLength(&near_blades)) { near_blades[idx] = cand.blade; }
    } else if (cand.kind == 2u) {
        let idx = wg_far_base + slot;
        if (idx < arrayLength(&far_blades)) { far_blades[idx] = cand.blade; }
    }
}

@compute @workgroup_size(1)
fn cs_finalize() {
    let nmax = arrayLength(&near_blades);
    if (atomicLoad(&args.near.instance_count) > nmax) { atomicStore(&args.near.instance_count, nmax); }
    let fmax = arrayLength(&far_blades);
    if (atomicLoad(&args.far.instance_count) > fmax) { atomicStore(&args.far.instance_count, fmax); }
}
