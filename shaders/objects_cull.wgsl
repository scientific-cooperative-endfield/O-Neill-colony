// ============================================================================
//  objects_cull.wgsl — GPU-driven culling + LOD for static instances
//  (buildings, trees, track). Writes per-mesh buckets for the camera view and
//  a separate caster list for the shadow map, then the renderer issues one
//  draw_indexed_indirect per bucket.
// ============================================================================

struct ObjInstance {
    pos:   vec3<f32>,
    yaw:   f32,
    size:  vec3<f32>,
    style: u32,        // bits 0-7 mesh kind, 8-15 material, 16-31 seed
};

struct DrawIndexedArgs {
    index_count:    u32,
    instance_count: atomic<u32>,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
};

struct KindInfo {
    lod_dist: f32,
    max_dist: f32,
    shadow_dist: f32,
    _pad: f32,
};

struct CullParams {
    counts: vec4<u32>,              // instance count, kind count, shadows on, -
    kinds: array<KindInfo, 16>,
    buckets: array<vec4<u32>, 48>,  // x = base, y = capacity
};

@group(1) @binding(0) var<storage, read> instances: array<ObjInstance>;
@group(1) @binding(1) var<storage, read_write> out_instances: array<ObjInstance>;
@group(1) @binding(2) var<storage, read_write> draws: array<DrawIndexedArgs>;
@group(1) @binding(3) var<uniform> cp: CullParams;
// Surviving spatial tiles: x = first instance, y = prefix sum of counts.
@group(1) @binding(4) var<storage, read> ranges: array<vec2<u32>>;

fn instance_index(i: u32) -> u32 {
    var lo = 0u;
    var hi = cp.counts.w;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (ranges[mid].y <= i) { lo = mid; } else { hi = mid; }
    }
    return ranges[lo].x + (i - ranges[lo].y);
}

fn sphere_visible(c: vec3<f32>, r: f32) -> bool {
    for (var i = 0; i < 6; i++) {
        let pl = g.frustum[i];
        if (dot(pl.xyz, c) + pl.w < -r) { return false; }
    }
    return true;
}

fn push(bucket: u32, inst: ObjInstance) {
    let b = cp.buckets[bucket];
    let slot = atomicAdd(&draws[bucket].instance_count, 1u);
    if (slot < b.y) {
        out_instances[b.x + slot] = inst;
    }
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= cp.counts.x) { return; }
    let inst = instances[instance_index(i)];
    let kind = inst.style & 0xffu;
    let info = cp.kinds[kind];
    let up = up_at(inst.pos);
    let c = inst.pos + up * (inst.size.y * 0.5);
    let r = 0.5 * length(inst.size);
    let d = distance(c, g.cam_pos.xyz) - r;
    let nk = cp.counts.y;

    if (d < info.max_dist && sphere_visible(c, r)) {
        let lod = select(0u, 1u, d > info.lod_dist);
        push(kind * 2u + lod, inst);
    }
    // Shadow casters: everything near the camera in the camera's valley.
    if (cp.counts.z != 0u && d < info.shadow_dist) {
        push(nk * 2u + kind, inst);
    }
}

@compute @workgroup_size(1)
fn cs_finalize() {
    let n = cp.counts.y * 3u;
    for (var b = 0u; b < n; b++) {
        let cap = cp.buckets[b].y;
        if (atomicLoad(&draws[b].instance_count) > cap) { atomicStore(&draws[b].instance_count, cap); }
    }
}
