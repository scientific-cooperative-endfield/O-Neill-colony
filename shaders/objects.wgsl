// ============================================================================
//  objects.wgsl — buildings, trees, maglev track, trains. Instances are read
//  from a storage buffer (GPU-culled bucket or CPU-written dynamic list).
//  Facades are fully procedural: floors, windows lit at night, neon bands,
//  rooftop gardens, blinking aviation beacons.
// ============================================================================

struct ObjInstance {
    pos:   vec3<f32>,
    yaw:   f32,
    size:  vec3<f32>,
    style: u32,
};

@group(1) @binding(0) var<storage, read> objs: array<ObjInstance>;

const M_GLASS: u32 = 0u;
const M_WHITE: u32 = 1u;
const M_RESIDENTIAL: u32 = 2u;
const M_CONCRETE: u32 = 3u;
const M_NEON: u32 = 4u;
const M_CONIFER: u32 = 5u;
const M_BROADLEAF: u32 = 6u;
const M_TRAIN: u32 = 7u;
const M_GREENHOUSE: u32 = 8u;
const M_STATION: u32 = 9u;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) part: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) obj: vec3<f32>,                     // object-space position (m)
    @location(3) obj_n: vec3<f32>,
    @location(4) @interpolate(flat) style: u32,
    @location(5) size: vec3<f32>,
    @location(6) part: f32,
};

struct Basis { x: vec3<f32>, y: vec3<f32>, z: vec3<f32> };

fn basis(inst: ObjInstance) -> Basis {
    let th = angle_of(inst.pos);
    let e = east_of(th);
    let up = up_at(inst.pos);
    let nz = vec3<f32>(0.0, 0.0, 1.0);
    let c = cos(inst.yaw);
    let s = sin(inst.yaw);
    var b: Basis;
    b.x = e * c + nz * s;
    b.y = up;
    b.z = -e * s + nz * c;
    return b;
}

fn transform(inst: ObjInstance, lp: vec3<f32>) -> vec3<f32> {
    let b = basis(inst);
    var m = lp * inst.size;
    let mat = (inst.style >> 8u) & 0xffu;
    if (mat == M_CONIFER || mat == M_BROADLEAF) {
        // Wind sway grows with height.
        let seed = f32(inst.style >> 16u) * 0.001;
        let k = lp.y * lp.y * inst.size.y * 0.012 * (0.4 + g.wind.z);
        let ph = g.cam_pos.w * 1.3 + seed * 17.0 + inst.pos.z * 0.05;
        m += vec3<f32>(g.wind.x, 0.0, g.wind.y) * k * (1.0 + 0.4 * sin(ph)) + vec3<f32>(sin(ph * 1.7), 0.0, cos(ph * 1.3)) * k * 0.25;
    }
    return inst.pos + b.x * m.x + b.y * m.y + b.z * m.z;
}

@vertex
fn vs_main(in: VsIn, @builtin(instance_index) ii: u32) -> VsOut {
    let inst = objs[ii];
    let b = basis(inst);
    let wp = transform(inst, in.pos);
    let ns = in.normal / inst.size;
    var o: VsOut;
    o.world_pos = wp;
    o.normal = normalize(b.x * ns.x + b.y * ns.y + b.z * ns.z);
    o.obj = in.pos * inst.size;
    o.obj_n = in.normal;
    o.style = inst.style;
    o.size = inst.size;
    o.part = in.part;
    o.clip = g.view_proj * vec4<f32>(bh_xf_at(wp, inst.pos), 1.0);
    return o;
}

@vertex
fn vs_shadow(in: VsIn, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
    let inst = objs[ii];
    return g.shadow_mat * vec4<f32>(bh_xf_at(transform(inst, in.pos), inst.pos), 1.0);
}

struct Surface {
    albedo: vec3<f32>,
    emissive: vec3<f32>,
    rough: f32,           // 0 = mirror-like glass, 1 = matte
    metal: f32,
    trans: f32,
};

fn facade(o: vec3<f32>, on: vec3<f32>, size: vec3<f32>, mat: u32, seed: u32, night: f32) -> Surface {
    var sf: Surface;
    sf.albedo = vec3<f32>(0.6);
    sf.emissive = vec3<f32>(0.0);
    sf.rough = 0.8;
    sf.metal = 0.0;
    sf.trans = 0.0;
    let rs = u2f(seed);
    let roof = on.y > 0.5;
    let top = o.y > size.y - 0.5;

    // Horizontal coordinate along the facade.
    let hcoord = select(o.x, o.z, abs(on.x) > abs(on.z)) + select(0.0, size.x, abs(on.x) > abs(on.z));
    let floor_h = select(3.6, 3.2, mat == M_RESIDENTIAL);
    let col_w = select(2.4, 3.0, mat == M_RESIDENTIAL);
    let fi = floor(o.y / floor_h);
    let ci = floor(hcoord / col_w);
    let fx = fract(hcoord / col_w);
    let fy = fract(o.y / floor_h);
    let wh = hash3(vec3<i32>(i32(ci), i32(fi), i32(seed & 0xffffu)));
    let lit_frac = mix(0.1, 0.72, night) * mix(0.6, 1.3, rs);
    let lit = step(u2f(wh), lit_frac);
    let warm = mix(vec3<f32>(1.0, 0.66, 0.34), vec3<f32>(0.78, 0.88, 1.0), step(0.8, u2f(pcg(wh))));

    if (roof) {
        sf.albedo = select(vec3<f32>(0.22, 0.22, 0.23), vec3<f32>(0.08, 0.17, 0.05), rs > 0.6);
        // Aviation beacons on tall towers.
        if (size.y > 70.0) {
            let corner = min(min(o.x + size.x * 0.5, size.x * 0.5 - o.x), min(o.z + size.z * 0.5, size.z * 0.5 - o.z));
            let blink = step(0.5, fract(g.cam_pos.w * 0.8 + rs));
            sf.emissive = vec3<f32>(1.0, 0.05, 0.02) * (1.0 - smoothstep(0.5, 1.2, corner)) * blink * 40.0;
        }
        return sf;
    }

    switch (mat) {
        case 0u: {   // curtain-wall glass tower
            let mull = max(step(0.94, fx), step(0.97, fy));
            let spandrel = step(0.78, fy) * (1.0 - mull);
            let mech = select(0.0, 1.0, (i32(fi) % 14) == 13);           // plant floor
            let louver = mech * step(0.5, fract(o.y * 1.6));
            let pane = mix(0.85, 1.12, u2f(pcg(wh ^ 0x9e37u)));                 // per-panel tint jitter
            let glass = mix(vec3<f32>(0.035, 0.06, 0.085), vec3<f32>(0.07, 0.1, 0.11), rs) * pane;
            var alb = mix(glass, vec3<f32>(0.1, 0.11, 0.12), spandrel);
            alb = mix(alb, vec3<f32>(0.2, 0.21, 0.22) * mix(0.7, 1.0, louver), mech);
            sf.albedo = mix(alb, vec3<f32>(0.5, 0.52, 0.55), mull);
            sf.rough = mix(mix(0.06, 0.35, spandrel), 0.55, max(mull, mech));
            sf.metal = (1.0 - mull) * (1.0 - mech) * mix(1.0, 0.6, spandrel);
            sf.emissive = warm * lit * (1.0 - mull) * (1.0 - spandrel) * (1.0 - mech) * (0.4 + night * 3.0);
        }
        case 1u: {   // white futuristic panels + light ribbons
            let ribbon = 1.0 - smoothstep(0.0, 0.08, abs(fy - 0.12));
            let win = step(0.25, fy) * step(fy, 0.85);
            sf.albedo = mix(vec3<f32>(0.75, 0.76, 0.78), vec3<f32>(0.05, 0.08, 0.1), win);
            sf.rough = mix(0.5, 0.1, win);
            sf.metal = win * 0.8;
            sf.emissive = warm * lit * win * (0.3 + night * 2.5) + vec3<f32>(0.7, 0.9, 1.0) * ribbon * (0.2 + night * 3.0);
        }
        case 2u: {   // residential: punched windows, balconies, warm walls
            let win = step(0.22, fx) * step(fx, 0.78) * step(0.25, fy) * step(fy, 0.82);
            let wall = mix(vec3<f32>(0.62, 0.52, 0.42), vec3<f32>(0.8, 0.78, 0.72), rs);
            let slab = step(0.93, fy);                                          // balcony slabs
            sf.albedo = mix(mix(wall, wall * 1.15 + 0.05, slab), vec3<f32>(0.04, 0.05, 0.06), win);
            sf.rough = mix(0.9, 0.15, win);
            sf.emissive = vec3<f32>(1.0, 0.68, 0.38) * lit * win * (0.3 + night * 2.8);
        }
        case 4u: {   // dark tower with neon bands
            let band = 1.0 - smoothstep(0.0, 0.15, abs(fract(o.y / (floor_h * 8.0)) - 0.02) * floor_h * 8.0);
            let neon = mix(vec3<f32>(0.1, 0.9, 1.0), vec3<f32>(1.0, 0.15, 0.7), step(0.5, rs));
            let mull = step(0.93, fx);
            sf.albedo = mix(vec3<f32>(0.02, 0.025, 0.03), vec3<f32>(0.15), mull);
            sf.rough = 0.1;
            sf.metal = 1.0;
            sf.emissive = neon * band * (1.5 + night * 8.0) + warm * lit * (1.0 - mull) * (0.2 + night * 1.5);
        }
        case 7u: {   // maglev train car
            let win = step(0.45, o.y / size.y) * step(o.y / size.y, 0.75);
            let stripe = 1.0 - smoothstep(0.0, 0.03, abs(o.y / size.y - 0.32));
            sf.albedo = mix(vec3<f32>(0.85, 0.86, 0.88), vec3<f32>(0.03, 0.04, 0.05), win);
            sf.rough = mix(0.25, 0.05, win);
            sf.metal = win;
            sf.emissive = vec3<f32>(0.2, 0.75, 1.0) * stripe * (1.0 + night * 4.0) + vec3<f32>(1.0, 0.9, 0.75) * win * (0.3 + night * 1.5);
            let front = o.z / size.z;
            if (abs(front) > 0.49) {
                sf.emissive += select(vec3<f32>(1.0, 0.05, 0.02), vec3<f32>(1.0, 0.95, 0.8), front > 0.0) * 6.0;
            }
        }
        case 8u: {   // greenhouse dome: glass over rows of crops, white space-frame
            let ang = atan2(o.z, o.x) * size.x * 0.5;
            let l1 = abs(fract(o.y / 2.4) - 0.5);
            let l2 = abs(fract((ang + o.y * 0.6) / 2.4) - 0.5);
            let l3 = abs(fract((ang - o.y * 0.6) / 2.4) - 0.5);
            let lattice = 1.0 - smoothstep(0.02, 0.05, min(l1, min(l2, l3)));
            let rows = smoothstep(0.3, 0.7, value_noise(vec2<f32>(o.x * 0.6, o.z * 3.0))) * (0.6 + 0.4 * value_noise(o.xz * 0.2));
            let plants = mix(vec3<f32>(0.01, 0.03, 0.012), vec3<f32>(0.05, 0.12, 0.03), rows) * smoothstep(size.y * 0.75, 0.0, o.y);
            sf.albedo = mix(plants, vec3<f32>(0.8, 0.82, 0.84), lattice);
            sf.rough = mix(0.04, 0.5, lattice);
            sf.metal = 0.0;
            sf.emissive = vec3<f32>(0.9, 0.25, 1.0) * (1.0 - lattice) * night * (0.4 + rows * 0.8);
        }
        case 10u: {  // art-deco sandstone: punched windows between vertical piers
            let pier = step(0.8, fx);
            let win = step(0.14, fx) * step(fx, 0.7) * step(0.22, fy) * step(fy, 0.86);
            let stone = mix(vec3<f32>(0.52, 0.44, 0.34), vec3<f32>(0.68, 0.6, 0.48), rs) * mix(0.88, 1.06, value_noise(vec2<f32>(hcoord * 0.4, o.y * 0.3)));
            sf.albedo = mix(stone * mix(1.0, 1.1, pier), vec3<f32>(0.03, 0.035, 0.04), win);
            sf.rough = mix(0.85, 0.2, win);
            sf.emissive = vec3<f32>(1.0, 0.64, 0.32) * lit * win * (0.35 + night * 3.2);
        }
        case 9u: {   // station
            let sign = step(0.85, o.y / size.y);
            sf.albedo = vec3<f32>(0.7, 0.72, 0.74);
            sf.emissive = vec3<f32>(0.3, 0.9, 1.0) * sign * (0.8 + night * 4.0);
        }
        default: {   // concrete (track, pylons)
            sf.albedo = vec3<f32>(0.42, 0.42, 0.41) * mix(0.85, 1.05, value_noise(o.xz * 0.3 + o.yy * 0.1));
            sf.rough = 0.9;
            let lamp = (1.0 - smoothstep(0.0, 0.25, abs(o.y - size.y * 0.5))) * step(0.85, fract(o.z / 20.0));
            sf.emissive = vec3<f32>(0.2, 0.8, 1.0) * lamp * night * 4.0;
        }
    }
    // Street level: shop fronts with awnings on residential / white blocks.
    if ((mat == 1u || mat == 2u) && o.y < 3.6 && size.y > 6.0) {
        let sy = o.y / 3.6;
        let shop = step(0.06, fx) * step(fx, 0.94) * step(0.04, sy) * step(sy, 0.72);
        let awning = step(0.76, sy) * step(sy, 0.9);
        let aw_col = mix(vec3<f32>(0.5, 0.08, 0.05), vec3<f32>(0.05, 0.25, 0.3), step(0.5, u2f(pcg(wh))));
        sf.albedo = mix(mix(sf.albedo, aw_col, awning), vec3<f32>(0.03, 0.035, 0.04), shop);
        sf.rough = mix(sf.rough, 0.08, shop);
        sf.metal = mix(sf.metal, 0.6, shop);
        sf.emissive = mix(sf.emissive, warm * (0.25 + night * 3.5) * mix(0.5, 1.2, u2f(wh)), shop);
    }
    return sf;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    var n = normalize(in.normal);
    let mat = (in.style >> 8u) & 0xffu;
    let seed = in.style >> 16u;
    let night = g.bounce_color.w;
    let up = up_at(p);
    let v = normalize(g.cam_pos.xyz - p);
    let sun = light_at(p);
    let surf = world_to_surf(p);
    let shadow = shadow_at(p, n) * cloud_shadow(surf.x, surf.y);
    let sun_rad = g.sun_color.rgb * sun.w * shadow;

    var sf: Surface;
    if (mat == M_CONIFER || mat == M_BROADLEAF) {
        let h1 = pcg(seed);
        let rs = u2f(h1);
        let rs2 = u2f(pcg(h1));
        let rs3 = u2f(pcg(h1 ^ 0x9e3779b9u));
        let conifer = mat == M_CONIFER;
        let birch = !conifer && rs3 < 0.14;
        let h01 = clamp(in.obj.y / in.size.y, 0.0, 1.0);
        var ao: f32;
        var trans = 0.0;
        var sheen = 0.0;
        if (in.part > 0.75) {
            // bark: vertical fissures, birch = white with dark lenticels
            let q = in.obj;
            let ang = atan2(in.obj_n.z, in.obj_n.x);
            let fiss = value_noise(vec2<f32>(ang * 3.0, q.y * 1.4)) ;
            if (birch) {
                let mark = step(0.78, value_noise(vec2<f32>(ang * 2.0, q.y * 6.0)));
                sf.albedo = mix(vec3<f32>(0.62, 0.6, 0.55), vec3<f32>(0.05), mark);
            } else {
                sf.albedo = mix(vec3<f32>(0.07, 0.05, 0.035), vec3<f32>(0.2, 0.15, 0.1), fiss);
            }
            ao = mix(0.45, 0.9, smoothstep(0.0, 0.5, h01));
        } else {
            // foliage
            let q = in.obj * select(1.1, 2.2, conifer);
            let clump = value_noise(q.xy * 0.9 + q.zz * 0.7) * 0.6 + value_noise(q.zy * 2.3 + 7.0) * 0.4;
            let fine = value_noise(q.xz * 5.0 + q.yy * 3.0);
            // bumpy leaf clusters: perturb the crown normal
            let dn = vec3<f32>(value_noise(q.yz * 1.7) - 0.5, value_noise(q.xz * 1.7 + 3.1) - 0.5, value_noise(q.xy * 1.7 + 5.3) - 0.5);
            n = normalize(n + dn * 1.1);
            var base: vec3<f32>;
            if (conifer) {
                base = mix(vec3<f32>(0.018, 0.05, 0.03), vec3<f32>(0.04, 0.085, 0.045), rs);
                base = mix(base, vec3<f32>(0.03, 0.06, 0.07), step(0.8, rs2));          // blue spruce
            } else {
                base = mix(vec3<f32>(0.035, 0.09, 0.018), vec3<f32>(0.085, 0.15, 0.03), rs);
                if (birch) { base = vec3<f32>(0.11, 0.18, 0.04); }
                if (rs2 < 0.07) { base = mix(vec3<f32>(0.32, 0.17, 0.02), vec3<f32>(0.4, 0.08, 0.02), rs); } // autumn
            }
            let tips = smoothstep(0.3, 0.5, in.part);                          // young growth on the outside
            base *= mix(vec3<f32>(1.0), vec3<f32>(1.25, 1.3, 0.9), tips * 0.4);
            sf.albedo = base * mix(0.55, 1.25, clump) * mix(0.75, 1.1, fine);
            let gaps = smoothstep(0.25, 0.55, fine * 0.6 + clump * 0.4);       // fake depth between leaf clumps
            ao = mix(0.18, 1.0, clamp(in.part * 2.0, 0.0, 1.0)) * mix(0.55, 1.0, gaps) * mix(0.6, 1.0, h01);
            trans = select(0.9, 0.45, conifer);
            sheen = select(0.12, 0.05, conifer);
        }
        let ndl = dot(n, sun.xyz);
        let diff = clamp((ndl + 0.5) / 1.5, 0.0, 1.0);
        let back = pow(clamp(dot(v, -sun.xyz), 0.0, 1.0), 4.0) * trans * smoothstep(-0.2, 0.4, -ndl + 0.3);
        let hv = normalize(sun.xyz + v);
        let spec = pow(max(dot(n, hv), 0.0), 24.0) * sheen;
        let tr_col = sf.albedo * vec3<f32>(1.4, 1.7, 0.6);
        var col = (sf.albedo * (sun_rad * diff + ambient(n, up) + orb_light(p, n))
                + sun_rad * (tr_col * back + vec3<f32>(spec))) * ao;
        return vec4<f32>(atmosphere(col, p), 1.0);
    }

    sf = facade(in.obj, in.obj_n, in.size, mat, seed, night);
    // Simple PBR-ish: diffuse + Schlick specular + environment reflection.
    let ndv = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), sf.albedo, sf.metal);
    let fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
    let r = reflect(-v, n);
    let env_k = clamp(dot(r, up), -1.0, 1.0) * 0.5 + 0.5;
    var env = mix(g.bounce_color.rgb * 1.3, g.sky_color.rgb * 1.8 + g.bounce_color.rgb * 0.3, env_k);
    env += g.sun_color.rgb * sun.w * pow(max(dot(r, sun.xyz), 0.0), mix(2000.0, 20.0, sf.rough)) * mix(40.0, 1.0, sf.rough) * shadow;
    let diffuse = sf.albedo * (1.0 - sf.metal) * (sun_rad * max(dot(n, sun.xyz), 0.0) + ambient(n, up) + orb_light(p, n));
    let refl = env * fres * mix(1.0, 0.15, sf.rough);
    // contact darkening at the foot of buildings
    let foot = select(1.0, mix(0.45, 1.0, smoothstep(0.0, 6.0, in.obj.y)), in.obj_n.y < 0.5 && mat <= 4u);
    var col = (diffuse + refl) * foot + sf.emissive;
    return vec4<f32>(atmosphere(col, p), 1.0);
}
