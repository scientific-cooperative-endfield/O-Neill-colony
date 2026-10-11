// ============================================================================
//  field.wgsl — grass dynamics (compute).
//
//  A 512² grid of damped springs (25 cm cells) that slides with the camera
//  over the valley surface. It is stored toroidally: a world cell (cu, cv) lives
//  in texel (cu mod N, cv mod N), so moving the window costs nothing — cells
//  that just entered the window are detected and reset.
//
//  state   : rgba16f (bend.u, bend.v, vel.u, vel.v)
//  trample : rgba16f (flatten, dir.u, dir.v, -)
// ============================================================================

@group(1) @binding(0) var prev_state: texture_2d<f32>;
@group(1) @binding(1) var prev_trample: texture_2d<f32>;
@group(1) @binding(2) var out_state: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var out_trample: texture_storage_2d<rgba16float, write>;

const STIFFNESS: f32 = 38.0;
const DAMPING: f32 = 3.0;
const COUPLING: f32 = 9.0;
const RECOVERY: f32 = 0.065;

fn pmod(a: i32, n: i32) -> i32 { return ((a % n) + n) % n; }

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(g.field_params.y);
    if (i32(id.x) >= n || i32(id.y) >= n) { return; }
    let c = vec2<i32>(id.xy);
    let cs = g.field_params.x;

    // Which world cell does this texel hold now, and did it hold it last frame?
    let base = g.field_base.xy;
    let prev = g.field_base.zw;
    let cell = base + vec2<i32>(pmod(c.x - base.x, n), pmod(c.y - base.y, n));
    let fresh = any(cell < prev) || any(cell >= prev + vec2<i32>(n));
    let p = (vec2<f32>(cell) + 0.5) * cs;
    let dt = min(g.space.w, 1.0 / 20.0);   // space.w carries the simulation dt

    var bend = vec2<f32>(0.0);
    var vel = vec2<f32>(0.0);
    var tr = vec4<f32>(0.0);
    if (!fresh) {
        let s = textureLoad(prev_state, c, 0);
        bend = s.xy;
        vel = s.zw;
        let v0 = textureLoad(prev_state, vec2<i32>(pmod(c.x + 1, n), c.y), 0).zw;
        let v1 = textureLoad(prev_state, vec2<i32>(pmod(c.x - 1, n), c.y), 0).zw;
        let v2 = textureLoad(prev_state, vec2<i32>(c.x, pmod(c.y + 1, n)), 0).zw;
        let v3 = textureLoad(prev_state, vec2<i32>(c.x, pmod(c.y - 1, n)), 0).zw;
        vel = mix(vel, (v0 + v1 + v2 + v3) * 0.25, min(COUPLING * dt, 0.5));
        tr = textureLoad(prev_trample, c, 0);
    }

    // Imprints of balls and the player's feet ---------------------------------
    tr.x = max(tr.x - dt * RECOVERY * (0.4 + tr.x), 0.0);
    var push = vec2<f32>(0.0);
    let count = i32(g.field_params.z);
    for (var i = 0; i < count; i++) {
        let b = g.balls[i];
        let bv = g.ball_vel[i];
        let to = p - b.xy;
        let d = length(to);
        let reach = b.w * 1.5;
        if (d > reach) { continue; }
        let contact = 1.0 - smoothstep(0.0, 0.4, b.z);
        if (contact <= 0.0) { continue; }
        let out_dir = select(vec2<f32>(1.0, 0.0), to / d, d > 1e-4);
        let spd = length(bv.xy);
        let rf = b.w * 0.85;
        if (d < rf) {
            let k = d / rf;
            let f = contact * bv.z * sqrt(1.0 - k * k);
            if (f > tr.x) {
                let roll = select(out_dir, bv.xy / max(spd, 1e-4), spd > 0.25);
                let m = mix(out_dir, roll, 0.7);
                tr = vec4<f32>(f, m / max(length(m), 1e-4), 0.0);
            }
        }
        let ring = contact * (1.0 - smoothstep(rf * 0.5, reach, d));
        push += out_dir * ring * (0.5 + min(spd, 8.0) * 0.12);
    }

    // Spring integration --------------------------------------------------------
    let goal = wind_goal(p, g.cam_pos.w) + push;
    let steps = clamp(u32(ceil(dt * 120.0)), 1u, 6u);
    let h = dt / f32(steps);
    for (var i = 0u; i < steps; i++) {
        let k = STIFFNESS + 140.0 * max(length(bend) - 0.75, 0.0);
        vel += (k * (goal - bend) - DAMPING * vel) * h;
        bend += vel * h;
    }
    let bl = length(bend);
    if (bl > 1.1) { bend *= 1.1 / bl; }

    textureStore(out_state, c, vec4<f32>(bend, vel));
    textureStore(out_trample, c, tr);
}
