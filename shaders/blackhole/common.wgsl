// Shared declarations of the black-hole passes. Geometric units, M = 1:
// the horizon is r = 2, the photon sphere r = 3, the ISCO r = 6.
// Black-hole frame: the accretion disk lies in the z = 0 plane and rotates
// counter-clockwise around +z.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const LUT_TMIN: f32 = 500.0;
const LUT_TMAX: f32 = 60000.0;

struct Params {
    cam_pos: vec4<f32>,    // xyz camera position, w = tan(fov_y / 2)
    cam_right: vec4<f32>,  // xyz, w = aspect (w / h)
    cam_up: vec4<f32>,     // xyz, w = disk texture time (M)
    cam_fwd: vec4<f32>,    // xyz, w = Lorentz factor of the camera relative to the rain frame
    cam_beta: vec4<f32>,   // xyz camera velocity relative to the local rain frame (c = 1), w = r_escape
    disk: vec4<f32>,       // r_in, r_out, T_peak (K), brightness
    disk2: vec4<f32>,      // opacity, halo strength, halo scale height (H / r), sky intensity
    beacon: vec4<f32>,     // station marker position xyz, w = radius (0 = off)
    beacon_u: vec4<f32>,   // marker 4-velocity in PG coordinates: xyz = u^i, w = u^T
    beacon_col: vec4<f32>, // rgb = rest-frame emission, w = rest-frame colour temperature (K)
    size: vec4<u32>,       // trace width, trace height, max steps, samples per pixel
    flags: vec4<u32>,      // x = debug bits (1 zones, 2 step heat-map, 4 halo), y = frame, z/w = output size
    misc: vec4<f32>,       // x = step scale, y = sky texel angle, z = sky mip count, w = tidal radius (debug zones)
    hull: vec4<f32>,       // x = snapshot count (0 = no hull), y = near-field range (M), z = union radius (M), w = light delay on
    hull_light: vec4<f32>, // rgb = illumination of the hull by the disk, w = interior glow (air pressure fraction)
    heat: vec4<f32>,       // debug heat map range: x = min, y = max of dτ/dt; z = max |clock offset| (s)
    kerr: vec4<f32>,       // x = spin a, y = ISCO, z = horizon r+, w = Kerr tracer on
}

// Station snapshot header (newest first) and hull panel (see sim.rs NodeGpu).
struct Snap {
    c: vec4<f32>,          // centre − camera (M), w = T_snapshot − T_camera (M)
    u: vec4<f32>,          // centre 4-velocity: xyz = u^i, w = u^T
    info: vec4<f32>,       // x = bounding radius (M), y = slot, z = node count
}

struct Node {
    p: vec4<f32>,          // offset from the snapshot centre (M), w = kind (0 land, 1 window, 2 cap)
    q: vec4<f32>,          // orientation quaternion (panel → BH frame)
    w: vec4<f32>,          // coordinate velocity dx/dT, w = dτ/dT
    h: vec4<f32>,          // half extents (M), w = dτ/dt
    e: vec4<f32>,          // clock offset (s), stress, fragment id, alive
}

fn bb_uv(t: f32) -> vec2<f32> {
    let x = clamp((log2(max(t, 1.0)) - log2(LUT_TMIN)) / (log2(LUT_TMAX) - log2(LUT_TMIN)), 0.0, 1.0);
    return vec2<f32>(x * (255.0 / 256.0) + 0.5 / 256.0, 0.5);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
