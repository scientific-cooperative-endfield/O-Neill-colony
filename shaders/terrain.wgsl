// ============================================================================
//  terrain.wgsl — valley floor, rendered as instanced chunks with 4 LODs.
//  Heights come from the shared heightmap; below-water texels are clamped to
//  the water level and shaded as rivers / lakes.
// ============================================================================

struct VsIn {
    @location(0) grid: vec3<f32>,      // x, y in [0,1], z = skirt flag
    @location(1) chunk: vec4<f32>,     // instance: u0, v0, du, dv
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) surf: vec3<f32>,      // u, v, water depth
    @location(2) normal: vec3<f32>,
    @location(3) fid: f32,
    @location(4) @interpolate(flat) fid_flat: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let u = in.chunk.x + in.grid.x * in.chunk.z;
    let v = clamp(in.chunk.y + in.grid.y * in.chunk.w, 0.0, colony_len());
    let h = sample_height(u, v);
    let hr = max(h, 0.0) - in.grid.z * 4.0;
    let th = u / colony_r();
    var o: VsOut;
    o.world_pos = surf_to_world(u, v, hr);
    o.surf = vec3<f32>(u, v, -h);
    var nl = surf_normal_local(u, v, 2.0);
    if (h < 0.0) { nl = vec3<f32>(0.0, 1.0, 0.0); }
    o.normal = local_to_world(nl, th);
    o.clip = g.view_proj * vec4<f32>(bh_xf(o.world_pos), 1.0);
    o.fid = bh_fid(o.world_pos);
    o.fid_flat = o.fid;
    return o;
}

fn water_normal(uv: vec2<f32>, t: f32) -> vec3<f32> {
    var d = vec2<f32>(0.0);
    var dirs = array<vec2<f32>, 4>(vec2<f32>(0.8, 0.6), vec2<f32>(-0.4, 0.9), vec2<f32>(0.95, -0.3), vec2<f32>(-0.7, -0.7));
    var freq = array<f32, 4>(0.35, 0.6, 1.1, 1.9);
    var amp = array<f32, 4>(0.07, 0.05, 0.03, 0.02);
    for (var i = 0; i < 4; i++) {
        let ph = dot(uv, dirs[i]) * freq[i] + t * (1.2 + f32(i) * 0.7);
        d += dirs[i] * cos(ph) * freq[i] * amp[i];
    }
    let n2 = value_noise(uv * 0.8 + vec2<f32>(t * 0.3, 0.0)) - 0.5;
    return normalize(vec3<f32>(-d.x - n2 * 0.05, 1.0, -d.y + n2 * 0.05));
}

// Rough radiance seen in a reflection: haze & far valleys above, sun image.
fn env_radiance(r: vec3<f32>, up: vec3<f32>, l: vec4<f32>, uv: vec2<f32>) -> vec3<f32> {
    let k = dot(r, up);
    // near-horizontal: dark shoreline / trees, higher: the land curving up
    // overhead, washed in haze, with the long windows between valleys.
    let shore = g.bounce_color.rgb * vec3<f32>(0.55, 0.85, 0.5) * (0.7 + 0.6 * value_noise(uv * 0.05));
    let haze = g.sky_color.rgb * 1.5 + g.bounce_color.rgb * 0.5;
    var c = mix(shore, haze, smoothstep(0.02, 0.35, k));
    c += g.sun_color.rgb * l.w * pow(max(dot(r, l.xyz), 0.0), 900.0) * 120.0;
    c += g.sun_color.rgb * l.w * pow(max(dot(r, l.xyz), 0.0), 40.0) * 0.6;
    return c;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (bh_seam(in.fid, in.fid_flat)) { discard; }
    let p = in.world_pos;
    let u = in.surf.x;
    let v = in.surf.y;
    let depth = in.surf.z;
    let th = fract(u / circumference()) * TAU;
    let st = strip_of(th);
    let s = st.y;
    let up = up_at(p);
    let n = normalize(in.normal);
    let dist = distance(p, g.cam_pos.xyz);
    let night = g.bounce_color.w;
    let sun = light_at(p);
    let vdir = normalize(g.cam_pos.xyz - p);
    let shadow = shadow_at(p, n) * cloud_shadow(u, v);
    let sun_rad = g.sun_color.rgb * sun.w * shadow;

    // ---- water ------------------------------------------------------------------
    if (depth > 0.0) {
        // fade small waves with distance to avoid shimmering bands
        let wl = normalize(mix(water_normal(vec2<f32>(u, v), g.cam_pos.w), vec3<f32>(0.0, 1.0, 0.0), smoothstep(12.0, 110.0, dist) * 0.92));
        let wn = local_to_world(wl, th);
        let cosv = max(dot(wn, vdir), 0.0);
        let fres = 0.02 + 0.98 * pow(1.0 - cosv, 5.0);
        let r = reflect(-vdir, wn);
        let absorb = 1.0 - exp(-depth * 0.6);
        let body = mix(vec3<f32>(0.05, 0.16, 0.13), vec3<f32>(0.004, 0.025, 0.035), absorb);
        // light scattered in the water column, brighter on wave crests
        let crest = smoothstep(0.985, 1.0, dot(wl, normalize(vec3<f32>(0.3, 1.0, 0.2))));
        var col = body * (sun_rad * max(dot(up, sun.xyz), 0.0) * (0.5 + crest * 0.6) + ambient(up, up));
        col = mix(col, env_radiance(r, up, sun, vec2<f32>(u, v)) * mix(0.35, 1.0, shadow), fres);
        let foam = (1.0 - smoothstep(0.0, 0.25, depth)) * smoothstep(0.45, 0.7, value_noise(vec2<f32>(u, v) * 1.5 + g.cam_pos.w * 0.4));
        col = mix(col, (sun_rad + ambient(up, up)) * 0.8, foam * 0.6);
        return vec4<f32>(atmosphere(col, p), 1.0);
    }

    // ---- land -------------------------------------------------------------------
    let zone = sample_zone(u, v);
    let detail = value_noise(vec2<f32>(u, v) * 0.9) * 0.6 + value_noise(vec2<f32>(u, v) * 7.0) * 0.4;
    let far_k = smoothstep(g.grass.y * 0.3, g.grass.y * 0.95, dist);

    // Meadow: soil between blades near the camera, canopy colour further away.
    let soil = mix(vec3<f32>(0.045, 0.035, 0.022), vec3<f32>(0.075, 0.06, 0.035), detail);
    let canopy = mix(vec3<f32>(0.07, 0.15, 0.025), vec3<f32>(0.12, 0.20, 0.04), detail);
    var albedo = mix(mix(soil, vec3<f32>(0.04, 0.06, 0.015), zone.x), canopy, far_k);
    var emissive = vec3<f32>(0.0);
    var ao = mix(mix(0.4, 0.8, 1.0 - zone.x), 1.0, far_k);
    var spec = 0.0;

    // Forest canopy colour (real trees are drawn near the camera).
    let forest_col = mix(vec3<f32>(0.012, 0.035, 0.014), vec3<f32>(0.035, 0.075, 0.022), value_noise(vec2<f32>(u, v) * 0.35) * 0.6 + value_noise(vec2<f32>(u, v) * 0.08) * 0.4);
    albedo = mix(albedo, forest_col, zone.w * smoothstep(g.params2.w * 0.6, g.params2.w, dist) * 0.85);

    // Farm patchwork with crop rows.
    if (zone.y > 0.5) {
        let fc = farm_cell(s, v, st.x);
        var crop = crop_albedo(fc.x, fc.z);
        let rows = 0.5 + 0.5 * sin(s / (select(0.42, 1.3, fc.x > 1.5 && fc.x < 2.5)) * TAU);
        crop *= mix(1.0, mix(0.6, 1.0, rows), 1.0 - smoothstep(20.0, 120.0, dist));
        if (fc.y < 1.8) { crop = vec3<f32>(0.11, 0.08, 0.05); }
        let near_soil = mix(soil, crop * 0.4, 0.5);
        albedo = mix(near_soil, crop, max(far_k, select(0.0, 1.0, fc.x > 3.5 && fc.x < 4.5)));
        ao = mix(0.55, 1.0, far_k);
    }

    // City: pavement, roads with markings and night lighting.
    if (zone.z > 0.35) {
        let rd = road_dist(s, v);
        let paving = mix(vec3<f32>(0.18, 0.18, 0.19), vec3<f32>(0.26, 0.26, 0.27), detail);
        let lawn = select(paving, albedo, zone.x > 0.6);
        var city = lawn;
        // No asphalt on shores / under water (canal bridges are separate objects).
        let dry = smoothstep(0.35, 0.9, -depth);
        // Maglev pillars stand on the road centre line: draw a planted median there.
        let median = abs(abs(s) - 0.3 * g.colony.x) < 2.4;
        if (rd < ROAD_HALF + 0.4 && dry > 0.5 && median && rd < 2.4) {
            city = mix(vec3<f32>(0.2, 0.2, 0.19), vec3<f32>(0.05, 0.09, 0.03), step(0.5, rd) * step(rd, 2.0));
        } else if (rd < ROAD_HALF + 0.4 && dry > 0.5) {
            city = vec3<f32>(0.035, 0.037, 0.04) * mix(0.8, 1.2, detail);
            spec = 0.3;
            // centre line & lane dashes
            let along = select(s, v, abs(fract(s / BLOCK + 0.5) - 0.5) * BLOCK < ROAD_HALF + 0.4);
            if (rd < 0.15) { city = vec3<f32>(0.4, 0.38, 0.3); }
            if (abs(rd - 3.5) < 0.1 && fract(along / 6.0) < 0.5) { city = vec3<f32>(0.5); }
            // futuristic glowing guide line + street lights at night
            emissive += vec3<f32>(0.1, 0.8, 1.0) * (1.0 - smoothstep(0.04, 0.12, rd)) * (0.3 + night * 2.5);
            emissive += vec3<f32>(1.0, 0.55, 0.2) * night * 3.0 * (1.0 - smoothstep(0.0, 0.25, abs(rd - (ROAD_HALF - 0.3)))) * step(0.8, fract(along / 25.0));
        } else if (rd < ROAD_HALF + 2.5 && dry > 0.5) {
            city = vec3<f32>(0.32, 0.31, 0.3);   // kerb / sidewalk
        }
        albedo = mix(albedo, city, smoothstep(0.35, 0.55, zone.z) * mix(0.0, 1.0, dry));
        ao = mix(ao, 1.0, smoothstep(0.35, 0.55, zone.z));
    }

    // Sand on shores, rock on steep slopes.
    let sand = mix(vec3<f32>(0.32, 0.27, 0.18), vec3<f32>(0.42, 0.36, 0.24), detail);
    let slope = 1.0 - dot(n, up);
    albedo = mix(albedo, sand, (1.0 - smoothstep(0.15, 0.9, -depth)) * (1.0 - zone.z));
    let rock = mix(vec3<f32>(0.12, 0.11, 0.10), vec3<f32>(0.22, 0.2, 0.18), detail);
    albedo = mix(albedo, rock, smoothstep(0.3, 0.5, slope));

    // Promenade along the windows, with a ribbon of lights.
    let edge = half_land_width() - abs(s);
    if (edge < 14.0) {
        let tile = step(0.92, fract(v / 3.0)) + step(0.92, fract(edge / 3.0));
        albedo = mix(vec3<f32>(0.16, 0.16, 0.15), vec3<f32>(0.24, 0.23, 0.21), detail) * (1.0 - 0.35 * min(tile, 1.0));
        ao = 1.0;
        let lamp = (1.0 - smoothstep(0.0, 0.3, abs(edge - 2.0))) * step(0.7, fract(v / 12.0));
        emissive += vec3<f32>(0.9, 0.95, 1.0) * lamp * (0.2 + night * 5.0);
    }

    let hv = normalize(sun.xyz + vdir);
    var col = albedo * (sun_rad * max(dot(n, sun.xyz), 0.0) + ambient(n, up) + orb_light(p, n)) * ao
            + sun_rad * spec * pow(max(dot(n, hv), 0.0), 60.0)
            + emissive;
    return vec4<f32>(atmosphere(col, p), 1.0);
}

@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    let u = in.chunk.x + in.grid.x * in.chunk.z;
    let v = clamp(in.chunk.y + in.grid.y * in.chunk.w, 0.0, colony_len());
    let h = max(sample_height(u, v), 0.0) - in.grid.z * 4.0;
    return g.shadow_mat * vec4<f32>(bh_xf(surf_to_world(u, v, h)), 1.0);
}
