//! WGSL sources. Scene modules get `common.wgsl` prepended (shared structs,
//! group-0 bindings and helpers); the post-processing module is standalone.

pub const COMMON: &str = include_str!("../shaders/common.wgsl");
pub const FIELD: &str = include_str!("../shaders/field.wgsl");
pub const GRASS_CULL: &str = include_str!("../shaders/grass_cull.wgsl");
pub const GRASS: &str = include_str!("../shaders/grass.wgsl");
pub const TERRAIN: &str = include_str!("../shaders/terrain.wgsl");
pub const SHELL: &str = include_str!("../shaders/shell.wgsl");
pub const OBJECTS_CULL: &str = include_str!("../shaders/objects_cull.wgsl");
pub const OBJECTS: &str = include_str!("../shaders/objects.wgsl");
pub const BALLS: &str = include_str!("../shaders/balls.wgsl");
pub const AIRCARS: &str = include_str!("../shaders/aircars.wgsl");
pub const PROPS: &str = include_str!("../shaders/props.wgsl");
pub const PARTICLES: &str = include_str!("../shaders/particles.wgsl");
pub const LINES: &str = include_str!("../shaders/lines.wgsl");
pub const POST: &str = include_str!("../shaders/post.wgsl");

#[allow(dead_code)]
pub const SCENE: [(&str, &str); 12] = [
    ("props", PROPS),
    ("particles", PARTICLES),
    ("field", FIELD),
    ("grass_cull", GRASS_CULL),
    ("grass", GRASS),
    ("terrain", TERRAIN),
    ("shell", SHELL),
    ("objects_cull", OBJECTS_CULL),
    ("objects", OBJECTS),
    ("balls", BALLS),
    ("aircars", AIRCARS),
    ("lines", LINES),
];

pub fn compose(body: &str) -> String {
    let mut s = String::with_capacity(COMMON.len() + body.len() + 1);
    s.push_str(COMMON);
    s.push('\n');
    s.push_str(body);
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn scene_shaders_validate() {
        for (name, body) in super::SCENE {
            let src = super::compose(body);
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: parse error:\n{}", e.emit_to_string(&src)));
            let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
            v.validate(&module).unwrap_or_else(|e| panic!("{name}: validation error:\n{}", e.emit_to_string(&src)));
        }
    }
}
