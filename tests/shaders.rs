//! Validates every WGSL module with naga (no GPU needed): `cargo test`.

#[path = "../src/shaders.rs"]
mod shaders;

fn validate(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: parse error:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    if let Err(e) = v.validate(&module) {
        panic!("{name}: validation error:\n{}", e.emit_to_string(src));
    }
}

#[test]
fn scene_shaders_validate() {
    for (name, body) in shaders::SCENE {
        validate(name, &shaders::compose(body));
    }
}

#[test]
fn post_shader_validates() {
    validate("post", shaders::POST);
}
