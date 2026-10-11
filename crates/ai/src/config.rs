//! Bot skill levels (RON, `assets/difficulty.ron`).

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Difficulty {
    pub name: String,
    /// delay between first sighting and opening fire (s)
    pub reaction_s: f64,
    /// hand tremor / misjudgement, 1σ (mrad)
    pub aim_error_mrad: f64,
    /// maximum view turn rate (rad/s)
    pub turn_rate: f64,
    /// fraction of the Coriolis / spin-gravity terms in the bot's ballistic model
    pub coriolis_model: f64,
    pub gravity_model: f64,
    pub drag_model: bool,
    /// fraction of the target velocity used for leading
    pub lead: f64,
    /// fire when the muzzle is within this angle of the solution (mrad)
    pub fire_tolerance_mrad: f64,
    pub vision_range: f64,
    pub fov_deg: f64,
    /// quietest sound noticed (dB SPL at the ear)
    pub hearing_db: f64,
    /// rounds per automatic burst before letting the recoil settle
    pub burst: u32,
    /// willingness to use the curvature device (0…1)
    pub device_use: f64,
    /// how early the bot seeks cover / retreats (0…1)
    pub caution: f64,
}

#[derive(Debug, Deserialize)]
struct File {
    levels: Vec<Difficulty>,
}

pub const DIFFICULTY_RON: &str = include_str!("../assets/difficulty.ron");

pub fn parse(text: &str) -> Result<Vec<Difficulty>, String> {
    let f: File = ron::from_str(text).map_err(|e| format!("difficulty.ron: {e}"))?;
    if f.levels.is_empty() {
        return Err("difficulty.ron: no levels".into());
    }
    Ok(f.levels)
}

pub fn builtin() -> Vec<Difficulty> {
    parse(DIFFICULTY_RON).expect("built-in difficulty.ron is valid")
}

pub fn by_name(levels: &[Difficulty], name: &str) -> Option<Difficulty> {
    levels.iter().find(|d| d.name == name).cloned()
}
