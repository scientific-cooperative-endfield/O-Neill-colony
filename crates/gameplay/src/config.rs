//! Data-driven parameters (RON). The defaults are compiled in from
//! `crates/gameplay/assets/*.ron`; `GameConfig::load_dir` reads overrides.

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct WeaponCfg {
    pub id: String,
    pub name: String,
    /// bullet mass (g) and calibre (mm)
    pub bullet_mass_g: f64,
    pub caliber_mm: f64,
    /// ballistic coefficient against the G7 standard projectile (lb/in², as printed on ammo boxes)
    pub bc_g7: f64,
    pub muzzle_velocity: f64,
    pub rpm: f64,
    pub automatic: bool,
    pub magazine: u32,
    pub reserve: u32,
    pub reload_s: f64,
    /// dispersion (MOA, 1σ) from the hip and aimed
    pub spread_moa_hip: f64,
    pub spread_moa_ads: f64,
    /// recoil impulse → muzzle climb per shot (mrad), recovers at `recoil_recovery` (1/s)
    pub recoil_mrad: f64,
    pub recoil_recovery: f64,
    /// damage at muzzle energy, scaled by √(E_impact / E_muzzle)
    pub damage: f64,
    pub head_mult: f64,
    /// sound level at 1 m (dB SPL)
    pub loudness_db: f64,
    /// projectile life (s)
    pub max_flight_s: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WeaponSet {
    pub weapons: Vec<WeaponCfg>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct JetpackCfg {
    /// thrust (N) on each body axis, specific impulse (s), propellant (kg)
    pub thrust_n: f64,
    pub isp_s: f64,
    pub propellant_kg: f64,
    /// attitude rate limit (rad/s) for 6-DOF flight
    pub max_turn_rate: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PlayerCfg {
    pub mass_kg: f64,
    pub radius: f64,
    pub height: f64,
    pub crouch_height: f64,
    /// eye below the top of the capsule (m)
    pub eye_below_top: f64,
    pub walk_speed: f64,
    pub run_speed: f64,
    pub crouch_speed: f64,
    /// shoe–floor friction coefficient: the horizontal force is limited by μ·N,
    /// so in low gravity you cannot accelerate (or stop) quickly
    pub friction: f64,
    /// take-off speed of a jump (legs give about the same Δv in any gravity)
    pub jump_speed: f64,
    /// thrustless air control (m/s²): body twisting / arm swing, tiny
    pub air_control: f64,
    /// below this felt gravity (m/s²) the player floats instead of walking
    pub walk_min_g: f64,
    /// maximum walkable slope (degrees)
    pub max_slope_deg: f64,
    pub step_height: f64,
    /// highest obstacle (m above the feet) the player can vault onto with JUMP
    pub mantle_height: f64,
    pub max_hp: f64,
    pub suit_o2_s: f64,
    /// magnetic boots: max holding force on steel hull plating (N)
    pub mag_boot_n: f64,
    pub jetpack: JetpackCfg,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DamageCfg {
    /// landing / impact speed (m/s) with no damage and the speed that kills
    pub impact_safe: f64,
    pub impact_lethal: f64,
    /// pressures (kPa): hypoxia below `hypoxia_kpa`, ebullism below the Armstrong limit
    pub hypoxia_kpa: f64,
    pub armstrong_kpa: f64,
    pub hypoxia_dps: f64,
    pub ebullism_dps: f64,
    /// barotrauma: pressure drop faster than this (kPa/s) hurts, dps per kPa/s above it
    pub barotrauma_kpa_s: f64,
    pub barotrauma_dps_per_kpa_s: f64,
    /// tidal stretch across the body (m/s² head-to-foot) tolerated, dps per m/s² above it
    pub tidal_safe: f64,
    pub tidal_dps_per: f64,
    /// absorbed dose (Gy) → hp
    pub hp_per_gray: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BubbleCfg {
    pub radius: f64,
    /// height of the metric barrier per unit mass (J/kg): a projectile with
    /// ½v_r² below it is reflected, above it passes with v_r² reduced by 2U
    pub barrier_j_per_kg: f64,
    /// dτ/dt inside relative to the station clock
    pub time_rate: f64,
    pub duration_s: f64,
    pub cost_j: f64,
    pub instability: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MicroHoleCfg {
    pub mass_kg: f64,
    pub launch_speed: f64,
    /// containment absorbs the Hawking luminosity; the heat sink has this capacity (J)
    pub containment_capacity_j: f64,
    /// the device recalls (re-captures) the hole after this time
    pub duration_s: f64,
    pub cost_j: f64,
    pub instability: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FoldCfg {
    pub mouth_radius: f64,
    pub max_range: f64,
    pub duration_s: f64,
    /// tidal acceleration at the throat for a stable fold (m/s² per metre)
    pub throat_tidal: f64,
    pub cost_j: f64,
    pub instability: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DeviceCfg {
    pub capacity_j: f64,
    pub recharge_w: f64,
    /// accumulated instability decays at this rate (1/s)
    pub instability_decay: f64,
    /// probability of failure on activation = 1 − exp(−instability / failure_scale)
    pub failure_scale: f64,
    pub cooldown_s: f64,
    pub bubble: BubbleCfg,
    pub micro_hole: MicroHoleCfg,
    pub fold: FoldCfg,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RulesCfg {
    pub time_limit_s: f64,
    pub frag_limit: u32,
    pub respawn_s: f64,
    pub spawn_protection_s: f64,
}

#[derive(Clone, Debug)]
pub struct GameConfig {
    pub weapons: Vec<WeaponCfg>,
    pub player: PlayerCfg,
    pub damage: DamageCfg,
    pub devices: DeviceCfg,
    pub rules: RulesCfg,
}

#[derive(Debug, Deserialize)]
struct PlayerFile {
    player: PlayerCfg,
    damage: DamageCfg,
    rules: RulesCfg,
}

pub const WEAPONS_RON: &str = include_str!("../assets/weapons.ron");
pub const PLAYER_RON: &str = include_str!("../assets/player.ron");
pub const DEVICES_RON: &str = include_str!("../assets/devices.ron");

impl GameConfig {
    pub fn from_ron(weapons: &str, player: &str, devices: &str) -> Result<GameConfig, String> {
        let w: WeaponSet = ron::from_str(weapons).map_err(|e| format!("weapons.ron: {e}"))?;
        let p: PlayerFile = ron::from_str(player).map_err(|e| format!("player.ron: {e}"))?;
        let d: DeviceCfg = ron::from_str(devices).map_err(|e| format!("devices.ron: {e}"))?;
        if w.weapons.is_empty() {
            return Err("weapons.ron: no weapons".into());
        }
        Ok(GameConfig { weapons: w.weapons, player: p.player, damage: p.damage, devices: d, rules: p.rules })
    }

    /// Compiled-in defaults.
    pub fn builtin() -> GameConfig {
        Self::from_ron(WEAPONS_RON, PLAYER_RON, DEVICES_RON).expect("built-in RON configs are valid")
    }

    /// Reads `weapons.ron`, `player.ron`, `devices.ron` from `dir`, falling back
    /// to the built-in text for missing files.
    pub fn load_dir(dir: &str) -> Result<GameConfig, String> {
        let rd = |name: &str, def: &str| std::fs::read_to_string(format!("{dir}/{name}")).unwrap_or_else(|_| def.to_string());
        Self::from_ron(&rd("weapons.ron", WEAPONS_RON), &rd("player.ron", PLAYER_RON), &rd("devices.ron", DEVICES_RON))
    }

    pub fn weapon_index(&self, id: &str) -> Option<usize> {
        self.weapons.iter().position(|w| w.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_configs_parse() {
        let c = GameConfig::builtin();
        assert_eq!(c.weapons.len(), 3);
        assert!(c.weapon_index("sniper").is_some());
        assert!(c.player.mass_kg > 40.0);
        assert!(c.devices.bubble.radius > 0.0);
    }
}
