//! Component types. All plain data, `Clone`, hashed into the state hash.

use crate::det::StateHasher;
use crate::ecs::Entity;
use crate::input::Command;
use crate::map::Material;
use glam::{DQuat, DVec3};

#[derive(Clone, Debug, PartialEq)]
pub struct Pilot {
    pub id: u8,
    pub name: String,
    pub team: u8,
    pub bot: bool,
    pub prev_buttons: u16,
    pub last_cmd: Command,
    /// neuro-interface trajectory projection on the HUD
    pub neuro: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveMode {
    /// standing / walking on a support
    Walk,
    /// no support and no thrust: ballistic (jumping, falling, drifting at the axis)
    Float,
    /// 6-DOF jetpack flight
    Jetpack,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    /// feet (bottom of the capsule), colony frame
    pub pos: DVec3,
    pub vel: DVec3,
    pub mass: f64,
    pub radius: f64,
    pub height: f64,
    pub crouched: bool,
    pub grounded: bool,
    pub ground: Option<Material>,
    pub ground_pane: Option<(i32, i32)>,
    pub mode: MoveMode,
    /// walking view: yaw / pitch relative to the local (east, up, north) frame
    pub yaw: f64,
    pub pitch: f64,
    /// 6-DOF orientation (body z = forward, y = up)
    pub orient: DQuat,
    /// distance walked since the last footstep (m)
    pub stride: f64,
    /// last tick's local "up" (for tidal axis etc.)
    pub up: DVec3,
}

impl Body {
    pub fn centre(&self) -> DVec3 {
        self.pos + self.axis() * (0.5 * self.height)
    }
    /// body long axis (feet → head)
    pub fn axis(&self) -> DVec3 {
        match self.mode {
            MoveMode::Jetpack => self.orient * DVec3::Y,
            _ => self.up,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeathCause {
    Bullet(u8),
    Impact,
    Decompression,
    Tidal,
    Radiation,
    Spaghettified,
    LostInSpace,
    Debris,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Health {
    pub hp: f64,
    pub max: f64,
    pub alive: bool,
    pub respawn_in: f64,
    pub protect: f64,
    pub last_attacker: Option<Entity>,
    pub last_attack_tick: u32,
    pub dose_gy: f64,
    pub cause: Option<DeathCause>,
    /// accumulated since last tick (damage this tick, for AI "under fire")
    pub recent_damage: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Suit {
    pub o2_s: f64,
    /// 0…1: bullet holes make the suit leak
    pub integrity: f64,
    pub sealed: bool,
    pub propellant_kg: f64,
    /// ambient pressure last tick (Pa), for barotrauma
    pub p_prev: f64,
}

/// Proper time τ of the entity: dτ/dt_station = (SR factor from the inertial
/// speed) × (gravitational/field lapse) × (black-hole clock of its panel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProperTime {
    pub tau: f64,
    pub rate: f64,
    /// time-rate multiplier for dynamics (fields: bubbles), ≈ 1 normally
    pub dyn_rate: f64,
}

impl Default for ProperTime {
    fn default() -> Self {
        ProperTime { tau: 0.0, rate: 1.0, dyn_rate: 1.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeaponState {
    pub mag: u32,
    pub reserve: u32,
    pub cooldown: f64,
    pub reload: f64,
    pub shots: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arsenal {
    pub slots: Vec<WeaponState>,
    pub current: usize,
    /// current muzzle climb (rad)
    pub recoil: f64,
    pub switching: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Projectile {
    pub pos: DVec3,
    pub vel: DVec3,
    pub owner: Entity,
    pub weapon: u8,
    pub age: f64,
    /// muzzle energy (J)
    pub e0: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Debris {
    pub pos: DVec3,
    pub vel: DVec3,
    pub mass: f64,
    pub size: f64,
    pub life: f64,
    pub material: Material,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMode {
    Bubble,
    MicroHole,
    Fold,
}

impl DeviceMode {
    pub fn next(self) -> Self {
        match self {
            DeviceMode::Bubble => DeviceMode::MicroHole,
            DeviceMode::MicroHole => DeviceMode::Fold,
            DeviceMode::Fold => DeviceMode::Bubble,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            DeviceMode::Bubble => "Пузырь-барьер",
            DeviceMode::MicroHole => "Микро-ЧД",
            DeviceMode::Fold => "Складка (кротовая нора)",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub mode: DeviceMode,
    pub charge_j: f64,
    pub instability: f64,
    pub cooldown: f64,
    pub active: Option<Entity>,
    pub uses: u32,
    pub failures: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub kills: u32,
    pub deaths: u32,
    pub suicides: u32,
    pub shots: u32,
    pub hits: u32,
    pub headshots: u32,
    pub damage_dealt: f64,
    pub device_uses: u32,
    pub device_failures: u32,
    pub env_deaths: u32,
}

// ---------------------------------------------------------------- hashing --

pub trait HashState {
    fn hash(&self, h: &mut StateHasher);
}

impl HashState for Body {
    fn hash(&self, h: &mut StateHasher) {
        h.v3(self.pos);
        h.v3(self.vel);
        h.f64(self.mass);
        h.f64(self.height);
        h.bool(self.grounded);
        h.u64(self.mode as u64);
        h.f64(self.yaw);
        h.f64(self.pitch);
        h.quat(self.orient);
        h.f64(self.stride);
    }
}
impl HashState for Health {
    fn hash(&self, h: &mut StateHasher) {
        h.f64(self.hp);
        h.bool(self.alive);
        h.f64(self.respawn_in);
        h.f64(self.dose_gy);
        h.u64(self.last_attacker.map_or(u64::MAX, |e| e.bits()));
    }
}
impl HashState for Suit {
    fn hash(&self, h: &mut StateHasher) {
        h.f64(self.o2_s);
        h.f64(self.integrity);
        h.bool(self.sealed);
        h.f64(self.propellant_kg);
        h.f64(self.p_prev);
    }
}
impl HashState for ProperTime {
    fn hash(&self, h: &mut StateHasher) {
        h.f64(self.tau);
        h.f64(self.rate);
    }
}
impl HashState for Arsenal {
    fn hash(&self, h: &mut StateHasher) {
        for s in &self.slots {
            h.u64(s.mag as u64 | (s.reserve as u64) << 32);
            h.f64(s.cooldown);
            h.f64(s.reload);
            h.u64(s.shots as u64);
        }
        h.u64(self.current as u64);
        h.f64(self.recoil);
    }
}
impl HashState for Projectile {
    fn hash(&self, h: &mut StateHasher) {
        h.v3(self.pos);
        h.v3(self.vel);
        h.u64(self.owner.bits());
        h.f64(self.age);
    }
}
impl HashState for Debris {
    fn hash(&self, h: &mut StateHasher) {
        h.v3(self.pos);
        h.v3(self.vel);
        h.f64(self.life);
    }
}
impl HashState for Device {
    fn hash(&self, h: &mut StateHasher) {
        h.u64(self.mode as u64);
        h.f64(self.charge_j);
        h.f64(self.instability);
        h.f64(self.cooldown);
        h.u64(self.uses as u64 | (self.failures as u64) << 32);
    }
}
impl HashState for Stats {
    fn hash(&self, h: &mut StateHasher) {
        h.u64(self.kills as u64 | (self.deaths as u64) << 32);
        h.u64(self.shots as u64 | (self.hits as u64) << 32);
        h.f64(self.damage_dealt);
    }
}
impl HashState for Pilot {
    fn hash(&self, h: &mut StateHasher) {
        h.u64(self.id as u64);
        h.u64(self.prev_buttons as u64);
        h.bool(self.neuro);
    }
}
impl HashState for crate::fields::Field {
    fn hash(&self, h: &mut StateHasher) {
        h.v3(self.pos);
        h.v3(self.vel);
        h.f64(self.age);
        h.f64(self.ttl);
        h.f64(self.heat);
        match self.kind {
            crate::fields::FieldKind::Bubble { radius, .. } => h.f64(radius),
            crate::fields::FieldKind::MicroHole { mass, contained } => {
                h.f64(mass);
                h.bool(contained);
            }
            crate::fields::FieldKind::Fold { exit, stability, .. } => {
                h.v3(exit);
                h.f64(stability);
            }
        }
    }
}
