//! Damage models: impacts, the environment (pressure, tides, radiation, O₂).

use crate::components::{Body, DeathCause, Health, Suit};
use crate::config::DamageCfg;
use crate::fields::FieldSet;
use crate::frame::{Atmosphere, StationFrame};

/// Damage for a normal impact at `speed` (m/s): zero up to the safe speed,
/// quadratic (∝ excess kinetic energy) up to 100 hp at the lethal speed.
pub fn impact(speed: f64, cfg: &DamageCfg) -> f64 {
    if speed <= cfg.impact_safe {
        return 0.0;
    }
    let x = (speed - cfg.impact_safe) / (cfg.impact_lethal - cfg.impact_safe);
    100.0 * x * x
}

/// A sealed suit (EMU-like, 30 kPa of pure O₂) gives the same O₂ partial
/// pressure as sea-level air: breathing-equivalent 101 kPa.
const SUIT_EQUIV_KPA: f64 = 101.0;

/// Ambient effects for one tick. Returns (damage, cause) of the worst source.
pub fn environment(
    b: &Body,
    suit: &mut Suit,
    hp: &mut Health,
    cfg: &DamageCfg,
    frame: &StationFrame,
    atmo: &Atmosphere,
    fields: &FieldSet,
    dt: f64,
) -> Option<(f64, DeathCause)> {
    let head = b.pos + b.axis() * (b.height - 0.1);
    let p = atmo.pressure(frame, head);
    // helmet seals automatically below 70 kPa
    if p < 70_000.0 {
        suit.sealed = true;
    } else if p > 85_000.0 {
        suit.sealed = false;
    }
    let mut worst: Option<(f64, DeathCause)> = None;
    let add = |d: f64, c: DeathCause, w: &mut Option<(f64, DeathCause)>| {
        if d > 0.0 {
            match w {
                Some((x, wc)) => {
                    if d > *x {
                        *wc = c;
                    }
                    *x += d;
                }
                None => *w = Some((d, c)),
            }
        }
    };
    // breathing pressure (kPa equivalent)
    let eff_kpa = if suit.sealed && suit.o2_s > 0.0 {
        // a holed suit uses O₂ faster and its breathing pressure falls towards ambient
        suit.o2_s -= dt * (1.0 + 4.0 * (1.0 - suit.integrity));
        let k = suit.integrity.clamp(0.0, 1.0);
        SUIT_EQUIV_KPA * k + p / 1000.0 * (1.0 - k)
    } else {
        p / 1000.0
    };
    if eff_kpa < cfg.armstrong_kpa {
        add(cfg.ebullism_dps * dt, DeathCause::Decompression, &mut worst);
    } else if eff_kpa < cfg.hypoxia_kpa {
        let x = 1.0 - (eff_kpa - cfg.armstrong_kpa) / (cfg.hypoxia_kpa - cfg.armstrong_kpa);
        add(cfg.hypoxia_dps * x * dt, DeathCause::Decompression, &mut worst);
    }
    // barotrauma from a fast pressure drop (unprotected)
    if !suit.sealed || suit.integrity < 0.5 {
        let rate = (suit.p_prev - p) / 1000.0 / dt.max(1e-6);
        if rate > cfg.barotrauma_kpa_s {
            add((rate - cfg.barotrauma_kpa_s) * cfg.barotrauma_dps_per_kpa_s * dt, DeathCause::Decompression, &mut worst);
        }
    }
    suit.p_prev = p;
    // tidal stretch head-to-foot (black hole, micro holes, fold throats, spin gradient)
    let c = b.centre();
    let ax = b.axis();
    let stretch = frame.stretch(c, ax, b.height) + fields.stretch(c, ax, b.height);
    if stretch > cfg.tidal_safe {
        add((stretch - cfg.tidal_safe) * cfg.tidal_dps_per * dt, DeathCause::Tidal, &mut worst);
    }
    // Hawking radiation: flux × projected area (0.7 m²) → Gy (J/kg) over 80 kg of tissue
    let flux = fields.radiation(c);
    if flux > 1.0 {
        let gy = flux * 0.7 * dt / 80.0;
        hp.dose_gy += gy;
        add(gy * cfg.hp_per_gray, DeathCause::Radiation, &mut worst);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GameConfig;

    #[test]
    fn falls() {
        let c = GameConfig::builtin().damage;
        // 2 m drop in 1 g: 6.3 m/s → safe; 15 m: 17.2 m/s → lethal
        assert_eq!(impact((2.0 * 2.0 * 9.81f64).sqrt(), &c), 0.0);
        assert!(impact((2.0 * 15.0 * 9.81f64).sqrt(), &c) >= 100.0);
        assert!(impact(12.0, &c) > 10.0 && impact(12.0, &c) < 60.0);
    }
}
