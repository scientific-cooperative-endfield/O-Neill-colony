//! Utility AI: every decision tick each goal gets a score from the bot's
//! blackboard; the best one wins, with a hysteresis bonus for the current
//! goal and a minimum commitment time so bots do not dither.

use crate::config::Difficulty;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Goal {
    Patrol,
    Investigate,
    Attack,
    SeekCover,
    Retreat,
    Flank,
}

impl Goal {
    pub const ALL: [Goal; 6] = [Goal::Patrol, Goal::Investigate, Goal::Attack, Goal::SeekCover, Goal::Retreat, Goal::Flank];
    pub fn label(self) -> &'static str {
        match self {
            Goal::Patrol => "patrol",
            Goal::Investigate => "investigate",
            Goal::Attack => "attack",
            Goal::SeekCover => "cover",
            Goal::Retreat => "retreat",
            Goal::Flank => "flank",
        }
    }
}

/// What the brain looks at (filled by `bot.rs` from perception and own state).
#[derive(Clone, Copy, Debug, Default)]
pub struct Blackboard {
    /// own health fraction
    pub hp: f64,
    /// a visible enemy and its distance
    pub visible: Option<f64>,
    /// seconds since the best known (not visible) contact was last seen / heard
    pub known_age: Option<f64>,
    /// seconds since we were last hit
    pub hit_age: Option<f64>,
    /// a loud noise to investigate (age, s)
    pub noise_age: Option<f64>,
    pub reloading: bool,
    pub out_of_ammo: bool,
    /// a usable cover spot exists near us
    pub cover_available: bool,
    pub in_cover: bool,
    /// number of enemies currently visible
    pub n_visible: usize,
}

pub fn score(g: Goal, b: &Blackboard, d: &Difficulty) -> f64 {
    let threat = b.visible.is_some() || b.hit_age.is_some_and(|a| a < 2.0);
    let hurt = 1.0 - b.hp;
    match g {
        Goal::Patrol => 0.2,
        Goal::Investigate => {
            let k = b.known_age.map_or(0.0, |a| (1.0 - a / 15.0).max(0.0) * 0.5);
            let n = b.noise_age.map_or(0.0, |a| (1.0 - a / 10.0).max(0.0) * 0.4);
            k.max(n)
        }
        Goal::Attack => match b.visible {
            Some(dist) => {
                let mut s = 0.65 + 0.25 * b.hp - 0.05 * (b.n_visible.saturating_sub(1)) as f64;
                if b.reloading || b.out_of_ammo {
                    s -= 0.4;
                }
                if dist > 900.0 {
                    s -= 0.3;
                }
                s
            }
            None => 0.0,
        },
        Goal::SeekCover => {
            if !threat || !b.cover_available && !b.in_cover {
                return 0.0;
            }
            let mut s = 0.35 + 0.6 * hurt * d.caution;
            if b.reloading {
                s += 0.35;
            }
            if b.hit_age.is_some_and(|a| a < 1.0) {
                s += 0.2 * d.caution;
            }
            if b.n_visible > 1 {
                s += 0.1;
            }
            s
        }
        Goal::Retreat => {
            if !threat {
                return 0.0;
            }
            let mut s = if b.hp < 0.15 + 0.25 * d.caution { 0.6 + 0.4 * hurt } else { 0.0 };
            if b.out_of_ammo {
                s = s.max(0.85);
            }
            s
        }
        Goal::Flank => match (b.visible, b.known_age) {
            (None, Some(a)) if (1.5..12.0).contains(&a) && b.hp > 0.5 => 0.42 + 0.2 * (1.0 - d.caution),
            _ => 0.0,
        },
    }
}

/// Picks the next goal. `current_age` = seconds the current goal has been active.
pub fn choose(current: Goal, current_age: f64, b: &Blackboard, d: &Difficulty) -> Goal {
    let mut best = (current, score(current, b, d) + if current_age < 1.0 { 0.3 } else { 0.1 });
    for g in Goal::ALL {
        let s = score(g, b, d);
        if s > best.1 {
            best = (g, s);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d() -> Difficulty {
        crate::config::by_name(&crate::config::builtin(), "regular").unwrap()
    }

    #[test]
    fn decisions_make_sense() {
        let d = d();
        let calm = Blackboard { hp: 1.0, ..Default::default() };
        assert_eq!(choose(Goal::Patrol, 5.0, &calm, &d), Goal::Patrol);
        let fight = Blackboard { hp: 1.0, visible: Some(120.0), n_visible: 1, cover_available: true, ..Default::default() };
        assert_eq!(choose(Goal::Patrol, 5.0, &fight, &d), Goal::Attack);
        let reload = Blackboard { reloading: true, hit_age: Some(0.5), ..fight };
        assert_eq!(choose(Goal::Attack, 5.0, &reload, &d), Goal::SeekCover);
        let dying = Blackboard { hp: 0.12, hit_age: Some(0.3), cover_available: false, ..fight };
        assert_eq!(choose(Goal::Attack, 5.0, &dying, &d), Goal::Retreat);
        let lost = Blackboard { hp: 0.9, known_age: Some(4.0), ..Default::default() };
        let g = choose(Goal::Attack, 5.0, &lost, &d);
        assert!(g == Goal::Flank || g == Goal::Investigate, "{g:?}");
        let shot = Blackboard { hp: 1.0, noise_age: Some(1.0), ..Default::default() };
        assert_eq!(choose(Goal::Patrol, 5.0, &shot, &d), Goal::Investigate);
    }
}
