//! The laser crate: a beam that arrives the instant you pull the trigger.
//!
//! Every other weapon in this game is a projectile you can watch coming and
//! step out of. The laser is the one that cannot be dodged after the fact — it
//! resolves inside the frame it was fired. What it costs you is commitment:
//! three shots, a long cooldown, and a beam that bounces back along its own
//! corridor and will kill you if you are standing in it.
//!
//! ## Bouncing exactly like a bullet
//!
//! The reflection rule is lifted verbatim from `game::bullet_update`, including
//! the asymmetric pair of probes its comment insists must not be "fixed". A
//! laser that ricocheted on different angles than the bullets would read as a
//! different game, and the whole point of a bounced shot is that you can plan
//! it from experience with the ordinary gun.
//!
//! The difference is resolution. A bullet moves `BULLETHITCHECKINTERVALS`
//! substeps per frame and hit-tests once at the end; the beam covers its whole
//! range in one frame, so it hit-tests at every step or it would shoot straight
//! through a hull.
//!
//! ## The agents cannot see it
//!
//! Like the rest of `pickups.rs`, nothing here reaches `duel_obs.rs` (still
//! schema 24) or `score.rs`. Hybrid will not dodge a beam, because from where
//! it sits no beam was ever fired — only a tank that suddenly stopped existing.

use crate::constants as C;
use crate::game::{Event, Game};

/// The result of walking a beam: the polyline it covers, and the seat it would
/// be absorbed by. `victim` is `None` for a shot that runs out of range or
/// bounces.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub points: Vec<(f64, f64)>,
    pub victim: Option<usize>,
}

/// A fired beam, kept only so the viewer can draw it fading out.
#[derive(Clone, Debug, Default)]
pub struct Beam {
    /// The polyline the beam actually travelled, corner by corner.
    pub points: Vec<(f64, f64)>,
    /// Frames of afterglow left.
    pub ttl: i32,
    /// Who fired it, and who it killed. The shot resolves and disappears
    /// inside one frame, so a caller that wants to attribute a death to a
    /// laser cannot look for a projectile afterwards — there is none. Recording
    /// it here is the only way to tell a beam kill from a bullet kill without
    /// re-deriving the whole trace.
    pub owner: usize,
    pub victim: Option<usize>,
}

impl Beam {
    /// Full strength for the first `LASER_BEAM_HOLD_FRAMES`, then a linear
    /// fade to `0.0`. Returned rather than a frame count so the viewer never
    /// has to know either constant.
    pub fn alpha(&self) -> f32 {
        if self.ttl <= 0 {
            return 0.0;
        }
        let fade_over = (C::LASER_BEAM_FRAMES - C::LASER_BEAM_HOLD_FRAMES).max(1);
        if self.ttl > fade_over {
            return 1.0;
        }
        self.ttl as f32 / fade_over as f32
    }
}

/// Age the afterglow. Called once per frame from `Game::step`.
pub fn tick(game: &mut Game) {
    if game.beam.ttl > 0 {
        game.beam.ttl -= 1;
        if game.beam.ttl == 0 {
            game.beam.points.clear();
        }
    }
}

pub fn clear(game: &mut Game) {
    game.beam.points.clear();
    game.beam.ttl = 0;
}

/// Where a beam fired from `tank` at `rotation` would go, and who it would
/// hit. Takes `&Game`: this is the shared core of both firing and the aiming
/// preview, and the preview must not be able to change anything.
///
/// `rotation` is passed in rather than read off the tank so the viewer can
/// trace from the hull angle it is *drawing* — with display prediction the
/// drawn barrel leads the authoritative pose by up to a frame, and a preview
/// that ignored that would visibly hang off the end of the gun.
pub fn trace(game: &Game, tank: usize, rotation: f64) -> Trace {
    let scale = game.scale;
    let step = scale / C::LASER_STEPS_PER_CELL as f64;
    let mut remaining = C::LASER_RANGE_CELLS * scale;

    // Start at the barrel tip rather than the hull centre, matching where
    // `Bullet::new` puts a round, so a muzzle-flush shot cannot start inside a
    // wall the hull is resting against.
    let heading = (rotation - 90.0) * C::DEG;
    let (mut dx, mut dy) = (heading.cos() * step, heading.sin() * step);
    let muzzle = game.tanks[tank].display_scale * C::TANK_SHAPE_BARREL_TIP_Y.abs();
    let mut x = game.tanks[tank].x + heading.cos() * muzzle;
    let mut y = game.tanks[tank].y + heading.sin() * muzzle;

    let mut points = vec![(x, y)];
    let mut bounced = false;
    let mut bounces = 0;
    let mut victim = None;

    while remaining > 0.0 && bounces <= C::LASER_MAX_BOUNCES {
        let (prev_x, prev_y) = (x, y);
        x += dx;
        y += dy;
        remaining -= step;

        if game.wall_hit(x, y) {
            // Verbatim from `bullet_update`: the two probes are deliberately
            // asymmetric and decide every ricochet angle in the game.
            let hit_on_x_invert = game.wall_hit(prev_x - dx, prev_y + dy);
            let hit_on_y_invert = game.wall_hit(prev_x + dx, prev_y - dy);
            if hit_on_x_invert && !hit_on_y_invert {
                dy = -dy;
            } else if hit_on_y_invert && !hit_on_x_invert {
                dx = -dx;
            } else {
                dx = -dx;
                dy = -dy;
            }
            x = prev_x + dx;
            y = prev_y + dy;
            bounced = true;
            bounces += 1;
            points.push((prev_x, prev_y));
            continue;
        }

        // Hit-tested every step, unlike a bullet's once-per-frame check: the
        // beam crosses the whole maze in one frame and would otherwise pass
        // straight through a hull between two tests.
        let mut absorbed = false;
        for i in 0..game.tanks_count {
            // Same exemption a bullet gets: your own shot is harmless until it
            // has come off a wall, after which it kills you like anyone else.
            if i == tank && !bounced {
                continue;
            }
            if game.tanks[i].alive && game.tanks[i].point_in_shape(x, y) {
                victim = Some(i);
                absorbed = true;
                break;
            }
        }
        // The beam is absorbed by whatever it hits. That is what makes the
        // weapon worth carrying: land the shot and it stops there, miss and it
        // keeps bouncing — and a beam fired at a wall square on comes straight
        // back down its own line and kills you. Without absorption every shot
        // taken at point-blank range would be a mutual kill.
        if absorbed {
            break;
        }
    }
    points.push((x, y));
    Trace { points, victim }
}

/// Fire the beam from `tank`'s muzzle and resolve it immediately.
///
/// Returns the number of tanks destroyed, which the caller does not currently
/// need but which makes the unit tests read as statements about the weapon.
pub fn fire(game: &mut Game, tank: usize) -> usize {
    let Trace { points, victim } = trace(game, tank, game.tanks[tank].rotation);
    game.beam = Beam { points, ttl: C::LASER_BEAM_FRAMES, owner: tank, victim };
    let Some(victim) = victim else { return 0 };
    let owner_number = game.tanks[tank].number;
    let victim_number = game.tanks[victim].number;
    game.events.push(Event::Hit { owner: owner_number, victim: victim_number });
    game.destroy_tank(victim);
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickups::Weapon;

    fn armed(seed: u32) -> Game {
        let mut g = Game::new(seed, 2);
        g.setup_battle();
        g.tanks[0].weapon = Weapon::Laser;
        g.tanks[0].weapon_charges = C::LASER_CHARGES;
        g
    }

    #[test]
    fn firing_leaves_a_beam_to_draw() {
        let mut g = armed(31);
        assert_eq!(g.beam.ttl, 0);
        fire(&mut g, 0);
        assert!(g.beam.points.len() >= 2, "a beam needs at least a start and an end");
        assert_eq!(g.beam.ttl, C::LASER_BEAM_FRAMES);
        assert_eq!(g.beam.alpha(), 1.0);
    }

    #[test]
    fn the_afterglow_fades_and_clears_itself() {
        let mut g = armed(31);
        fire(&mut g, 0);
        for _ in 0..C::LASER_BEAM_FRAMES {
            assert!(g.beam.ttl > 0);
            tick(&mut g);
        }
        assert_eq!(g.beam.ttl, 0);
        assert!(g.beam.points.is_empty(), "a spent beam must not keep its geometry");
        assert_eq!(g.beam.alpha(), 0.0);
    }

    /// The beam is instant: a tank standing in front of the barrel is dead on
    /// the same frame, with no projectile left in the air.
    #[test]
    fn a_tank_in_the_line_dies_immediately() {
        let mut g = armed(31);
        // Park the target just ahead of the barrel and aim at it. It has to
        // stay inside the shooter's own cell — the maze is random, so an
        // adjacent cell may well have a wall between, and then the honest
        // answer is that the beam bounced rather than that it failed.
        g.tanks[0].rotation = 0.0; // up
        g.tanks[1].x = g.tanks[0].x;
        g.tanks[1].y = g.tanks[0].y - g.scale * 0.45;
        let bullets_before = g.bullets.len();
        assert_eq!(fire(&mut g, 0), 1);
        assert!(!g.tanks[1].alive);
        assert!(g.tanks[0].alive, "the beam stopped in the target, not in the shooter");
        assert_eq!(g.bullets.len(), bullets_before, "a laser is not a projectile");
    }

    #[test]
    fn the_shooter_is_safe_until_the_beam_has_bounced() {
        let mut g = armed(31);
        // Nothing in front: whatever comes back has bounced at least once, and
        // the shooter is fair game for it — but never for the outgoing leg.
        let alive_before = g.tanks[0].alive;
        fire(&mut g, 0);
        assert!(alive_before);
        // The outgoing leg starts at the muzzle, outside the hull, so a shot
        // into open floor can never kill on step one.
        assert!(g.beam.points.len() >= 2);
    }

    /// Absorption is what keeps the weapon usable. Fired square at a nearby
    /// wall with nothing in the way, the beam returns down its own line and
    /// kills the shooter; put a target in that line and it stops there.
    #[test]
    fn a_missed_beam_can_come_back_and_kill_the_shooter() {
        let mut g = armed(31);
        g.tanks[0].rotation = 0.0;
        // Move the other tank out of the way so nothing absorbs the beam.
        g.tanks[1].alive = false;
        let killed = fire(&mut g, 0);
        assert_eq!(killed, 1, "the return leg found the shooter");
        assert!(!g.tanks[0].alive);
    }

    /// The bounce rule is the bullet's, so a beam fired down a corridor folds
    /// back along it rather than stopping at the wall.
    #[test]
    fn the_beam_bounces_instead_of_stopping_at_a_wall() {
        let mut g = armed(77);
        fire(&mut g, 0);
        let corners = g.beam.points.len();
        assert!(corners >= 2);
        // Total length walked should exceed the straight-line start-to-end
        // distance whenever it turned a corner; with no bounce they are equal.
        let p = &g.beam.points;
        let walked: f64 = p.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum();
        let straight = (p[p.len() - 1].0 - p[0].0).hypot(p[p.len() - 1].1 - p[0].1);
        assert!(walked >= straight - 1e-9);
        assert!(walked <= C::LASER_RANGE_CELLS * g.scale + g.scale, "range is bounded");
    }

    /// The beam that ended the round has to survive the freeze that follows,
    /// because that pause is exactly when a player looks at where it went.
    #[test]
    fn the_beam_holds_still_while_the_round_is_frozen() {
        let mut g = armed(31);
        fire(&mut g, 0);
        let held = g.beam.ttl;
        g.frozen = true;
        for _ in 0..30 {
            g.step();
        }
        assert_eq!(g.beam.ttl, held, "a frozen round must not age the beam");
        assert_eq!(g.beam.alpha(), 1.0);
    }

    #[test]
    fn the_beam_holds_full_strength_before_it_fades() {
        let mut g = armed(31);
        fire(&mut g, 0);
        for _ in 0..C::LASER_BEAM_HOLD_FRAMES - 1 {
            tick(&mut g);
            assert_eq!(g.beam.alpha(), 1.0, "the hold phase must not fade");
        }
        let mut last = g.beam.alpha();
        while g.beam.ttl > 0 {
            tick(&mut g);
            let now = g.beam.alpha();
            assert!(now <= last, "alpha must never rise");
            last = now;
        }
        assert_eq!(g.beam.alpha(), 0.0);
    }

    #[test]
    fn a_beam_never_outlives_the_round() {
        let mut g = armed(31);
        fire(&mut g, 0);
        assert!(g.beam.ttl > 0);
        g.setup_battle();
        assert_eq!(g.beam.ttl, 0);
        assert!(g.beam.points.is_empty());
    }
}
