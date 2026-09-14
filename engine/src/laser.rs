//! The laser crate: a bolt that crosses the maze five times faster than a
//! bullet, and the aiming line that makes it possible to place.
//!
//! ## Why it is a projectile
//!
//! It was a hitscan first, and that was wrong. Everything this engine can do
//! about a threat assumes the threat exists for some frames and can be watched
//! coming: the observation's ten bullet slots, `THREAT_OFFSET`'s urgency,
//! `score::dodge_safety`, and `risk::incoming_risk`, which flies each round
//! seventy-five frames ahead to decide whether it matters. None of that has
//! anything to read about a weapon that resolves inside one frame. The policy
//! was not failing to dodge the beam — there was no beam to dodge, and against
//! a carrier that could aim it, it died in 88% of rounds.
//!
//! As a bolt it is an ordinary round with two differences: it travels
//! `LASER_SPEED_MULTIPLIER` times faster, and it expires at the end of its
//! range instead of after ten seconds. Every defensive channel starts working
//! again for free, and it is still by a distance the best thing in a crate.
//! Every weapon the original port inherited is a projectile with a speed —
//! `BULLETSPEED`, `FRAGSPEED`, `GATLINGSPEED` — so this also puts it back in
//! the company it was always meant to keep.
//!
//! ## The aiming line
//!
//! `trace` walks where a shot fired now would go. It survives from the hitscan
//! version and is more honest than it was: the bolt really does follow that
//! path, rather than it being a drawing of something already resolved. Its
//! reflection rule is lifted verbatim from `game::bullet_update`, including the
//! asymmetric pair of probes that comment insists must not be "fixed", so the
//! line and the bolt bounce the same way. What the line cannot know is whether
//! something moves into the path first.

use crate::constants as C;
use crate::game::{Bullet, Game};

/// The result of walking a beam: the polyline it covers, and the seat it would
/// be absorbed by. `victim` is `None` for a shot that runs out of range or
/// bounces.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub points: Vec<(f64, f64)>,
    pub victim: Option<usize>,
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
    let mut victim = None;

    // Range is the only limit, and it is the bolt's limit too, so the line and
    // the shot stop in the same place. A separate bounce cap would have drawn
    // one path and flown another.
    while remaining > 0.0 {
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

/// Turn a freshly built round into a laser bolt.
///
/// Called from `Game::fire_weapon` after `Bullet::new`, so the muzzle offset,
/// the magazine accounting, the fire event and the owner's own
/// harmless-until-it-bounces exemption are all shared with the ordinary gun.
/// Only speed and lifetime differ.
pub fn make_bolt(bullet: &mut Bullet) {
    bullet.laser = true;
    bullet.x_speed *= C::LASER_SPEED_MULTIPLIER;
    bullet.y_speed *= C::LASER_SPEED_MULTIPLIER;
    // Expire at the end of the range rather than after ten seconds. A bullet
    // covers `BULLETSPEED * scale / 50` pixels per frame and a cell is `scale`
    // wide, so the cells-per-frame the scale cancels out of is what this needs.
    let cells_per_frame = C::BULLETSPEED * C::LASER_SPEED_MULTIPLIER / 50.0;
    bullet.lifetime = (C::LASER_RANGE_CELLS / cells_per_frame).ceil() as i32;
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

    /// The point of the rewrite: it leaves something in the world. Everything
    /// the engine can do about a threat needs a projectile to look at.
    #[test]
    fn firing_leaves_a_bolt_in_the_air() {
        let mut g = armed(31);
        g.fire_weapon(0);
        assert_eq!(g.bullets.len(), 1);
        let bolt = g.bullets[0];
        assert!(bolt.laser, "the round is not marked as a laser");
        assert_eq!(g.tanks[0].bullets_fired, 1, "a bolt has to occupy the magazine");
        assert!(g.tanks[1].alive, "nothing resolves on the frame it is fired");
    }

    #[test]
    fn a_bolt_outruns_a_bullet_by_the_multiplier() {
        let mut plain = armed(31);
        plain.tanks[0].weapon = Weapon::Normal;
        plain.fire_weapon(0);
        let mut laser = armed(31);
        laser.fire_weapon(0);

        let speed = |g: &Game| g.bullets[0].x_speed.hypot(g.bullets[0].y_speed);
        let ratio = speed(&laser) / speed(&plain);
        assert!((ratio - C::LASER_SPEED_MULTIPLIER).abs() < 1e-9, "ratio was {ratio}");
        // And it burns out at the end of its range instead of after ten
        // seconds, or a bolt at five times the speed would ricochet for the
        // rest of the round.
        assert!(laser.bullets[0].lifetime < plain.bullets[0].lifetime);
        let cells = laser.bullets[0].lifetime as f64
            * C::BULLETSPEED * C::LASER_SPEED_MULTIPLIER / 50.0;
        assert!((cells - C::LASER_RANGE_CELLS).abs() < 1.0, "covered {cells} cells");
    }

    /// The magazine accounting the hitscan version had to skip. A bolt that
    /// did not take a slot would decrement `bullets_fired` below zero when it
    /// expired, and the tank would end the round able to fire for free.
    #[test]
    fn a_spent_bolt_returns_its_magazine_slot() {
        let mut g = armed(31);
        g.fire_weapon(0);
        assert_eq!(g.tanks[0].bullets_fired, 1);
        for _ in 0..g.bullets[0].lifetime + 2 {
            g.step();
        }
        assert_eq!(g.tanks[0].bullets_fired, 0, "the slot never came back");
    }

    /// Same exemption a bullet gets, for the same reason: the muzzle sits
    /// inside the hull's own cell.
    #[test]
    fn the_shooter_is_safe_until_the_bolt_has_bounced() {
        let mut g = armed(31);
        g.fire_weapon(0);
        for _ in 0..4 {
            g.step();
            if !g.bullets.is_empty() && !g.bullets[0].has_bounced {
                assert!(g.tanks[0].alive, "killed by its own outgoing bolt");
            }
        }
    }

    /// A kill has to be attributable, or the drill cannot be scored.
    #[test]
    fn a_bolt_that_lands_is_recorded_as_a_laser() {
        let mut g = armed(31);
        g.tanks[0].rotation = 0.0; // up
        g.tanks[1].x = g.tanks[0].x;
        g.tanks[1].y = g.tanks[0].y - g.scale * 0.45;
        g.fire_weapon(0);
        for _ in 0..6 {
            g.step();
            if !g.tanks[1].alive {
                break;
            }
        }
        assert!(!g.tanks[1].alive, "a point-blank bolt missed");
        let hit = g.hit_records.iter().find(|h| h.victim == g.tanks[1].number);
        assert!(hit.is_some_and(|h| h.laser), "the kill was not attributed to the laser");
    }

    /// The aiming line has to describe the shot, since that is now its only
    /// job: the bolt really does travel this path.
    #[test]
    fn the_line_bounces_off_walls_rather_than_stopping() {
        for seed in [3u32, 17, 31, 4242] {
            let g = armed(seed);
            let trace = trace(&g, 0, g.tanks[0].rotation);
            assert!(trace.points.len() >= 2);
            for &(x, y) in &trace.points {
                assert!(x.is_finite() && y.is_finite());
            }
            // Range-limited, so it always terminates and never runs away.
            let mut travelled = 0.0;
            for pair in trace.points.windows(2) {
                travelled += (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1);
            }
            assert!(travelled <= C::LASER_RANGE_CELLS * g.scale + g.scale,
                    "the line ran past its range: {travelled}");
        }
    }

    #[test]
    fn the_line_reports_a_target_standing_in_it() {
        let mut g = armed(31);
        g.tanks[0].rotation = 0.0;
        g.tanks[1].x = g.tanks[0].x;
        g.tanks[1].y = g.tanks[0].y - g.scale * 0.45;
        assert_eq!(trace(&g, 0, 0.0).victim, Some(1));
        g.tanks[1].alive = false;
        assert_ne!(trace(&g, 0, 0.0).victim, Some(1), "a dead tank is not a target");
    }
}
