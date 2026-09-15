//! The homing missile: a bullet that steers.
//!
//! It is deliberately nothing more than that. It leaves the muzzle at a
//! bullet's speed, bounces off walls on a bullet's rules, dies on a bullet's
//! hit test, and occupies a bullet's slot in the magazine. The only thing it
//! does differently is turn a few degrees a frame toward the other tank, for
//! `HOMING_FRAMES`, after which it flies on as an ordinary round.
//!
//! Keeping it a bullet is what makes it cheap. The observation's ten bullet
//! slots already carry its position, velocity and bounce state;
//! `risk::incoming_risk` already flies it forward to decide whether it
//! threatens anyone. None of that had to be told the missile exists.
//!
//! What those channels get wrong is the *future*: they extrapolate a straight
//! line, and this one curves. So a missile reads as a threat that is about to
//! miss right up until it comes around, which is exactly the property that
//! makes it worth having — a weapon the machinery sees but cannot fully
//! predict. That is a fair thing to lose to, unlike a hitscan beam that
//! nothing could see at all.
//!
//! ## Homing through walls
//!
//! Steering is toward the other tank's bearing, whether or not there is a wall
//! in between. The missile does not phase through anything — it bounces like
//! any round — so what this produces is a missile that presses against the
//! wall you are hiding behind and ricochets around it, rather than one that
//! gives up because it lost sight of you. Cover still works; it just has to be
//! cover you keep moving behind.

use crate::constants as C;
use crate::game::{Bullet, Game};

/// Arm a freshly built round as a missile. Called from `Game::fire_weapon`
/// after `Bullet::new`, so the muzzle offset, the magazine accounting and the
/// owner's harmless-until-it-bounces exemption are all shared with the gun.
pub fn make_missile(bullet: &mut Bullet) {
    bullet.homing_frames = C::HOMING_FRAMES;
}

/// Turn one missile toward its target. Called once a frame, before the
/// substeps, so the curve is made of per-frame chords rather than the velocity
/// being rewritten underneath a half-finished step.
pub fn steer(game: &Game, bullet: &mut Bullet) {
    if bullet.homing_frames <= 0 {
        return;
    }
    bullet.homing_frames -= 1;

    // Two tanks, so the target is simply the other seat. A missile does not
    // chase a corpse: once the round is decided it flies straight, which keeps
    // the settlement window from being swept by a shot that no longer matters.
    let target = 1 - bullet.owner.min(1);
    if target >= game.tanks_count || !game.tanks[target].alive {
        return;
    }

    let speed = bullet.x_speed.hypot(bullet.y_speed);
    if speed <= 0.0 {
        return;
    }
    let wanted = (game.tanks[target].y - bullet.y).atan2(game.tanks[target].x - bullet.x);
    let heading = bullet.y_speed.atan2(bullet.x_speed);
    let mut delta = wanted - heading;
    // Shortest way round, so a missile pointing just past its target turns the
    // two degrees back rather than the three hundred and fifty-eight forward.
    while delta > std::f64::consts::PI {
        delta -= std::f64::consts::TAU;
    }
    while delta < -std::f64::consts::PI {
        delta += std::f64::consts::TAU;
    }

    let limit = C::HOMING_TURN_DEGREES * C::DEG;
    let turn = delta.clamp(-limit, limit);
    // Rotate the velocity rather than rebuilding it from the bearing: speed is
    // then preserved exactly, and a missile that has bounced keeps whatever
    // the wall did to it.
    let (sin, cos) = turn.sin_cos();
    let (vx, vy) = (bullet.x_speed, bullet.y_speed);
    bullet.x_speed = vx * cos - vy * sin;
    bullet.y_speed = vx * sin + vy * cos;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickups::Weapon;

    fn armed(seed: u32) -> Game {
        let mut g = Game::new(seed, 2);
        g.setup_battle();
        g.tanks[0].weapon = Weapon::Homing;
        g.tanks[0].weapon_charges = C::HOMING_CHARGES;
        g
    }

    #[test]
    fn firing_leaves_a_missile_with_a_chase_clock() {
        let mut g = armed(31);
        g.fire_weapon(0);
        assert_eq!(g.bullets.len(), 1);
        assert_eq!(g.bullets[0].homing_frames, C::HOMING_FRAMES);
        assert_eq!(g.tanks[0].bullets_fired, 1, "a missile occupies the magazine");
    }

    /// The point of the weapon: pointed away from the target, it comes about.
    #[test]
    fn a_missile_turns_toward_the_other_tank() {
        let mut g = armed(31);
        // Put the target directly above and fire directly downward.
        g.tanks[1].x = g.tanks[0].x;
        g.tanks[1].y = g.tanks[0].y - g.scale * 3.0;
        g.tanks[0].rotation = 180.0;
        g.fire_weapon(0);

        let bearing_error = |b: &Bullet, g: &Game| {
            let wanted = (g.tanks[1].y - b.y).atan2(g.tanks[1].x - b.x);
            let heading = b.y_speed.atan2(b.x_speed);
            let mut d = wanted - heading;
            while d > std::f64::consts::PI { d -= std::f64::consts::TAU; }
            while d < -std::f64::consts::PI { d += std::f64::consts::TAU; }
            d.abs()
        };
        let before = bearing_error(&g.bullets[0], &g);
        for _ in 0..10 {
            let mut b = g.bullets[0];
            steer(&g, &mut b);
            g.bullets[0] = b;
        }
        let after = bearing_error(&g.bullets[0], &g);
        assert!(after < before, "aim did not improve: {before} -> {after}");
        assert!(after < before - 10.0 * C::HOMING_TURN_DEGREES * C::DEG * 0.9,
                "turned less than the rate allows: {before} -> {after}");
    }

    #[test]
    fn steering_preserves_speed() {
        let mut g = armed(31);
        g.fire_weapon(0);
        let speed = |b: &Bullet| b.x_speed.hypot(b.y_speed);
        let before = speed(&g.bullets[0]);
        for _ in 0..20 {
            let mut b = g.bullets[0];
            steer(&g, &mut b);
            g.bullets[0] = b;
        }
        let after = speed(&g.bullets[0]);
        assert!((after - before).abs() < 1e-9, "{before} -> {after}");
    }

    #[test]
    fn the_chase_runs_out_and_the_missile_flies_straight() {
        let mut g = armed(31);
        g.fire_weapon(0);
        for _ in 0..C::HOMING_FRAMES {
            let mut b = g.bullets[0];
            steer(&g, &mut b);
            g.bullets[0] = b;
        }
        assert_eq!(g.bullets[0].homing_frames, 0);
        let mut b = g.bullets[0];
        let (vx, vy) = (b.x_speed, b.y_speed);
        steer(&g, &mut b);
        assert_eq!((b.x_speed, b.y_speed), (vx, vy), "still steering past its clock");
    }

    /// A decided round must not be swept by a missile still in the air.
    #[test]
    fn a_missile_does_not_chase_a_dead_tank() {
        let mut g = armed(31);
        g.fire_weapon(0);
        g.tanks[1].alive = false;
        let mut b = g.bullets[0];
        let (vx, vy) = (b.x_speed, b.y_speed);
        steer(&g, &mut b);
        assert_eq!((b.x_speed, b.y_speed), (vx, vy));
    }

    /// An ordinary round is never steered, so nothing about the base game
    /// changes when the missile is merely available.
    #[test]
    fn an_ordinary_round_is_left_alone() {
        let mut g = armed(31);
        g.tanks[0].weapon = Weapon::Normal;
        g.fire_weapon(0);
        assert_eq!(g.bullets[0].homing_frames, 0);
        let mut b = g.bullets[0];
        let (vx, vy) = (b.x_speed, b.y_speed);
        steer(&g, &mut b);
        assert_eq!((b.x_speed, b.y_speed), (vx, vy));
    }
}
