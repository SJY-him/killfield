//! Weapon crates, in the spirit of the original Tank Trouble.
//!
//! The port kept the original's crate *cadence* — `game.rs`'s `crate_timer`
//! counts down and rerolls itself on the original formula — but dropped the
//! crates themselves, because the duel curriculum wanted a clean two-tank
//! fight. That timer could not simply be deleted: it draws from the game RNG
//! every reroll, so removing it would shift the whole random stream and
//! invalidate every recorded benchmark. This module hangs the crates back onto
//! that surviving hook.
//!
//! ## Determinism
//!
//! Placement draws from `Game::pickup_rng`, a chain of its own, never from
//! `Game::rng`. With `pickups_enabled` false, nothing here touches any state
//! the rest of the engine reads, so a build with this module compiled in still
//! plays a seed bit-for-bit identically to one without it. That matters
//! because the Hybrid checkpoint and the MPC benchmarks were both measured on
//! the untouched stream.
//!
//! ## The agents cannot see any of this
//!
//! `duel_obs.rs` is schema 24 and stays schema 24: no crate reaches Hybrid's
//! observation, and `score.rs` does not model pickups either. Both agents will
//! walk over a crate without meaning to and will never walk toward one. That
//! asymmetry is the point — it is the one axis on which a human outplays them.

use crate::constants as C;
use crate::game::{Bullet, Game};
use crate::rng::Rng;

/// What a crate grants. `Normal` is the default gun and is never in a crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weapon {
    Normal,
    /// Rapid fire from a deep magazine, at the cost of the usual five-in-flight
    /// discipline being the only thing keeping you from shooting yourself.
    Gatling,
    /// One trigger pull, several pellets across a spread. Trades precision for
    /// not needing any.
    Shotgun,
    /// Not a gun: absorbs exactly one lethal hit, including your own ricochet.
    Shield,
    /// A fast bouncing bolt. See `laser.rs`.
    Laser,
    /// A bullet that steers toward the other tank for a few seconds.
    Homing,
}

impl Weapon {
    /// The order crates roll from. `Normal` is excluded deliberately.
    pub const DROPS: [Weapon; 5] = [
        Weapon::Gatling, Weapon::Shotgun, Weapon::Shield, Weapon::Laser,
        Weapon::Homing,
    ];

    /// Wire encoding for the render buffer and the viewer.
    pub fn code(self) -> f32 {
        match self {
            Weapon::Normal => 0.0,
            Weapon::Gatling => 1.0,
            Weapon::Shotgun => 2.0,
            Weapon::Shield => 3.0,
            Weapon::Laser => 4.0,
            Weapon::Homing => 5.0,
        }
    }

    /// How many shots the pickup carries. `Shield` is a state, not a magazine.
    pub fn charges(self) -> i32 {
        match self {
            Weapon::Normal | Weapon::Shield => 0,
            Weapon::Gatling => C::GATLING_CHARGES,
            Weapon::Shotgun => C::SHOTGUN_CHARGES,
            Weapon::Laser => C::LASER_CHARGES,
            Weapon::Homing => C::HOMING_CHARGES,
        }
    }

    /// Frames the trigger is locked after a shot. The default gun has no
    /// cooldown at all — it is limited by the five-bullet magazine instead.
    pub fn cooldown(self) -> i32 {
        match self {
            Weapon::Gatling => C::GATLING_COOLDOWN_FRAMES,
            Weapon::Laser => C::LASER_COOLDOWN_FRAMES,
            Weapon::Homing => C::HOMING_COOLDOWN_FRAMES,
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pickup {
    pub x: f64,
    pub y: f64,
    pub weapon: Weapon,
}

/// Clear the field and reroll the spawn clock. Called from `setup_battle`.
pub fn reset_round(game: &mut Game) {
    game.pickups.clear();
    game.pickup_timer = first_delay(game);
}

fn first_delay(game: &mut Game) -> f64 {
    // Shaped like the original's cadence — fixed base, random spread, a term
    // that makes big mazes wait longer — but on its own constants. See the
    // note above `PICKUP_SPAWN_TIMEBASE` for why it cannot borrow the old ones.
    C::PICKUP_SPAWN_TIMEBASE
        + game.pickup_rng.randrange(C::PICKUP_SPAWN_TIMERANDOM) as f64
        + C::PICKUP_SPAWN_MAZESIZESCALE / game.reachable.len().max(1) as f64
}

/// Tick the spawn clock and drop a crate when it expires. No-op while the
/// round is frozen, so the settlement window does not litter the floor.
pub fn tick_spawn(game: &mut Game) {
    if !game.pickups_enabled || game.frozen {
        return;
    }
    game.pickup_timer -= 1.0;
    if game.pickup_timer > 0.0 {
        return;
    }
    game.pickup_timer = first_delay(game);
    if game.pickups.len() >= C::CRATE_MAX_ON_FIELD {
        return;
    }
    if let Some((cx, cy)) = free_cell(game) {
        let half = game.scale / 2.0;
        let weapon = Weapon::DROPS[game.pickup_rng.randrange(Weapon::DROPS.len() as i32) as usize];
        game.pickups.push(Pickup {
            x: cx as f64 * game.scale + half,
            y: cy as f64 * game.scale + half,
            weapon,
        });
    }
}

/// A reachable cell holding no crate and no tank. Cell centres are always
/// clear: walls live on cell edges and are far thinner than half a cell.
fn free_cell(game: &mut Game) -> Option<(usize, usize)> {
    let n = game.reachable.len();
    if n == 0 {
        return None;
    }
    // Bounded probing rather than building a candidate list every spawn: the
    // floor is never so crowded that a handful of draws fails to find a gap.
    for _ in 0..C::CRATE_SPAWN_ATTEMPTS {
        let k = game.pickup_rng.randrange(n as i32) as usize;
        let (cx, cy) = game.reachable[k];
        let half = game.scale / 2.0;
        let (x, y) = (cx as f64 * game.scale + half, cy as f64 * game.scale + half);
        let clear_of_crates = game
            .pickups
            .iter()
            .all(|p| (p.x - x).abs() > half || (p.y - y).abs() > half);
        // Never drop one into somebody's lap; that reads as a random gift.
        let clear_of_tanks = game.tanks.iter().all(|t| {
            !t.alive || (t.x - x).hypot(t.y - y) > C::CRATE_SPAWN_CLEARANCE_CELLS * game.scale
        });
        if clear_of_crates && clear_of_tanks {
            return Some((cx, cy));
        }
    }
    None
}

/// Hand out any crate a live tank is standing on. Later pickups replace an
/// earlier weapon outright — no stacking, same as the original.
pub fn collect(game: &mut Game) {
    if !game.pickups_enabled || game.pickups.is_empty() {
        return;
    }
    let radius = C::CRATE_PICKUP_RADIUS_CELLS * game.scale;
    let mut taken: Option<(usize, Weapon)> = None;
    'outer: for i in 0..game.tanks_count {
        if !game.tanks[i].alive {
            continue;
        }
        for (k, p) in game.pickups.iter().enumerate() {
            if (game.tanks[i].x - p.x).hypot(game.tanks[i].y - p.y) <= radius {
                taken = Some((k, p.weapon));
                break 'outer;
            }
        }
    }
    let Some((index, weapon)) = taken else { return };
    let tank = (0..game.tanks_count)
        .find(|&i| {
            game.tanks[i].alive
                && (game.tanks[i].x - game.pickups[index].x)
                    .hypot(game.tanks[i].y - game.pickups[index].y)
                    <= radius
        })
        .expect("the tank that triggered the pickup is still standing there");
    game.pickups.remove(index);
    equip(game, tank, weapon);
    // One crate per frame keeps the borrow simple; two tanks reaching the same
    // crate on one tick is decided by seat order, which is stable and rare.
    collect(game);
}

fn equip(game: &mut Game, tank: usize, weapon: Weapon) {
    match weapon {
        Weapon::Shield => game.tanks[tank].shield = true,
        _ => {
            game.tanks[tank].weapon = weapon;
            game.tanks[tank].weapon_charges = weapon.charges();
            game.tanks[tank].fire_cooldown = 0;
        }
    }
}

/// Drop back to the default gun once the special magazine runs dry.
pub fn note_shot(game: &mut Game, tank: usize) {
    let t = &mut game.tanks[tank];
    t.fire_cooldown = t.weapon.cooldown();
    if t.weapon == Weapon::Normal {
        return;
    }
    t.weapon_charges -= 1;
    if t.weapon_charges <= 0 {
        t.weapon = Weapon::Normal;
        t.weapon_charges = 0;
    }
}

/// Extra barrels for one trigger pull. The centre pellet is the ordinary shot
/// `fire_weapon` already pushed, so this only adds the ones either side of it.
pub fn extra_pellets(game: &mut Game, tank: usize) {
    if game.tanks[tank].weapon != Weapon::Shotgun {
        return;
    }
    let base = game.tanks[tank].rotation;
    let scale = game.scale;
    for i in 1..=C::SHOTGUN_SIDE_PELLETS {
        for sign in [-1.0f64, 1.0] {
            game.bullet_depth += 1;
            let mut source = game.tanks[tank];
            source.rotation = base + sign * i as f64 * C::SHOTGUN_SPREAD_DEG;
            let mut b = Bullet::new(game.bullet_depth, tank, &source, scale);
            b.just_created = true;
            game.bullets.push(b);
        }
    }
}

/// A dedicated stream so crate placement never perturbs the game's own RNG.
pub fn new_rng(seed: u32) -> Rng {
    // Any offset but `Rng::new`'s own diffusion constant, which would cancel
    // against the xor inside it and hand back the game's exact chain.
    Rng::new(seed ^ 0xC0FF_EE01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;

    fn run(seed: u32, frames: usize, pickups: bool) -> Game {
        let mut g = Game::new(seed, 2);
        g.setup_battle();
        g.pickups_enabled = pickups;
        if pickups {
            reset_round(&mut g);
        }
        for _ in 0..frames {
            g.step();
        }
        g
    }

    /// The whole point of the separate chain: turning crates on must not move
    /// the maze, the spawns, or anything else the benchmarks were measured on.
    #[test]
    fn enabling_crates_does_not_disturb_the_game_rng() {
        let off = run(4242, 900, false);
        let on = run(4242, 900, true);
        assert_eq!(off.rng.state, on.rng.state, "game RNG diverged");
        assert_eq!(off.maze.w, on.maze.w);
        assert_eq!(off.maze.h, on.maze.h);
        assert_eq!(off.round_number, on.round_number);
        for i in 0..off.tanks_count {
            assert_eq!(off.tanks[i].x.to_bits(), on.tanks[i].x.to_bits());
            assert_eq!(off.tanks[i].y.to_bits(), on.tanks[i].y.to_bits());
        }
    }

    #[test]
    fn nothing_spawns_while_crates_are_off() {
        assert!(run(7, 1500, false).pickups.is_empty());
    }

    /// A round against these agents lasts roughly 125 frames, so the first
    /// crate has to be reachable well inside that or the feature is dead code.
    #[test]
    fn the_first_crate_lands_inside_a_typical_round() {
        let early = run(7, 40, true);
        assert!(early.pickups.is_empty(), "a crate landed before the base delay");
        let mid = run(7, 150, true);
        assert!(!mid.pickups.is_empty(), "no crate within a normal round");
        let long = run(7, 1500, true);
        assert!(long.pickups.len() <= C::CRATE_MAX_ON_FIELD, "the floor filled up");
    }

    /// Crates sit at cell centres, which are always clear of walls.
    #[test]
    fn crates_land_on_cell_centres_inside_the_maze() {
        let g = run(11, 2000, true);
        assert!(!g.pickups.is_empty());
        for p in &g.pickups {
            let cx = (p.x / g.scale - 0.5).round();
            let cy = (p.y / g.scale - 0.5).round();
            assert!((p.x - (cx * g.scale + g.scale / 2.0)).abs() < 1e-9);
            assert!((p.y - (cy * g.scale + g.scale / 2.0)).abs() < 1e-9);
            assert!(cx >= 0.0 && (cx as usize) < g.maze.w);
            assert!(cy >= 0.0 && (cy as usize) < g.maze.h);
            assert!(g.reachable.contains(&(cx as usize, cy as usize)));
        }
    }

    #[test]
    fn driving_onto_a_crate_collects_it() {
        let mut g = Game::new(3, 2);
        g.setup_battle();
        g.pickups_enabled = true;
        let (x, y) = (g.tanks[0].x, g.tanks[0].y);
        g.pickups.push(Pickup { x, y, weapon: Weapon::Gatling });
        collect(&mut g);
        assert!(g.pickups.is_empty(), "the crate was not consumed");
        assert_eq!(g.tanks[0].weapon, Weapon::Gatling);
        assert_eq!(g.tanks[0].weapon_charges, C::GATLING_CHARGES);
        assert_eq!(g.tanks[1].weapon, Weapon::Normal, "the other seat got nothing");
    }

    /// `collect` being correct is worth nothing if `step` never calls it. This
    /// is the wiring test: drop a crate under a tank and advance one frame.
    #[test]
    fn stepping_the_game_collects_a_crate_underfoot() {
        let mut g = Game::new(21, 2);
        g.setup_battle();
        g.pickups_enabled = true;
        let (x, y) = (g.tanks[0].x, g.tanks[0].y);
        g.pickups.push(Pickup { x, y, weapon: Weapon::Shield });
        assert!(!g.tanks[0].shield);
        g.step();
        assert!(g.tanks[0].shield, "step() never ran the pickup check");
        assert!(g.pickups.is_empty());
    }

    /// And it must stay wired off when the feature is off.
    #[test]
    fn stepping_collects_nothing_while_crates_are_off() {
        let mut g = Game::new(21, 2);
        g.setup_battle();
        g.pickups_enabled = false;
        let (x, y) = (g.tanks[0].x, g.tanks[0].y);
        g.pickups.push(Pickup { x, y, weapon: Weapon::Shield });
        g.step();
        assert!(!g.tanks[0].shield);
        assert_eq!(g.pickups.len(), 1);
    }

    #[test]
    fn a_crate_out_of_reach_is_left_alone() {
        let mut g = Game::new(3, 2);
        g.setup_battle();
        g.pickups_enabled = true;
        let (x, y) = (g.tanks[0].x + g.scale, g.tanks[0].y + g.scale);
        g.pickups.push(Pickup { x, y, weapon: Weapon::Gatling });
        collect(&mut g);
        assert_eq!(g.pickups.len(), 1);
        assert_eq!(g.tanks[0].weapon, Weapon::Normal);
    }

    #[test]
    fn a_shield_absorbs_exactly_one_lethal_hit() {
        let mut g = Game::new(5, 2);
        g.setup_battle();
        g.tanks[0].shield = true;
        g.destroy_tank(0);
        assert!(g.tanks[0].alive, "the shield did not absorb the hit");
        assert!(!g.tanks[0].shield, "the shield was not spent");
        assert_eq!(g.alive_count, 2);
        g.destroy_tank(0);
        assert!(!g.tanks[0].alive, "the second hit should land");
        assert_eq!(g.alive_count, 1);
    }

    #[test]
    fn a_shotgun_shot_puts_five_pellets_in_the_air() {
        let mut g = Game::new(9, 2);
        g.setup_battle();
        g.tanks[0].weapon = Weapon::Shotgun;
        g.tanks[0].weapon_charges = C::SHOTGUN_CHARGES;
        g.fire_weapon(0);
        let expected = 1 + 2 * C::SHOTGUN_SIDE_PELLETS as usize;
        assert_eq!(g.bullets.len(), expected);
        // Fanned around the barrel, and every pellet belongs to the shooter.
        let mut rotations: Vec<i64> = g
            .bullets
            .iter()
            .map(|b| {
                assert_eq!(b.owner, 0);
                (b.y_speed.atan2(b.x_speed).to_degrees() * 1000.0).round() as i64
            })
            .collect();
        rotations.sort_unstable();
        rotations.dedup();
        assert_eq!(rotations.len(), expected, "pellets share a heading");
    }

    /// Ammo accounting: a spent magazine falls back to the default gun.
    #[test]
    fn an_empty_magazine_returns_the_default_gun() {
        let mut g = Game::new(13, 2);
        g.setup_battle();
        g.tanks[0].weapon = Weapon::Shotgun;
        g.tanks[0].weapon_charges = 1;
        g.fire_weapon(0);
        assert_eq!(g.tanks[0].weapon, Weapon::Normal);
        assert_eq!(g.tanks[0].weapon_charges, 0);
    }

    #[test]
    fn a_gatling_gets_a_cooldown_and_a_deeper_magazine() {
        let mut g = Game::new(17, 2);
        g.setup_battle();
        g.tanks[0].weapon = Weapon::Gatling;
        g.tanks[0].weapon_charges = C::GATLING_CHARGES;
        assert_eq!(g.in_flight_cap(0), C::GATLING_MAX_IN_FLIGHT);
        assert!(g.weapon_ready(0));
        g.fire_weapon(0);
        assert!(!g.weapon_ready(0), "the trigger should be locked for a moment");
        for _ in 0..C::GATLING_COOLDOWN_FRAMES {
            g.step();
        }
        assert!(g.weapon_ready(0), "the cooldown never expired");
        // And the plain gun keeps the five it always had.
        g.tanks[1].weapon = Weapon::Normal;
        assert_eq!(g.in_flight_cap(1), g.settings_max_bullets);
    }
}
