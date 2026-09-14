//! Port of `killfield/src/constants.js`.
//!
//! The simulation is a fixed 25 FPS maze tank duel with ricocheting bullets.
//! Every physical quantity is expressed at a reference cell size of
//! SCALE = 50 and multiplied by (scale / 50) at runtime, because the cell
//! size is re-derived from the maze dimensions every round.

// ---- frame rate ----
pub const FPS: i32 = 25;

// ---- playfield layout ----
pub const MOVIEWIDTH: f64 = 692.0;
pub const MOVIEHEIGHT: f64 = 480.0;
pub const HEIGHTTOBOTTOM: f64 = 80.0;

// ---- bullets ----
pub const BULLETSPEED: f64 = 4.5; // px/frame at SCALE=50
pub const BULLETLIFETIME: i32 = 250; // frames (10 s)
pub const BULLETHITCHECKINTERVALS: i32 = 7; // substeps per frame
pub const BULLETDEADLY: i32 = 0;

// Referenced by the AI's dodge logic even though these weapons never spawn
// in duel mode (the active-weapon list is empty).
pub const FRAGSPEED: f64 = 4.5;
pub const GATLINGSPEED: f64 = 5.5;

// ---- crates (never spawn in duel mode, but the timer still consumes RNG) ----
pub const CRATESPAWNTIMEBASE: f64 = 350.0;
pub const CRATESPAWNTIMERANDOM: i32 = 200;
pub const CRATESPAWNMAZESIZESCALE: f64 = 2000.0;

// ---- weapon crates (see pickups.rs) ----
/// The crates run on their own clock, NOT on `CRATESPAWNTIMEBASE`. Two
/// reasons. The original's 350-frame base was tuned for human-vs-human rounds;
/// against these agents a round is over in about 125 frames, so a crate on the
/// old cadence would never once reach the floor. And the old constants still
/// drive the vestigial `Game::crate_timer`, whose reset *rate* decides how
/// often it draws from the game RNG — retuning them would shift the very
/// stream `pickups::tests::enabling_crates_does_not_disturb_the_game_rng`
/// exists to protect.
pub const PICKUP_SPAWN_TIMEBASE: f64 = 50.0;
pub const PICKUP_SPAWN_TIMERANDOM: i32 = 50;
/// Bigger mazes wait a little longer, because there is more floor to cross.
pub const PICKUP_SPAWN_MAZESIZESCALE: f64 = 600.0;

/// Crates waiting to be collected at once. The original let the floor fill up;
/// two keeps a duel about the fight rather than about shopping.
pub const CRATE_MAX_ON_FIELD: usize = 2;
/// Random cells probed for a free spot before giving up for this cycle.
pub const CRATE_SPAWN_ATTEMPTS: i32 = 24;
/// Cells of clearance from any live tank, so a crate never lands in your lap.
pub const CRATE_SPAWN_CLEARANCE_CELLS: f64 = 1.25;
/// How close the hull centre must be to collect. Under half a cell, so you
/// have to drive onto it rather than past it.
pub const CRATE_PICKUP_RADIUS_CELLS: f64 = 0.40;

/// Gatling: shots per pickup, frames between them, and its own in-flight cap.
/// The default five would make it barely distinguishable from the plain gun.
pub const GATLING_CHARGES: i32 = 24;
pub const GATLING_COOLDOWN_FRAMES: i32 = 2;
pub const GATLING_MAX_IN_FLIGHT: i32 = 12;

/// The laser is a projectile, not a hitscan.
///
/// It began as an instant beam and that was a mistake. Every defensive
/// mechanism this engine has assumes a threat that exists in the world for
/// some number of frames and can therefore be seen coming: the observation's
/// ten bullet slots, `THREAT_OFFSET`'s urgency, `score::dodge_safety`, and
/// `risk::incoming_risk`, which flies each round forward seventy-five frames
/// to decide whether it is a danger. All 113 of those channels read "clear"
/// against a weapon that resolves inside the frame it is fired. The policy was
/// not failing to dodge it; there was nothing there to dodge, and measured
/// against a competent carrier it died in 88% of rounds.
///
/// Every weapon the original port inherited is a projectile with a speed —
/// `BULLETSPEED`, `FRAGSPEED`, `GATLINGSPEED`. A bolt travelling five times a
/// bullet's speed is still overwhelmingly the best thing in a crate, and it
/// costs nothing to defend against that the engine did not already have.
pub const LASER_SPEED_MULTIPLIER: f64 = 25.0;

/// Laser: shots per pickup, range in cells, how finely the aiming preview
/// is marched, and how long the afterglow is drawn for. The cooldown is long
/// on purpose — an instant, undodgeable hit has to be rationed.
pub const LASER_CHARGES: i32 = 3;
pub const LASER_COOLDOWN_FRAMES: i32 = 12;
pub const LASER_RANGE_CELLS: f64 = 14.0;
/// March resolution. A hull is about a third of a cell across, so sixteen
/// steps per cell puts five or so tests inside one — enough that the beam
/// cannot tunnel through a tank.
pub const LASER_STEPS_PER_CELL: i32 = 16;
/// How long the beam stays on screen. It resolves in a single frame, so this
/// is purely so a human can read where it went: eight frames was a flash you
/// could not trace across three bounces. The first `HOLD` frames draw at full
/// strength, the rest fade out. `laser::tick` is gated on `!frozen`, so a beam
/// that ended the round also hangs in the air through the freeze.
pub const LASER_BEAM_FRAMES: i32 = 45;
pub const LASER_BEAM_HOLD_FRAMES: i32 = 10;

/// Shotgun: trigger pulls per pickup, pellets either side of the centre one,
/// and the angle between neighbouring pellets.
pub const SHOTGUN_CHARGES: i32 = 4;
pub const SHOTGUN_SIDE_PELLETS: i32 = 2;
pub const SHOTGUN_SPREAD_DEG: f64 = 8.0;

// ---- round lifecycle ----
pub const NUMBEROFFRAMESBEFOREEND: i32 = 125; // world keeps running after a kill
pub const NUMBEROFFRAMESFROZEN: i32 = 50; // freeze + score at this endCount
pub const NUMBEROFFRAMESBEFORERESET: i32 = 5;

/// Residual-bullet settlement window: 125 - 50 = 75 frames (3 s) in which the
/// apparent winner can still be killed by a bullet already in the air.
pub const SETTLEMENT_FRAMES: i32 = NUMBEROFFRAMESBEFOREEND - NUMBEROFFRAMESFROZEN;

// ---- visual effects ----
pub const MAXSHAKE: f64 = 8.0;

// ---- pathfinding ----
pub const MAXDEADENDPENALTY: f64 = 5.0;

// ---- settings ----
pub const SETTINGS_MAX_BULLETS: usize = 5;
pub const SETTINGS_MAX_CRATES: usize = 3;
pub const SETTINGS_CRATE_SPAWN_MODIFIER: f64 = 1.0;

// ---- tank physics ----
pub const TANK_FORWARD_SPEED_BASE: f64 = 4.0; // x (scale/50) px/frame
pub const TANK_BACKUP_SPEED_BASE: f64 = 2.5; // x (scale/50) px/frame
pub const TANK_TURN_SPEED: f64 = 10.0; // deg/frame
pub const TANK_MOVE_STEPS: i32 = 5; // substeps per frame

// Wall contact is resolved at 5-substep precision. The blocked normal is
// removed and the tangent is retained with more drag at steeper incidence.
pub const TANK_WALL_SLIDE_MIN_RETENTION: f64 = 0.70;
pub const TANK_WALL_SLIDE_MAX_RETENTION: f64 = 0.96;
pub const TANK_WALL_SLIDE_INCIDENCE_DRAG: f64 = 0.30;
pub const TANK_WALL_ALIGN_SPEED: f64 = 2.0; // max contact-induced deg/frame

// A turn beside a wall can put one probe a fraction of a pixel inside the
// stroke even though shifting the hull slightly outward would make it valid.
pub const TANK_WALL_SEPARATION_BASE: f64 = 1.0; // px at reference scale
pub const TANK_WALL_SEPARATION_STEPS: i32 = 5;

// ---- tank geometry, in local sprite units ----
// Rotation 0 points UP (-y). The barrel fires along (rotation - 90) deg.
pub const TANK_BASE_WIDTH: f64 = 61.0;
pub const TANK_BASE_HEIGHT: f64 = 81.0;
pub const TANK_TURRET_WIDTH: f64 = 45.0;
pub const TANK_TURRET_HEIGHT: f64 = 77.5;
pub const TANK_DISPLAY_SCALE_FACTOR: f64 = 0.55 / 100.0; // x scale

/// Union bounds of the whole tank, for the cheap bounding-box pre-test.
pub const TANK_BOUNDS_LOCAL: [f64; 4] = [-30.5, -55.0, 30.5, 40.5];

// Wall collision probe points at the barrel tip.
pub const TANK_BARREL_HALF_WIDTH: f64 = TANK_TURRET_WIDTH / 6.0; // 7.5
pub const TANK_BARREL_TIP_Y: f64 = (-TANK_TURRET_HEIGHT / 16.0) * 11.0; // -53.28125

// Bullet-vs-tank hit shape: base rectangle union barrel rectangle.
// The turret dome is entirely inside the base rectangle, so it adds nothing.
pub const TANK_SHAPE_BARREL_HALF_WIDTH: f64 = 8.5;
pub const TANK_SHAPE_BARREL_TIP_Y: f64 = -55.0;

// Render-only. Bullets are dimensionless points to the hit test.
pub const BULLET_VISUAL_RADIUS: f64 = 3.5;

pub const DEG: f64 = std::f64::consts::PI / 180.0;
