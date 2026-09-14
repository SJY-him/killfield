/**
 * Desktop mouse aiming for the human seat.
 *
 * The barrel is welded to the hull — `constants.js` puts rotation 0 up and
 * fires along `rotation - 90` — so aiming *is* steering. `input.js` already
 * solves absolute-heading control for the touch wheel, but `joystickButtons`
 * couples heading to throttle: on a thumbstick, how far you push has to mean
 * both. Whether a mouse should inherit that coupling is a matter of taste, so
 * both are offered. Either way the wheel's 128-direction lattice and its
 * half-turn-rate alignment deadband are reused, so every human path quantises
 * identically.
 *
 *   AIM_MODE_AIM    the cursor owns the heading, the keyboard owns the
 *                   throttle. Pointing at something far away does not commit
 *                   you to driving into it.
 *   AIM_MODE_DRIVE  the cursor owns both. Distance from the hull is the
 *                   throttle, exactly as radius is on the wheel, so the tank
 *                   goes where you point and parking the cursor on the hull
 *                   is how you stop. No keyboard needed.
 *
 * Same wasm contract as every other human path:
 *
 *   wasm.kf_set_input(handle, tank, forward, backup, turnLeft, turnRight, fire, 1)
 *
 * with continuous=1, so the hull turns by a fraction of the ten-degree lattice
 * rather than snapping onto it — a discrete controller would pass 1.0.
 *
 * When the cursor leaves the canvas there is no heading to hold, so steering
 * falls back to the keyboard's own turn keys rather than freezing the hull.
 */

import * as C from "./constants.js";

/** The wheel's lattice, reused so both human paths quantise identically. */
const AIM_DIRECTIONS = 128;
const AIM_STEP_DEG = 360 / AIM_DIRECTIONS;
/** Matches the wheel: half a frame's turn is close enough to stop correcting. */
const AIM_DEADBAND_DEG = C.TANK_TURN_SPEED / 2;
/** Inside this radius the cursor is on the hull and carries no direction. */
const MIN_AIM_RADIUS = 6;

export const AIM_MODE_OFF = "off";
export const AIM_MODE_AIM = "aim";
export const AIM_MODE_DRIVE = "drive";

/**
 * Drive mode's throttle ramp, in cells. Nothing until the cursor is clear of
 * the hull, full speed a cell out. The dead zone has to comfortably exceed the
 * hull's own half-width (about 0.17 cells) or the tank creeps while the cursor
 * rests on it; a cell of ramp leaves room for a deliberate slow approach.
 */
const DRIVE_START_CELLS = 0.35;
const DRIVE_FULL_CELLS = 1.0;

/**
 * Throttle from how far the cursor sits from the hull. `scale` is the round's
 * cell size, so the feel does not change with maze size.
 */
export function aimDrive(vx, vy, scale) {
  if (!Number.isFinite(vx) || !Number.isFinite(vy) || !(scale > 0)) return 0;
  const cells = Math.hypot(vx, vy) / scale;
  return clamp01((cells - DRIVE_START_CELLS) / (DRIVE_FULL_CELLS - DRIVE_START_CELLS));
}

function clamp01(value) {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

export function normaliseAngle(degrees) {
  let value = degrees % 360;
  if (value > 180) value -= 360;
  else if (value <= -180) value += 360;
  return value;
}

/**
 * World-space offset from hull to cursor, as a heading on the lattice.
 * `null` when the cursor is too close to the hull to mean anything.
 */
export function aimHeading(vx, vy) {
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return null;
  if (Math.hypot(vx, vy) < MIN_AIM_RADIUS) return null;
  // atan2(x, -y): screen y grows downward, rotation 0 points up.
  const raw = Math.atan2(vx, -vy) / C.DEG;
  return normaliseAngle(Math.round(raw / AIM_STEP_DEG) * AIM_STEP_DEG);
}

/**
 * Combine a cursor offset with the keyboard's drive strengths.
 *
 * `drive` is a `sampleWindowStrengths()` result, so forward/backup arrive
 * already time-weighted across the physics frame. Turn strength is full or
 * nothing: unlike the wheel there is no radius to ramp it over, and the
 * deadband is what prevents lattice oscillation.
 */
export function aimButtons(vx, vy, currentRotation, drive, mode = AIM_MODE_AIM, scale = 0) {
  const driving = mode === AIM_MODE_DRIVE;
  // In drive mode the cursor is the whole stick, so the keyboard's throttle is
  // ignored rather than added to it — holding a key must not double the speed.
  const forward = driving ? aimDrive(vx, vy, scale) : clamp01(drive?.forward);
  // Reverse has no meaning when the tank always turns to face the cursor.
  const backup = driving ? 0 : clamp01(drive?.backup);
  const heading = aimHeading(vx, vy);
  if (heading === null) {
    return {
      forward,
      backup,
      // With the cursor on the hull there is nothing to steer toward. The
      // keyboard's turn keys still work in aim mode; in drive mode the mouse
      // owns steering outright, so the hull simply holds its heading.
      turnLeft: driving ? 0 : clamp01(drive?.turnLeft),
      turnRight: driving ? 0 : clamp01(drive?.turnRight),
      targetRotation: null,
    };
  }
  const delta = normaliseAngle(heading - currentRotation);
  const turn = Math.abs(delta) > AIM_DEADBAND_DEG ? 1 : 0;
  return {
    forward,
    backup,
    turnLeft: delta < 0 ? turn : 0,
    turnRight: delta > 0 ? turn : 0,
    targetRotation: heading,
  };
}

/**
 * Cursor tracking over the canvas. Holds client coordinates only: turning
 * those into world space needs the round's maze footprint, which lives in the
 * render buffer, so the viewer does that conversion and passes the offset in.
 */
export class MouseAim {
  constructor(surface, windowTarget = globalThis) {
    this.surface = surface;
    this.enabled = false;
    this.mode = AIM_MODE_OFF;
    /** Client coordinates, or null whenever the cursor is not over the canvas. */
    this.client = null;
    this.firePressed = false;
    this.onFireChange = null;

    const track = (event) => {
      if (!this.enabled || event.pointerType === "touch") return;
      this.client = { x: event.clientX, y: event.clientY };
    };
    surface.addEventListener("pointermove", track);
    // Chromium coalesces pointermove to the display rate; the raw stream gives
    // the freshest heading available for the 25Hz sample and the prediction.
    surface.addEventListener("pointerrawupdate", track);
    surface.addEventListener("pointerenter", track);
    surface.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "touch") return;
      this.client = null;
    });

    surface.addEventListener("pointerdown", (event) => {
      if (!this.enabled || event.pointerType === "touch" || event.button !== 0) return;
      event.preventDefault();
      this.client = { x: event.clientX, y: event.clientY };
      this.setFire(true);
    });
    // Release on the window: a drag that ends off-canvas must not latch the
    // trigger, which is the same failure a lost keyup would cause.
    const release = () => this.setFire(false);
    windowTarget.addEventListener("pointerup", release);
    windowTarget.addEventListener("pointercancel", release);
    windowTarget.addEventListener("blur", release);
    // Only the left button fires; suppress the menu so a stray right-click
    // cannot swallow the pointerup that follows it.
    surface.addEventListener("contextmenu", (event) => {
      if (this.enabled) event.preventDefault();
    });
  }

  /** @param {"off"|"aim"|"drive"} mode */
  setMode(mode) {
    this.mode = mode;
    this.enabled = mode !== AIM_MODE_OFF;
    if (!this.enabled) {
      this.client = null;
      this.setFire(false);
    }
    this.surface.classList.toggle("aiming", this.enabled);
    // Drive mode hides the system cursor: the reticle already shows where the
    // tank is heading, and two pointers on one canvas read as a bug.
    this.surface.classList.toggle("driving", mode === AIM_MODE_DRIVE);
  }

  setFire(pressed) {
    if (pressed === this.firePressed) return;
    this.firePressed = pressed;
    if (this.onFireChange) this.onFireChange(pressed);
  }

  /** True only when there is a cursor on the canvas to take a heading from. */
  active() {
    return this.enabled && this.client !== null;
  }

  clear() {
    this.client = null;
    this.setFire(false);
  }

  /**
   * @param {{x:number,y:number}|null} aim hull-to-cursor offset, world units
   * @returns {{snappedRotation:number|null, input:object}} mirrors
   *   `TouchControls.applyTo` so the viewer can branch on one shape.
   */
  applyTo(wasm, handle, tank, keyboardStrengths, aim, rotation, instantTurn = false, scale = 0) {
    const movement = aimButtons(
      aim?.x ?? 0, aim?.y ?? 0, rotation, keyboardStrengths, this.mode, scale,
    );
    let snappedRotation = null;
    // The engine refuses a pose that would clip a wall, which is what keeps an
    // instant heading from teleporting the hull through one. Refused, we fall
    // through to turning at the normal rate.
    if (instantTurn && movement.targetRotation !== null
        && wasm.kf_set_rotation_if_clear(handle, tank, movement.targetRotation)) {
      movement.turnLeft = 0;
      movement.turnRight = 0;
      snappedRotation = movement.targetRotation;
    }
    const input = {
      forward: movement.forward,
      backup: movement.backup,
      turnLeft: movement.turnLeft,
      turnRight: movement.turnRight,
      fire: (keyboardStrengths.fire > 0 || this.firePressed) ? 1 : 0,
    };
    wasm.kf_set_input(handle, tank, input.forward, input.backup,
      input.turnLeft, input.turnRight, input.fire, 1);
    return { snappedRotation, input };
  }
}
