import assert from "node:assert/strict";
import {
  AIM_MODE_AIM, AIM_MODE_DRIVE, MouseAim, aimButtons, aimDrive, aimHeading,
} from "../src/mouse-aim.js";

// ------------------------------------------------------------------ heading

// Rotation 0 points up and grows clockwise, matching constants.js and the
// wheel's atan2(x, -y).
assert.equal(aimHeading(0, -100), 0);
assert.equal(aimHeading(100, 0), 90);
assert.equal(aimHeading(0, 100), 180);
assert.equal(aimHeading(-100, 0), -90);
assert.equal(aimHeading(100, -100), 45);

// A cursor resting on the hull carries no heading, so it must not be read as
// "due north" — that would spin the tank whenever the mouse stopped on it.
assert.equal(aimHeading(2, -2), null);
assert.equal(aimHeading(NaN, 0), null);

// Headings quantise onto the wheel's 128-direction lattice, not to raw floats.
assert.equal(aimHeading(1, -100) % (360 / 128), 0);

// ------------------------------------------------------------------ buttons

// Throttle comes from the keyboard and never from how far away the cursor is:
// that decoupling is the whole reason this is not joystickButtons.
const far = aimButtons(0, -4000, 0, { forward: 0.4, backup: 0 });
const near = aimButtons(0, -40, 0, { forward: 0.4, backup: 0 });
assert.equal(far.forward, near.forward);
assert.equal(far.forward, 0.4);

// Hull up, cursor east: turn right until aligned, then stop correcting.
const turning = aimButtons(100, 0, 0, { forward: 0, backup: 0 });
assert.equal(turning.turnRight, 1);
assert.equal(turning.turnLeft, 0);
assert.equal(turning.targetRotation, 90);
const aligned = aimButtons(100, 0, 90, { forward: 0, backup: 0 });
assert.equal(aligned.turnRight, 0);
assert.equal(aligned.turnLeft, 0);

// Hull up, cursor west: the short way round is left, not 270 degrees right.
assert.equal(aimButtons(-100, 0, 0, { forward: 0, backup: 0 }).turnLeft, 1);

// Reverse stays available: the cursor owns the nose, S still backs along it.
const reversing = aimButtons(0, -100, 0, { forward: 0, backup: 0.7 });
assert.equal(reversing.backup, 0.7);
assert.equal(reversing.forward, 0);

// With no heading to hold, the keyboard's own turn keys come back rather than
// the hull freezing.
const fallback = aimButtons(0, 0, 0, { forward: 0, backup: 0, turnLeft: 1, turnRight: 0 });
assert.equal(fallback.turnLeft, 1);
assert.equal(fallback.targetRotation, null);

// Strengths are clamped, so a malformed sample cannot drive the engine out of
// its [0, 1] contract.
const clamped = aimButtons(0, -100, 0, { forward: 4, backup: -2 });
assert.equal(clamped.forward, 1);
assert.equal(clamped.backup, 0);

// ------------------------------------------------------------------ pointer

class FakeTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) {
    const existing = this.listeners.get(type) ?? [];
    existing.push(listener);
    this.listeners.set(type, existing);
  }
  dispatch(type, event = {}) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener({ pointerType: "mouse", button: 0, preventDefault() {}, ...event });
    }
  }
}

const surface = new FakeTarget();
// A classList that actually records, so the cursor-affordance classes can be
// asserted rather than silently swallowed.
surface.classes = new Set();
surface.classList = {
  toggle(name, on) {
    if (on) surface.classes.add(name); else surface.classes.delete(name);
  },
};
const fakeWindow = new FakeTarget();
const aim = new MouseAim(surface, fakeWindow);

const fireEdges = [];
aim.onFireChange = (pressed) => fireEdges.push(pressed);

// While off, the canvas is inert: no cursor is tracked and no trigger fires.
surface.dispatch("pointermove", { clientX: 10, clientY: 20 });
surface.dispatch("pointerdown", { clientX: 10, clientY: 20 });
assert.equal(aim.client, null);
assert.equal(aim.active(), false);
assert.deepEqual(fireEdges, []);

aim.setMode("aim");
surface.dispatch("pointermove", { clientX: 10, clientY: 20 });
assert.deepEqual(aim.client, { x: 10, y: 20 });
assert.equal(aim.active(), true);

// Touch keeps going to the wheel; the two paths must not fight over one hull.
surface.dispatch("pointermove", { pointerType: "touch", clientX: 999, clientY: 999 });
assert.deepEqual(aim.client, { x: 10, y: 20 });

// Press and release are edges, reported once each.
surface.dispatch("pointerdown", { clientX: 30, clientY: 40 });
assert.equal(aim.firePressed, true);
assert.deepEqual(aim.client, { x: 30, y: 40 });
surface.dispatch("pointerdown", { clientX: 30, clientY: 40 });
assert.deepEqual(fireEdges, [true]);
fakeWindow.dispatch("pointerup");
assert.equal(aim.firePressed, false);
assert.deepEqual(fireEdges, [true, false]);

// Only the left button fires.
surface.dispatch("pointerdown", { button: 2, clientX: 50, clientY: 60 });
assert.equal(aim.firePressed, false);

// A drag that ends off-canvas, or a window that loses focus, must not latch
// the trigger — the same failure a lost keyup would cause.
surface.dispatch("pointerdown", { clientX: 30, clientY: 40 });
assert.equal(aim.firePressed, true);
fakeWindow.dispatch("blur");
assert.equal(aim.firePressed, false);

// Leaving the canvas drops the heading but leaves the mode on.
surface.dispatch("pointerleave", {});
assert.equal(aim.client, null);
assert.equal(aim.active(), false);
assert.equal(aim.enabled, true);

// ------------------------------------------------------------------ applyTo

const calls = [];
const fakeWasm = {
  clear: true,
  kf_set_input(...args) { calls.push(["input", ...args]); },
  kf_set_rotation_if_clear(handle, tank, rotation) {
    calls.push(["rotate", handle, tank, rotation]);
    return this.clear;
  },
};

aim.setMode("aim");
surface.dispatch("pointermove", { clientX: 0, clientY: 0 });
surface.dispatch("pointerdown", { clientX: 0, clientY: 0 });

// Instant turn: the hull snaps and the turn inputs are dropped, because the
// engine has already placed it.
calls.length = 0;
let applied = aim.applyTo(fakeWasm, 7, 0, { forward: 0.5, backup: 0, fire: 0 },
  { x: 100, y: 0 }, 0, true);
assert.equal(applied.snappedRotation, 90);
assert.equal(applied.input.turnRight, 0);
assert.equal(applied.input.fire, 1, "a held mouse button fires");
assert.deepEqual(calls[0], ["rotate", 7, 0, 90]);
assert.deepEqual(calls[1], ["input", 7, 0, 0.5, 0, 0, 0, 1, 1]);

// Refused because the pose would clip a wall: fall back to turning at the
// normal rate rather than teleporting through it.
fakeWasm.clear = false;
applied = aim.applyTo(fakeWasm, 7, 0, { forward: 0, backup: 0, fire: 0 },
  { x: 100, y: 0 }, 0, true);
assert.equal(applied.snappedRotation, null);
assert.equal(applied.input.turnRight, 1);

// Instant turn off: never asks the engine to place the hull at all.
calls.length = 0;
applied = aim.applyTo(fakeWasm, 7, 0, { forward: 0, backup: 0, fire: 0 },
  { x: 100, y: 0 }, 0, false);
assert.equal(calls.filter((c) => c[0] === "rotate").length, 0);
assert.equal(applied.input.turnRight, 1);

// The keyboard's own fire key still works while the mouse button is up.
fakeWindow.dispatch("pointerup");
applied = aim.applyTo(fakeWasm, 7, 0, { forward: 0, backup: 0, fire: 1 },
  { x: 100, y: 0 }, 0, false);
assert.equal(applied.input.fire, 1);
applied = aim.applyTo(fakeWasm, 7, 0, { forward: 0, backup: 0, fire: 0 },
  { x: 100, y: 0 }, 0, false);
assert.equal(applied.input.fire, 0);

// Turning the mode off releases a held trigger instead of stranding it.
surface.dispatch("pointerdown", { clientX: 0, clientY: 0 });
assert.equal(aim.firePressed, true);
aim.setMode("off");
assert.equal(aim.firePressed, false);
assert.equal(aim.client, null);

// Drive mode tracks the cursor exactly as aim mode does, and hides the system
// pointer so the reticle is the only thing marking it.
aim.setMode("drive");
assert.equal(aim.enabled, true);
assert.equal(surface.classes.has("aiming"), true);
assert.equal(surface.classes.has("driving"), true);
aim.setMode("aim");
assert.equal(surface.classes.has("driving"), false, "aim mode keeps the arrow");
aim.setMode("off");
assert.equal(surface.classes.has("aiming"), false);

// ------------------------------------------------------------- drive mode

const SCALE = 40;

// Throttle comes from distance in cells, so the feel survives a maze resize.
assert.equal(aimDrive(0, 0, SCALE), 0);
assert.equal(aimDrive(SCALE * 0.2, 0, SCALE), 0, "cursor on the hull must not creep");
assert.equal(aimDrive(SCALE * 0.35, 0, SCALE), 0, "the dead zone ends at 0.35 cells");
assert.equal(aimDrive(SCALE, 0, SCALE), 1, "a cell out is full speed");
assert.equal(aimDrive(SCALE * 4, 0, SCALE), 1, "and it stays clamped");
assert.ok(aimDrive(SCALE * 0.6, 0, SCALE) > 0 && aimDrive(SCALE * 0.6, 0, SCALE) < 1);
// Half a maze away on a big board must still read as full speed, not overflow.
assert.equal(aimDrive(SCALE * 0.6, 0, SCALE), aimDrive(SCALE * 1.2, 0, SCALE * 2),
  "the ramp is measured in cells, not pixels");
assert.equal(aimDrive(10, 10, 0), 0, "a missing scale cannot produce throttle");
assert.equal(aimDrive(NaN, 0, SCALE), 0);

// Drive mode ignores the keyboard's throttle rather than adding to it.
const held = { forward: 1, backup: 1, turnLeft: 1, turnRight: 1, fire: 0 };
const driving = aimButtons(0, -SCALE * 2, 0, held, AIM_MODE_DRIVE, SCALE);
assert.equal(driving.forward, 1);
assert.equal(driving.backup, 0, "reverse has no meaning when the hull turns to face");
assert.equal(driving.turnLeft, 0, "already facing the cursor");
assert.equal(driving.turnRight, 0);

// Aim mode is unchanged: the keyboard still owns the throttle.
const aiming = aimButtons(0, -SCALE * 2, 0, held, AIM_MODE_AIM, SCALE);
assert.equal(aiming.forward, 1);
assert.equal(aiming.backup, 1);

// A cursor behind the hull turns the tank around instead of reversing.
const behind = aimButtons(0, SCALE * 2, 0, held, AIM_MODE_DRIVE, SCALE);
assert.equal(behind.backup, 0);
assert.equal(behind.forward, 1);
assert.ok(behind.turnLeft === 1 || behind.turnRight === 1, "it must turn to face");
assert.equal(behind.targetRotation, 180);

// Cursor resting on the hull: no heading, no throttle, and in drive mode the
// keyboard's turn keys must not leak back in.
const parked = aimButtons(1, -1, 0, held, AIM_MODE_DRIVE, SCALE);
assert.equal(parked.forward, 0, "parking the cursor on the hull is how you stop");
assert.equal(parked.turnLeft, 0);
assert.equal(parked.turnRight, 0);
assert.equal(parked.targetRotation, null);
