import assert from "node:assert/strict";
import {
  NEUTRAL_ACTION, NEUTRAL_ACTION_FIRING, STILLNESS_LIMIT_FRAMES, StillnessGuard,
} from "../src/opponent.js";

// The scores a stalled seat actually produces: the neutral action leads even
// after `idle_logit_penalty` has been applied in full. Measured on the
// deployed checkpoint — the policy's own brake is not broken, it is outvoted.
const STALLED = new Float32Array(18).fill(0);
STALLED[NEUTRAL_ACTION] = 13.9;
STALLED[NEUTRAL_ACTION_FIRING] = 9.0;
STALLED[2] = 8.1;   // the best action that actually moves
STALLED[14] = 7.4;

const guard = new StillnessGuard();

// Up to the limit the guard is invisible: the policy gets what it asked for.
for (let frame = 0; frame < STILLNESS_LIMIT_FRAMES; frame += 1) {
  assert.equal(guard.choose(NEUTRAL_ACTION, STALLED), NEUTRAL_ACTION,
    `overrode on frame ${frame}, before the limit`);
}
// One frame past it, the seat is made to move — to its own next preference,
// not to something arbitrary.
assert.equal(guard.choose(NEUTRAL_ACTION, STALLED), 2);
// And the count restarts, so it does not then override every frame.
assert.equal(guard.choose(NEUTRAL_ACTION, STALLED), NEUTRAL_ACTION);

// Firing while stationary is still stationary: the idle streak the policy sees
// ignores the trigger, and so must this.
const firing = new StillnessGuard(3);
assert.equal(firing.choose(NEUTRAL_ACTION, STALLED), NEUTRAL_ACTION);
assert.equal(firing.choose(NEUTRAL_ACTION_FIRING, STALLED), NEUTRAL_ACTION_FIRING);
assert.equal(firing.choose(NEUTRAL_ACTION, STALLED), NEUTRAL_ACTION);
assert.equal(firing.choose(NEUTRAL_ACTION_FIRING, STALLED), 2, "mixed neutrals must still count");

// Any move at all clears it, so ordinary play never approaches the limit.
const moving = new StillnessGuard(3);
for (let i = 0; i < 50; i += 1) {
  assert.equal(moving.choose(NEUTRAL_ACTION, STALLED), NEUTRAL_ACTION);
  assert.equal(moving.choose(4, STALLED), 4);
}

// A new round is a new arena, not a continued stall.
const rounds = new StillnessGuard(3);
rounds.choose(NEUTRAL_ACTION, STALLED);
rounds.choose(NEUTRAL_ACTION, STALLED);
rounds.reset();
for (let i = 0; i < 3; i += 1) {
  assert.equal(rounds.choose(NEUTRAL_ACTION, STALLED), NEUTRAL_ACTION);
}

// Without scores there is nothing to fall back to, so it must not invent one.
const blind = new StillnessGuard(1);
blind.choose(NEUTRAL_ACTION, null);
assert.equal(blind.choose(NEUTRAL_ACTION, null), NEUTRAL_ACTION);

// A degenerate row where only the neutral actions score at all: the guard
// still has to hand back something legal rather than -1.
const onlyNeutral = new Float32Array(18).fill(-Infinity);
onlyNeutral[NEUTRAL_ACTION] = 1;
onlyNeutral[NEUTRAL_ACTION_FIRING] = 0;
const degenerate = new StillnessGuard(1);
degenerate.choose(NEUTRAL_ACTION, onlyNeutral);
const escape = degenerate.choose(NEUTRAL_ACTION, onlyNeutral);
assert.ok(escape >= 0 && escape < 18, `escape action out of range: ${escape}`);
assert.ok(escape !== NEUTRAL_ACTION && escape !== NEUTRAL_ACTION_FIRING,
  "the escape must actually move");

console.log("StillnessGuard: all checks passed");
