/**
 * The opponent's per-frame state machine.
 *
 * Play mode's opponent is not simply "the engine": Killfield's MPC has to be
 * paused for the opening delay, and every controller's action can be held back
 * by an actuation delay queue. Both live here so the frame loop stays readable.
 *
 * This file used to also carry the ranked-session recorder, which existed so
 * the browser and the CI verifier drove the opponent from one implementation.
 * The leaderboard is gone from this local build, so only the driver remains.
 */

export const NEUTRAL_ACTION = 8; // stationary, no fire
/**
 * Ranked play is the default match with no handicap granted to the opponent.
 *
 * Both knobs exist to make the game approachable, and both make it easier: the
 * delay holds the opponent's actuation back by whole frames, and the opening
 * pause keeps it still while the round starts. A record is only comparable —
 * and only exercises the configuration the engine is actually tested in — if
 * the delay is zero and the pause is no longer than the default. A shorter
 * pause is allowed because it only makes the run harder.
 */
export const FPS = 25;
export const KILLFIELD_RAYS = 512;
export const OPPONENT_SEAT = 0;
export const HYBRID_OBS_DIM = 1028;
export const HYBRID_BULLET_SLOTS = 10;
export const HYBRID_DODGE_OFFSET = 1018;
export const HYBRID_DODGE_DIM = 9;

/** Convert Discrete(18) into the exact full-strength human input recorded by
 * the browser. This is used by the opt-in policy pilot and mirrors
 * engine/src/score.rs CANDIDATES. */
export function policyActionToInput(action) {
  const throttle = Math.floor(action / 6);
  const turn = Math.floor((action % 6) / 2);
  return {
    forward: throttle === 2 ? 1 : 0,
    backup: throttle === 0 ? 1 : 0,
    turnLeft: turn === 0 ? 1 : 0,
    turnRight: turn === 2 ? 1 : 0,
    fire: action % 2,
  };
}

/** Read one seat's Hybrid observation out of wasm memory. */
export function readObservation(wasm, handle, seat) {
  const ptr = wasm.kf_hybrid_observation(handle, seat);
  const view = new Float32Array(wasm.memory.buffer, ptr, wasm.kf_hybrid_observation_len());
  const mask = new Array(HYBRID_BULLET_SLOTS);
  for (let i = 0; i < HYBRID_BULLET_SLOTS; i += 1) {
    mask[i] = view[HYBRID_OBS_DIM + i] > 0.5;
  }
  return {
    observation: view,
    mask,
    dodge: view.subarray(HYBRID_DODGE_OFFSET, HYBRID_DODGE_OFFSET + HYBRID_DODGE_DIM),
  };
}

/**
 * Drives seat 0 through one ranked session.
 *
 * Laika and Killfield are driven inside `kf_step`, so for those this only
 * manages the opening pause; there is no action to record and nothing a
 * submission could forge. Hybrid is driven from JS, so its action is recorded
 * and re-derived by the verifier.
 */
export class OpponentDriver {
  constructor({ opponent, delayFrames, openingDelayFrames, policy = null }) {
    this.opponent = opponent;
    this.delayFrames = delayFrames;
    this.openingDelayFrames = opponent === "laika" ? 0 : openingDelayFrames;
    this.policy = policy;
    this.queue = [];
    this.pause = this.openingDelayFrames;
  }

  /** Wire the engine-side opponent up on a fresh handle. */
  attach(wasm, handle) {
    if (this.opponent !== "killfield") return;
    // A human is not Laika's script, so the planner gets the honest "assume
    // they hold their current buttons" model.
    wasm.kf_attach_mpc(handle, OPPONENT_SEAT, 7, KILLFIELD_RAYS, 1);
    wasm.kf_set_mpc_delay(handle, OPPONENT_SEAT, this.delayFrames);
    wasm.kf_set_mpc_enabled(handle, OPPONENT_SEAT, this.pause === 0 ? 1 : 0);
  }

  /**
   * Work out seat 0's action for this frame, advancing the delay queue.
   * Returns null when the engine drives the seat itself.
   *
   * Deciding is split from applying so the verifier can drive the engine with
   * the action the submission recorded — keeping the replayed trajectory
   * identical to the one the player saw — while still auditing that action
   * against the decision it derived independently.
   */
  decide(wasm, handle) {
    if (this.opponent !== "hybrid") return null;
    if (this.pause > 0) return { action: NEUTRAL_ACTION, logits: null };

    const { observation, mask, dodge } = readObservation(wasm, handle, OPPONENT_SEAT);
    const logits = this.policy.logits(observation, mask, dodge);
    let best = 0;
    for (let i = 1; i < logits.length; i += 1) if (logits[i] > logits[best]) best = i;
    this.queue.push({ action: best, logits });

    // Until the queue is deep enough the seat actuates nothing, which is what
    // the delay handicap means: it plans every frame but acts late.
    if (this.queue.length > this.delayFrames) return this.queue.shift();
    return { action: NEUTRAL_ACTION, logits: null };
  }

  apply(wasm, handle, action) {
    if (this.opponent !== "hybrid") return;
    wasm.kf_set_hybrid_action(handle, OPPONENT_SEAT, action);
  }

  /** Advance the opening pause and reset on a round boundary, after `kf_step`. */
  afterStep(wasm, handle, flags) {
    const newRound = (flags & 1) !== 0;
    if (newRound) {
      this.queue.length = 0;
      this.pause = this.openingDelayFrames;
      if (this.opponent === "killfield") {
        wasm.kf_set_mpc_enabled(handle, OPPONENT_SEAT, this.pause === 0 ? 1 : 0);
      }
      return;
    }
    if (this.pause > 0) {
      this.pause -= 1;
      if (this.pause === 0 && this.opponent === "killfield") {
        wasm.kf_set_mpc_enabled(handle, OPPONENT_SEAT, 1);
      }
    }
  }
}
