/**
 * Browser front end for the Rust/WASM engine.
 *
 * Ported from killfield/src/main.js + render.js. There is no game logic here:
 * the wasm module is the same crate the trainer links, so what you watch is
 * byte-for-byte what training sees. This file pushes input into wasm, reads
 * the flat f32 render buffer straight out of wasm memory, draws it, and wires
 * the surrounding page (mode switches, sound, fullscreen, i18n).
 *
 * The fixed-timestep loop, i18n strings, audio and input handling are ported
 * close to verbatim from killfield; only the points where killfield talked to
 * a JS `Game`/tank object now cross the wasm FFI instead (see the doc
 * comments on src/input.js and the tuning-push helper below).
 *
 * Two modes only:
 *   - Watch: either seat is Laika, Hybrid, or Killfield (the MPC planner),
 *     picked independently per seat. Laika and Killfield are driven inside
 *     `kf_step` with no per-frame JS involvement; a Hybrid seat is driven
 *     from here — see `driveHybridSeats()` — by running the exported policy
 *     (src/hybrid.js) against the observation `kf_hybrid_observation` builds,
 *     then handing the chosen action back with `kf_set_hybrid_action`.
 *   - Play: you against Killfield, fixed. The four controls here are the only
 *     tuning surface this page exposes; Killfield's own search parameters and
 *     ray count (512, always) are not user-facing.
 */

import * as C from "./src/constants.js?v=a85417b0";
import { STRINGS, loadLang, saveLang } from "./src/i18n.js?v=a85417b0";
import { Keyboard, TouchControls } from "./src/input.js?v=a85417b0";
import { AIM_MODE_AIM, AIM_MODE_DRIVE, AIM_MODE_OFF, MouseAim } from "./src/mouse-aim.js?v=a85417b0";
import { SoundEffects } from "./src/audio.js?v=a85417b0";
import { Rng } from "./src/rng.js?v=a85417b0";
import { interpolatePredictedPose, simulationBudget } from "./src/low-latency.js?v=a85417b0";
import { HybridPolicy } from "./src/hybrid.js?v=a85417b0";
/** Play mode always seats the human in tank 1. */
const HUMAN_SEAT = 1;
// engine/src/duel_obs.rs: the Hybrid observation is schema 24, 1028 semantic
// floats then 10 bullet-mask floats.
import {
  HYBRID_BULLET_SLOTS,
  HYBRID_OBS_DIM,
  HYBRID_OBS_SCHEMA,
  KILLFIELD_RAYS,
  OpponentDriver,
  policyActionToInput,
  readObservation,
} from "./src/opponent.js?v=a85417b0";

const STEP_MS = 1000 / C.FPS; // 40 ms
const MAX_CATCHUP_MS = 250;
const STREAK_STORAGE_KEY = "killfield-streak";
const INSTANT_TURN_STORAGE_KEY = "killfield-human-instant-turn-v2";
const MOUSE_AIM_STORAGE_KEY = "killfield-human-mouse-mode-v2";
/** Cycle order for the one button that carries all three states. */
const MOUSE_MODES = [AIM_MODE_OFF, AIM_MODE_AIM, AIM_MODE_DRIVE];
const PICKUPS_STORAGE_KEY = "killfield-weapon-crates";
const OPENING_DELAY_STORAGE_KEY = "killfield-opening-delay-seconds";
const REACTION_DELAY_STORAGE_KEY = "killfield-reaction-delay-frames";
const DEFAULT_OPENING_DELAY_SECONDS = 0.5;
// Owner/testing aid: the public policy can drive the human input path without
// bypassing recording or verification. It is opt-in and has no visible toggle.
const QUERY = new URLSearchParams(location.search);
const POLICY_PILOT = QUERY.get("pilot") === "policy";
/** `?weapon=laser` (or gatling/shotgun/shield) hands the human that weapon at
 *  the start of every round, so a weapon can be tried without waiting on a
 *  random crate. Same debug-hook convention as `?pilot=policy`. */
const FORCED_WEAPONS = { gatling: 1, shotgun: 2, shield: 3, laser: 4 };
const FORCED_WEAPON = FORCED_WEAPONS[QUERY.get("weapon")] ?? null;
const requestedPilotTarget = Number(QUERY.get("target"));
const POLICY_PILOT_TARGET = Number.isInteger(requestedPilotTarget)
  ? Math.min(20, Math.max(2, requestedPilotTarget))
  : 2;
const requestedPilotSeed = Number(QUERY.get("seed"));
const POLICY_PILOT_SEED = Number.isInteger(requestedPilotSeed)
  && requestedPilotSeed >= 0 && requestedPilotSeed <= 0xffffffff
  ? requestedPilotSeed >>> 0 : null;
const POLICY_PILOT_STEPS_PER_FRAME = 64;

// Render buffer layout, matching engine/src/wasm.rs's build_render() doc
// comment: 18 header slots, then 120 paint flags (unused here — killfield has
// no paint mechanic), then wall_count*4, tank_count*6, bullet_count*2.
const HEADER_SLOTS = 21;   // engine/src/wasm.rs; [18..20] pickups + laser
const TANK_SLOTS = 8;      // x, y, rotation, alive, number, scale, weapon, shield
const PICKUP_SLOTS = 3;    // x, y, weapon
const PAINT_SLOTS = 12 * 10;
const HEADER = HEADER_SLOTS + PAINT_SLOTS;

const THEME = {
  page: "#FFFFFF",
  ground: "#EFEDE8",
  wall: "#3F4550",
  bullet: "#101214",
  outline: "#08090B",
  // The beam: a near-white core inside a saturated halo, so it reads as light
  // rather than as another wall stroke.
  laserCore: "#FFF4E8",
  // The unfired aiming line, deliberately cooler than a live beam.
  laserAim: "#C2472E",
  laserHalo: "#FF3B30",
};
// Exactly two colours, by role rather than by tank index — a seat can be any
// controller in Watch mode, so "tank 0 is always killfield" no longer holds.
// Laika can only ever be selected on the "black" side (see index.html's
// controller-0, which has no Laika option), so this pairing is enforced by
// the option lists, not by a runtime rule that would need to override a
// seat's colour depending on who happens to be in it.
const ROLE_COLORS = {
  red: { base: "#9E101B", turret: "#D82432" },
  black: { base: "#17191C", turret: "#35383D" },
};
const CONTROLLER_NAMES = { killfield: "Killfield", laika: "Laika", hybrid: "Hybrid" };

/** Which role (red/black) each tank plays this mode. Watch: seat 0 (Left) is
 *  always red, seat 1 (Right) always black. Play: the human is red (the
 *  original "player" colour), the selectable opponent is black — matching
 *  the original Killfield-is-black convention regardless of which of the
 *  three controllers is standing in for it. */
function roleForSeat(seat) {
  if (mode === "play") return seat === 1 ? "red" : "black";
  return seat === 0 ? "red" : "black";
}
const MODES = {
  watch: { humanTank: null },
  play: { humanTank: 1 },
};

// ---------------------------------------------------------------- DOM refs
const canvas = document.getElementById("screen");
const roundline = document.getElementById("roundline");
const streakline = document.getElementById("streakline");
const nameLabels = [0, 1].map((i) => document.getElementById(`name-${i}`));
const scoreLabels = [0, 1].map((i) => document.getElementById(`score-${i}`));
const swatches = [0, 1].map((i) => document.getElementById(`swatch-${i}`));
const rerollButton = document.getElementById("reroll");
const resetScoreButton = document.getElementById("reset-score");
const instantTurnButton = document.getElementById("instant-turn");
const mouseAimButton = document.getElementById("mouse-aim");
const pickupsButton = document.getElementById("pickups");
const forwardAlignmentInput = document.getElementById("forward-alignment");
const forwardAlignmentLabel = document.getElementById("forward-alignment-label");
const forwardAlignmentValue = document.getElementById("forward-alignment-value");
const watchConfig = document.getElementById("watch-config");
const playConfig = document.getElementById("play-config");
const watchLeftLabel = document.getElementById("watch-left-label");
const watchRightLabel = document.getElementById("watch-right-label");
const controllerSelects = [0, 1].map((i) => document.getElementById(`controller-${i}`));
const themedPickers = [...document.querySelectorAll("[data-theme-picker]")];
const playOpponentSelect = document.getElementById("play-opponent");
const playOpponentLabel = document.getElementById("play-opponent-label");
const reactionDelaySelect = document.getElementById("reaction-delay");
const reactionDelayLabel = document.getElementById("reaction-delay-label");
const reactionDelayField = document.getElementById("reaction-delay-field");
const openingDelayInput = document.getElementById("opening-delay");
const openingDelayLabel = document.getElementById("opening-delay-label");
const openingDelayValue = document.getElementById("opening-delay-value");
const openingDelayField = document.getElementById("opening-delay-field");
const controlsHelp = document.getElementById("controls-help");
const controlsHelpTrigger = document.getElementById("controls-help-trigger");
const controlsHelpTitle = document.getElementById("controls-help-title");
const controlForward = document.getElementById("control-forward");
const controlBackup = document.getElementById("control-backup");
const controlLeft = document.getElementById("control-left");
const controlRight = document.getElementById("control-right");
const controlFire = document.getElementById("control-fire");
const controlReroll = document.getElementById("control-reroll");
const controlPause = document.getElementById("control-pause");
const watchButton = document.getElementById("mode-watch");
const playButton = document.getElementById("mode-play");
const stage = document.getElementById("stage");
const pauseButton = document.getElementById("pause");
const soundButton = document.getElementById("sound");
const fullscreenButton = document.getElementById("fullscreen");
const langToggle = document.getElementById("lang-toggle");
const touchControlsRoot = document.getElementById("touch-controls");
const touchVisibilityButton = document.getElementById("touch-visibility");
const orientationHint = document.getElementById("orientation-hint");
const orientationTitle = document.getElementById("orientation-title");
const orientationBody = document.getElementById("orientation-body");

const keyboard = new Keyboard();
const touchControls = new TouchControls(touchControlsRoot, touchVisibilityButton);
const mouseAim = new MouseAim(canvas);
const sounds = new SoundEffects();
let keyboardFirePressed = false;
let touchFirePressed = false;
let mouseFirePressed = false;
let immediateFirePressed = false;

let wasm = null;
let hybridPolicy = null;
let scratchPtr = null;
let openingDelaySeconds = DEFAULT_OPENING_DELAY_SECONDS;
try {
  openingDelaySeconds = normaliseOpeningDelay(localStorage.getItem(OPENING_DELAY_STORAGE_KEY));
} catch {
  // Keep the default when browser storage is unavailable.
}
let reactionDelayFrames = 0;
try {
  reactionDelayFrames = normaliseReactionDelay(localStorage.getItem(REACTION_DELAY_STORAGE_KEY));
} catch {
  // Keep the default when browser storage is unavailable.
}

function normaliseOpeningDelay(raw) {
  if (raw === null || raw === "") return DEFAULT_OPENING_DELAY_SECONDS;
  const value = Number(raw);
  if (!Number.isFinite(value)) return DEFAULT_OPENING_DELAY_SECONDS;
  return Math.max(0, Math.min(3, Math.round(value * 10) / 10));
}

function normaliseReactionDelay(raw) {
  const value = Number(raw);
  if (!Number.isInteger(value)) return 0;
  return Math.max(0, Math.min(3, value));
}

function openingDelayFrameCount() {
  return Math.round(openingDelaySeconds * C.FPS);
}

// ------------------------------------------------------------- renderer
const MAX_DPR = 2;

class Renderer {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d");
    // Drawing always happens in this fixed logical space: the maze's scale is
    // chosen (engine-side) so any round's footprint fits inside it. How large
    // it appears on screen is the stylesheet's business, not the renderer's —
    // resizing this per round is what made rectangular mazes blow up the box.
    this.width = C.MOVIEWIDTH + 20;
    this.height = C.MOVIEHEIGHT + 20;
    canvasEl.style.aspectRatio = `${this.width} / ${this.height}`;
    if (typeof ResizeObserver !== "undefined") {
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(canvasEl);
    }
    this.sizeCheckTick = 0;
    this.resize();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = rect.width || this.width;
    const cssHeight = rect.height || this.height;
    const deviceWidth = Math.max(1, Math.round(cssWidth * dpr));
    const deviceHeight = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== deviceWidth) this.canvas.width = deviceWidth;
    if (this.canvas.height !== deviceHeight) this.canvas.height = deviceHeight;
    this.ctx.setTransform(deviceWidth / this.width, 0, 0, deviceHeight / this.height, 0, 0);
  }

  syncSize() {
    if (this.sizeCheckTick++ % 15 !== 0) return;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    if (Math.abs(Math.round(rect.width * dpr) - this.canvas.width) > 1) this.resize();
  }
}

const renderer = new Renderer(canvas);
// Dedicated, fixed-seed RNG for the kill-shake jitter — decoupled from the
// engine's own RNG so drawing a frame never perturbs game determinism.
const shakeRng = new Rng(1);

// ---------------------------------------------------------------- wasm glue

/** The buffer view must be rebuilt each frame: wasm memory can grow. */
function renderBuffer() {
  const ptr = wasm.kf_render_ptr(handle);
  const len = wasm.kf_render_len(handle);
  return new Float32Array(wasm.memory.buffer, ptr, len);
}

function captureRenderState(buf) {
  const nWalls = buf[5] | 0;
  const nTanks = buf[6] | 0;
  const nBullets = buf[7] | 0;
  const tankBase = HEADER + nWalls * 4;
  const bulletBase = tankBase + nTanks * TANK_SLOTS;
  return {
    round: buf[9],
    tanks: Array.from({ length: nTanks }, (_, i) => {
      const o = tankBase + i * TANK_SLOTS;
      return { x: buf[o], y: buf[o + 1], rotation: buf[o + 2] };
    }),
    bullets: Array.from({ length: nBullets }, (_, i) => ({
      x: buf[bulletBase + i * 2], y: buf[bulletBase + i * 2 + 1],
    })),
  };
}

/** Interpolate through the short side of the wraparound at +/-180 degrees. */
function interpolateAngle(from, to, alpha) {
  let delta = (to - from) % 360;
  if (delta > 180) delta -= 360;
  else if (delta <= -180) delta += 360;
  return from + delta * alpha;
}

// ------------------------------------------------------------------ render

function drawTank(ctx, x, y, rotation, s, colors) {
  const th = rotation * C.DEG;
  const c = Math.cos(th);
  const sn = Math.sin(th);
  const px = (lx, ly) => x + s * (lx * c - ly * sn);
  const py = (lx, ly) => y + s * (lx * sn + ly * c);
  const poly = (pts) => {
    ctx.beginPath();
    ctx.moveTo(px(pts[0][0], pts[0][1]), py(pts[0][0], pts[0][1]));
    for (let i = 1; i < pts.length; i++) ctx.lineTo(px(pts[i][0], pts[i][1]), py(pts[i][0], pts[i][1]));
    ctx.closePath();
  };
  const bw2 = C.TANK_BASE_WIDTH / 2;
  const bh2 = C.TANK_BASE_HEIGHT / 2;
  ctx.lineJoin = "round";

  // Hull
  poly([[-bw2, -bh2], [bw2, -bh2], [bw2, bh2], [-bw2, bh2]]);
  ctx.fillStyle = colors.base;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Tracks
  ctx.fillStyle = THEME.outline;
  for (const side of [-1, 1]) {
    poly([
      [side * bw2, -bh2], [side * bw2 * 0.62, -bh2],
      [side * bw2 * 0.62, bh2], [side * bw2, bh2],
    ]);
    ctx.fill();
  }

  // Barrel
  const hw = C.TANK_SHAPE_BARREL_HALF_WIDTH;
  const tip = C.TANK_SHAPE_BARREL_TIP_Y;
  poly([[-hw, 0], [hw, 0], [hw, tip], [-hw, tip]]);
  ctx.fillStyle = colors.turret;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Turret dome
  ctx.beginPath();
  ctx.arc(px(0, 0), py(0, 0), s * 23.5, 0, Math.PI * 2);
  ctx.fillStyle = colors.turret;
  ctx.fill();
  ctx.strokeStyle = THEME.outline;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

/** Weapon codes from engine/src/pickups.rs's `Weapon::code`. */
const WEAPON_NONE = 0;
const WEAPON_GATLING = 1;
const WEAPON_SHOTGUN = 2;
const WEAPON_SHIELD = 3;
const WEAPON_LASER = 4;

/**
 * The aiming line for a loaded laser: the exact path `laser::trace` says the
 * shot would take, bounces included. Dashed and thin so it never competes with
 * a beam that was actually fired, and it goes solid-bright the moment the path
 * ends on a tank — that flip is the whole aiming signal.
 */
function drawLaserPreview(ctx, preview, ox, oy, scale) {
  if (!preview || preview.points.length < 2) return;
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  // Tuned by eye against the light floor: the first pass was a 1px line at
  // 0.4 alpha and was effectively invisible, which defeats the whole point.
  ctx.setLineDash(preview.wouldHit ? [] : [scale * 0.2, scale * 0.14]);
  ctx.strokeStyle = preview.wouldHit ? THEME.laserHalo : THEME.laserAim;
  ctx.globalAlpha = preview.wouldHit ? 0.95 : 0.62;
  ctx.lineWidth = Math.max(1.6, scale * (preview.wouldHit ? 0.075 : 0.05));
  ctx.beginPath();
  ctx.moveTo(ox + preview.points[0].x, oy + preview.points[0].y);
  for (let i = 1; i < preview.points.length; i += 1) {
    ctx.lineTo(ox + preview.points[i].x, oy + preview.points[i].y);
  }
  ctx.stroke();
  // A ring on the end point: where the shot stops, whether that is a hull, a
  // wall, or the end of its range.
  const last = preview.points[preview.points.length - 1];
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(ox + last.x, oy + last.y, scale * (preview.wouldHit ? 0.15 : 0.09), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

/**
 * The laser's afterglow: the polyline the beam actually travelled, drawn as a
 * hot core inside a wider halo, fading over `LASER_BEAM_FRAMES`. The engine
 * hands over an alpha rather than a frame count, so the fade lives in one place.
 */
function drawBeam(ctx, buf, base, pointCount, alpha, ox, oy, scale) {
  if (pointCount < 2 || !(alpha > 0)) return;
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const trace = () => {
    ctx.beginPath();
    ctx.moveTo(ox + buf[base], oy + buf[base + 1]);
    for (let i = 1; i < pointCount; i += 1) {
      ctx.lineTo(ox + buf[base + i * 2], oy + buf[base + i * 2 + 1]);
    }
    ctx.stroke();
  };
  ctx.globalAlpha = alpha * 0.3;
  ctx.strokeStyle = THEME.laserHalo;
  ctx.lineWidth = Math.max(3, scale * 0.16);
  trace();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = THEME.laserCore;
  ctx.lineWidth = Math.max(1.2, scale * 0.045);
  trace();
  ctx.restore();
}

/**
 * A crate on the floor. The glyph inside says which weapon without needing a
 * legend: three bars for the gatling's rate of fire, a fan for the shotgun's
 * spread, an arc for the shield, a bolt for the laser.
 */
function drawPickup(ctx, x, y, weapon, scale) {
  const r = scale * 0.21;
  ctx.save();
  ctx.translate(x, y);
  ctx.lineWidth = Math.max(1.5, scale * 0.035);
  ctx.fillStyle = THEME.page;
  ctx.strokeStyle = THEME.outline;
  ctx.beginPath();
  ctx.rect(-r, -r, r * 2, r * 2);
  ctx.fill();
  ctx.stroke();
  ctx.lineWidth = Math.max(1.2, scale * 0.028);
  ctx.beginPath();
  if (weapon === WEAPON_GATLING) {
    for (let i = -1; i <= 1; i += 1) {
      ctx.moveTo(-r * 0.45, i * r * 0.38);
      ctx.lineTo(r * 0.45, i * r * 0.38);
    }
  } else if (weapon === WEAPON_SHOTGUN) {
    for (let i = -1; i <= 1; i += 1) {
      ctx.moveTo(0, r * 0.5);
      ctx.lineTo(i * r * 0.5, -r * 0.5);
    }
  } else if (weapon === WEAPON_SHIELD) {
    ctx.arc(0, r * 0.25, r * 0.5, Math.PI, 0);
    ctx.moveTo(-r * 0.5, r * 0.25);
    ctx.lineTo(r * 0.5, r * 0.25);
  } else if (weapon === WEAPON_LASER) {
    // A bolt: one stroke with a kink, reading as a beam turning a corner.
    ctx.moveTo(-r * 0.55, r * 0.45);
    ctx.lineTo(r * 0.1, -r * 0.1);
    ctx.lineTo(r * 0.55, r * 0.2);
  }
  ctx.stroke();
  ctx.restore();
}

/** What a tank is carrying: a ring for the shield, pips for the loaded gun. */
function drawLoadout(ctx, x, y, weapon, shield, scale, color) {
  if (shield) {
    ctx.save();
    ctx.strokeStyle = color.turret;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = Math.max(1.5, scale * 0.035);
    ctx.beginPath();
    ctx.arc(x, y, scale * 0.33, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
  if (weapon === WEAPON_NONE) return;
  ctx.save();
  ctx.fillStyle = color.turret;
  ctx.strokeStyle = THEME.page;
  ctx.lineWidth = 1;
  const pips = weapon === WEAPON_GATLING ? 3 : 2;
  const step = scale * 0.09;
  for (let i = 0; i < pips; i += 1) {
    ctx.beginPath();
    ctx.arc(x + (i - (pips - 1) / 2) * step, y - scale * 0.36, scale * 0.035, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function draw(buf, colors, previous, alpha, localPlayer = null, aimOverlay = null,
  laserPreview = null) {
  renderer.syncSize();
  const ctx = renderer.ctx;
  const w = buf[0];
  const h = buf[1];
  const scale = buf[2];
  const halfT = buf[3];
  const shake = buf[4];
  const nWalls = buf[5] | 0;
  const nTanks = buf[6] | 0;
  const nBullets = buf[7] | 0;
  const worldW = w * scale;
  const worldH = h * scale;
  const width = renderer.width;
  const height = renderer.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.page;
  ctx.fillRect(0, 0, width, height);

  let ox = 10;
  let oy = 10;
  if (shake > 1) {
    const s = Math.max(1, Math.floor(shake));
    ox += shakeRng.randrange(s) - shake / 2;
    oy += shakeRng.randrange(s) - shake / 2;
  }
  ox += Math.max(0, (width - 20 - worldW) / 2);

  ctx.fillStyle = THEME.ground;
  ctx.fillRect(ox, oy, Math.floor(worldW), Math.floor(worldH));

  // Walls. Square caps are not decorative: the stroke's extent IS the
  // collision rectangle the simulation tests against.
  let p = HEADER;
  ctx.strokeStyle = THEME.wall;
  ctx.lineWidth = halfT * 2;
  ctx.lineCap = "square";
  ctx.beginPath();
  for (let i = 0; i < nWalls; i++) {
    ctx.moveTo(ox + buf[p], oy + buf[p + 1]);
    ctx.lineTo(ox + buf[p + 2], oy + buf[p + 3]);
    p += 4;
  }
  ctx.stroke();

  const tankBase = p;
  p += nTanks * TANK_SLOTS;
  const bulletBase = p;
  const pickupBase = bulletBase + nBullets * 2;
  const sameRound = previous && previous.round === buf[9];

  ctx.fillStyle = THEME.bullet;
  const br = Math.max(2.0, 2.5 * (scale / 50.0));
  // Bullets have no stable identity across the FFI boundary — the render
  // buffer only exposes them by slot index, and the engine's Vec can reorder
  // slots when a bullet is removed (e.g. a kill). Matching by index alone
  // would then interpolate two unrelated bullets' positions and produce a
  // visible teleport-jump exactly at kill moments. A real bullet can't move
  // farther than one frame's ballistic step, so treat any bigger jump as "a
  // different bullet now occupies this slot" and skip interpolation for it.
  const maxStep = C.BULLETSPEED * (scale / 50.0) * 1.5;
  const maxStepSq = maxStep * maxStep;
  for (let i = 0; i < nBullets; i++) {
    let old = sameRound ? previous.bullets[i] : null;
    if (old) {
      const dx = buf[bulletBase + i * 2] - old.x;
      const dy = buf[bulletBase + i * 2 + 1] - old.y;
      if (dx * dx + dy * dy > maxStepSq) old = null;
    }
    const bx = old ? old.x + (buf[bulletBase + i * 2] - old.x) * alpha : buf[bulletBase + i * 2];
    const by = old ? old.y + (buf[bulletBase + i * 2 + 1] - old.y) * alpha : buf[bulletBase + i * 2 + 1];
    ctx.beginPath();
    ctx.arc(ox + bx, oy + by, br, 0, Math.PI * 2);
    ctx.fill();
  }

  drawLaserPreview(ctx, laserPreview, ox, oy, scale);

  // The laser sits above the floor but below the hulls, so a beam that ends
  // in a tank reads as stopping at it rather than crossing it.
  drawBeam(ctx, buf, pickupBase + (buf[18] | 0) * PICKUP_SLOTS, buf[19] | 0, buf[20], ox, oy, scale);

  // Crates go under the tanks: driving onto one should read as covering it.
  const nPickups = buf[18] | 0;
  for (let i = 0; i < nPickups; i++) {
    const p = pickupBase + i * PICKUP_SLOTS;
    drawPickup(ctx, ox + buf[p], oy + buf[p + 1], buf[p + 2] | 0, scale);
  }

  for (let i = 0; i < nTanks; i++) {
    const o = tankBase + i * TANK_SLOTS;
    if (buf[o + 3] < 0.5) continue;
    const predicted = localPlayer?.tank === i ? localPlayer.pose : null;
    const old = !predicted && sameRound ? previous.tanks[i] : null;
    const x = predicted?.x ?? (old ? old.x + (buf[o] - old.x) * alpha : buf[o]);
    const y = predicted?.y ?? (old ? old.y + (buf[o + 1] - old.y) * alpha : buf[o + 1]);
    const rotation = predicted?.rotation
      ?? (old ? interpolateAngle(old.rotation, buf[o + 2], alpha) : buf[o + 2]);
    const number = buf[o + 4] | 0;
    drawTank(ctx, ox + x, oy + y, rotation, buf[o + 5], colors[number % colors.length]);
    drawLoadout(ctx, ox + x, oy + y, buf[o + 6] | 0, buf[o + 7] > 0.5, scale,
      colors[number % colors.length]);
  }

  if (aimOverlay !== null) {
    ctx.save();
    ctx.strokeStyle = THEME.outline;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(ox + aimOverlay.from.x, oy + aimOverlay.from.y);
    ctx.lineTo(ox + aimOverlay.to.x, oy + aimOverlay.to.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.arc(ox + aimOverlay.to.x, oy + aimOverlay.to.y, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

/**
 * Invert `draw`'s placement to read the cursor in world coordinates, then
 * offset it from the hull. Shake is deliberately left out of the inverse: the
 * maze jitters for a few frames after a kill, and aim must not jitter with it.
 */
function cursorAimOffset(tankPose) {
  if (!mouseAim.active() || !tankPose) return null;
  if (wasm === null || handle === null) return null;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const buf = renderBuffer();
  const ox = 10 + Math.max(0, (renderer.width - 20 - buf[0] * buf[2]) / 2);
  const oy = 10;
  const lx = (mouseAim.client.x - rect.left) / rect.width * renderer.width;
  const ly = (mouseAim.client.y - rect.top) / rect.height * renderer.height;
  return { x: lx - ox - tankPose.x, y: ly - oy - tankPose.y };
}

/** The dashed line and reticle that make an absolute heading readable. */
function aimOverlayFor(localPlayer) {
  const human = MODES[mode].humanTank;
  if (human === null) return null;
  const from = localPlayer?.tank === human
    ? localPlayer.pose
    : (previousRenderState?.tanks[human] ?? null);
  const offset = cursorAimOffset(from);
  if (offset === null) return null;
  return { from, to: { x: from.x + offset.x, y: from.y + offset.y } };
}

/**
 * Where your laser would land if you fired right now.
 *
 * An instant bouncing beam is unaimable without this: by the time you can see
 * the shot it has already resolved. The engine walks the same `laser::trace`
 * the shot itself uses, from the hull angle being *drawn* this frame rather
 * than the authoritative one, so the line stays welded to the barrel while the
 * hull is still turning. It mutates nothing.
 */
function laserPreviewFor(localPlayer, buf) {
  const human = MODES[mode].humanTank;
  if (human === null || wasm === null || handle === null) return null;
  const base = HEADER + (buf[5] | 0) * 4 + human * TANK_SLOTS;
  if (buf[base + 3] < 0.5 || (buf[base + 6] | 0) !== WEAPON_LASER) return null;
  const rotation = localPlayer?.tank === human
    ? localPlayer.pose.rotation
    : buf[base + 2];
  const count = wasm.kf_laser_preview(handle, human, rotation);
  if (count < 2) return null;
  const out = new Float32Array(
    wasm.memory.buffer, wasm.kf_laser_preview_ptr(), 2 + count * 2,
  );
  const points = [];
  for (let i = 0; i < count; i += 1) points.push({ x: out[2 + i * 2], y: out[2 + i * 2 + 1] });
  return { points, wouldHit: out[0] > 0.5 };
}

// ------------------------------------------------------------------ sound

function playSoundsForFlags(flags) {
  // Bit values from engine/src/wasm.rs's kf_step: 2=Fire, 16=Destroy, 32=Expire.
  if (flags & 2) sounds.playEvent(["fire"]);
  if (flags & 16) sounds.playEvent(["destroy"]);
  if (flags & 32) sounds.playEvent(["expire"]);
}

// ----------------------------------------------------------------- state

let lang = loadLang();
function t() { return STRINGS[lang]; }

function loadStreak() {
  try {
    const raw = localStorage.getItem(STREAK_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Number.isFinite(parsed.current) && Number.isFinite(parsed.longest)) {
        return { current: parsed.current, longest: parsed.longest };
      }
    }
  } catch {
    // localStorage can throw, or hold something we no longer trust; start fresh.
  }
  return { current: 0, longest: 0 };
}

function saveStreak() {
  try {
    localStorage.setItem(STREAK_STORAGE_KEY, JSON.stringify(streak));
  } catch {
    // Non-fatal: the streak just won't survive a reload.
  }
}

function syncForwardAlignmentControl() {
  const forward = touchControls.forwardAlignmentDegrees;
  const reverse = 360 - forward;
  const text = t().forwardAlignmentValue(forward, reverse);
  forwardAlignmentInput.value = String(forward);
  forwardAlignmentLabel.textContent = t().forwardAlignmentLabel;
  forwardAlignmentValue.textContent = text;
  forwardAlignmentInput.setAttribute(
    "aria-label", t().forwardAlignmentLabel + ": " + text,
  );
}

function syncReactionDelayControl() {
  const s = t();
  reactionDelayLabel.textContent = s.reactionDelayLabel;
  reactionDelaySelect.querySelectorAll("option").forEach((option, i) => {
    option.textContent = s.reactionDelayOptions[i];
  });
  reactionDelaySelect.value = String(reactionDelayFrames);
  reactionDelaySelect.setAttribute("aria-label", s.reactionDelayLabel);
}

function syncOpeningDelayControl() {
  const s = t();
  openingDelayLabel.textContent = s.openingDelayLabel;
  openingDelayValue.textContent = s.openingDelayValue(openingDelaySeconds);
  openingDelayInput.value = String(openingDelaySeconds);
  openingDelayInput.setAttribute(
    "aria-label", `${s.openingDelayLabel}: ${s.openingDelayValue(openingDelaySeconds)}`,
  );
}

function setThemedPickerOpen(picker, open) {
  picker.classList.toggle("open", open);
  const trigger = picker.querySelector(".controller-trigger");
  const menu = picker.querySelector(".controller-menu");
  trigger.setAttribute("aria-expanded", String(open));
  menu.setAttribute("aria-hidden", String(!open));
  menu.querySelectorAll("button").forEach((button) => { button.tabIndex = open ? 0 : -1; });
}

function closeThemedPickers(except = null) {
  themedPickers.forEach((picker) => {
    if (picker !== except) setThemedPickerOpen(picker, false);
  });
}

function syncThemedPicker(picker) {
  const select = document.getElementById(picker.dataset.selectId);
  const label = document.getElementById(picker.dataset.labelId);
  const value = select.value;
  const selected = [...select.options].find((option) => option.value === value);
  const trigger = picker.querySelector(".controller-trigger");
  trigger.querySelector("span").textContent = selected.textContent;
  trigger.setAttribute("aria-label", `${label.textContent}: ${selected.textContent}`);
  picker.querySelectorAll("[role=option]").forEach((option) => {
    const nativeOption = [...select.options].find((candidate) => candidate.value === option.dataset.value);
    if (nativeOption) option.textContent = nativeOption.textContent;
    option.setAttribute("aria-selected", String(option.dataset.value === value));
  });
}

function initialiseThemedPickers() {
  themedPickers.forEach((picker) => {
    const select = document.getElementById(picker.dataset.selectId);
    const trigger = picker.querySelector(".controller-trigger");
    setThemedPickerOpen(picker, false);
    syncThemedPicker(picker);
    trigger.addEventListener("click", () => {
      const open = !picker.classList.contains("open");
      closeThemedPickers(picker);
      setThemedPickerOpen(picker, open);
    });
    picker.querySelectorAll("[data-value]").forEach((option) => {
      option.addEventListener("click", () => {
        select.value = option.dataset.value;
        syncThemedPicker(picker);
        setThemedPickerOpen(picker, false);
        select.dispatchEvent(new Event("change"));
        trigger.focus();
      });
    });
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest("[data-theme-picker]")) closeThemedPickers();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeThemedPickers();
  });
}

function applyLanguage() {
  const s = t();
  document.documentElement.lang = s.htmlLang;
  langToggle.textContent = s.langToggleLabel;
  langToggle.setAttribute("aria-label", s.langToggleAria);
  watchButton.textContent = s.modeWatch;
  playButton.textContent = s.modePlay;
  watchLeftLabel.textContent = s.watchLeftLabel;
  watchRightLabel.textContent = s.watchRightLabel;
  playOpponentLabel.textContent = s.opponentLabel;
  rerollButton.textContent = s.reroll;
  resetScoreButton.textContent = s.resetScore;
  syncInstantTurnButton();
  syncMouseAimButton();
  syncPickupsButton();
  syncForwardAlignmentControl();
  syncReactionDelayControl();
  syncOpeningDelayControl();
  controlsHelpTrigger.textContent = s.controlsHelp.trigger;
  controlsHelpTrigger.setAttribute("aria-label", s.controlsHelp.trigger);
  controlsHelpTitle.textContent = s.controlsHelp.title;
  controlForward.textContent = s.controlsHelp.forward;
  controlBackup.textContent = s.controlsHelp.backup;
  controlLeft.textContent = s.controlsHelp.left;
  controlRight.textContent = s.controlsHelp.right;
  controlFire.textContent = s.controlsHelp.fire;
  controlReroll.textContent = s.controlsHelp.reroll;
  controlPause.textContent = s.controlsHelp.pause;
  themedPickers.forEach(syncThemedPicker);
  touchControls.setLabels(s.touchControls);
  orientationTitle.textContent = s.orientationTitle;
  orientationBody.textContent = s.orientationBody;
  syncFullscreenButton();
  syncPauseButton();
  syncSoundButton();
  updateScoreboard();
}

let mode = "watch";
let instantTurn = true;
try {
  const savedInstantTurn = localStorage.getItem(INSTANT_TURN_STORAGE_KEY);
  instantTurn = savedInstantTurn === null ? true : savedInstantTurn === "1";
} catch { /* Default stays on when browser storage is unavailable. */ }
let pickupsEnabled = false;
try {
  pickupsEnabled = localStorage.getItem(PICKUPS_STORAGE_KEY) === "1";
} catch { /* Off by default when browser storage is unavailable. */ }
let mouseMode = AIM_MODE_OFF;
try {
  const saved = localStorage.getItem(MOUSE_AIM_STORAGE_KEY);
  if (MOUSE_MODES.includes(saved)) mouseMode = saved;
} catch { /* Off by default when browser storage is unavailable. */ }
mouseAim.setMode(mouseMode);
let handle = null;
let paused = false;
let currentRound = 1;
let frozen = false;
let roundFrames = 0;
let previousRenderState = null;
/** Watch-mode controller assignment per seat, refreshed by newGame(). */
let seatController = ["hybrid", "laika"];
/** Seats a Hybrid policy must drive this tick — see driveHybridSeats(). */
let hybridSeats = [];
/** Play mode's opponent state machine: the actuation delay queue and the
 *  opening pause (src/opponent.js). Null in Watch mode, which has neither. */
let opponentDriver = null;

// Match score and win streak are tallied here, outside the engine: rebuilding
// the handle via kf_new (reroll or mode/controller change) resets the
// engine's own internal scores to 0. Only an explicit mode switch or the
// reset button clears our own tally on purpose.
let matchScore = [0, 0];
let streak = loadStreak();

function controllerForSeat(seat) {
  return mode === "play" ? (seat === 1 ? "human" : playOpponentSelect.value) : seatController[seat];
}

function activeTankColors() {
  return [0, 1].map((seat) => ROLE_COLORS[roleForSeat(seat)]);
}

function seatDisplayName(seat) {
  const c = controllerForSeat(seat);
  return c === "human" ? t().nameYou : CONTROLLER_NAMES[c];
}

function syncTeamColors() {
  const colors = activeTankColors();
  swatches.forEach((swatch, i) => {
    swatch.style.background = colors[i].turret;
    swatch.style.borderColor = colors[i].base;
  });
}

function resetScore() {
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  updateScoreboard();
}

/** Whose run the streak line reports: your own when you are playing, and the
 *  left seat when you are watching two agents. */
function streakSeat() {
  return mode === "play" ? HUMAN_SEAT : 0;
}

function applyRoundEnd(winner) {
  // -1: no winner yet; 2: double kill. Neither changes score or streak.
  if (winner !== 0 && winner !== 1) return;
  matchScore[winner] += 1;
  if (winner === streakSeat()) {
    streak.current += 1;
    if (streak.current > streak.longest) streak.longest = streak.current;
  } else {
    streak.current = 0;
  }
  saveStreak();
}

/** Neither delay control has any effect on Laika, which kf_step drives
 *  unconditionally — hide both rather than let them sit there inert. */
function syncPlayOpponentControls() {
  const hideDelays = mode === "play" && playOpponentSelect.value === "laika";
  reactionDelayField.hidden = hideDelays;
  openingDelayField.hidden = hideDelays;
}

function newGame() {
  const seed = POLICY_PILOT && POLICY_PILOT_SEED !== null
    ? POLICY_PILOT_SEED : (Math.random() * 0xffffffff) >>> 0;
  if (handle !== null) wasm.kf_free(handle);

  syncPlayOpponentControls();
  opponentDriver = null;
  if (mode === "play") {
    // Tank 1 is always the human; tank 0 is whichever opponent is selected.
    // The planner's opponent model must be honest here — a human is not
    // Laika's script — so opp_l1 is always on when Killfield is playing.
    const opponent = playOpponentSelect.value;
    handle = wasm.kf_new(seed, opponent === "laika" ? 1 : 0);
    hybridSeats = opponent === "hybrid" ? [0] : [];
    // The delay queue, the opening pause and Killfield's attachment all live
    // in the driver.
    opponentDriver = new OpponentDriver({
      opponent,
      delayFrames: reactionDelayFrames,
      openingDelayFrames: openingDelayFrameCount(),
      policy: hybridPolicy,
    });
    opponentDriver.attach(wasm, handle);
  } else {
    seatController = controllerSelects.map((select) => select.value);
    let laikaMask = 0;
    seatController.forEach((c, i) => { if (c === "laika") laikaMask |= (1 << i); });
    handle = wasm.kf_new(seed, laikaMask);
    hybridSeats = [];
    seatController.forEach((c, i) => {
      if (c === "killfield") {
        // Only Laika's script is worth simulating exactly; a Hybrid or
        // another Killfield opponent gets the honest "assume it holds its
        // current buttons" model instead.
        const otherIsLaika = seatController[1 - i] === "laika";
        wasm.kf_attach_mpc(handle, i, i === 0 ? 7 : 11, KILLFIELD_RAYS, otherIsLaika ? 0 : 1);
      } else if (c === "hybrid") {
        hybridSeats.push(i);
      }
    });
  }
  // Every new handle starts with crates off, so re-apply the preference.
  wasm.kf_set_pickups_enabled(handle, pickupsEnabled ? 1 : 0);
  applyForcedWeapon();
  syncTeamColors();

  roundFrames = 0;
  previousRenderState = captureRenderState(renderBuffer());
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;

  // In human play the world and human controls start immediately; only the
  // opponent's tank waits out the opening pause, which is what gives the
  // player real reaction time. Laika has no such hook — the engine drives it
  // unconditionally inside kf_step — so both delay controls are hidden for
  // that choice (see syncPlayOpponentControls).
}

function setMode(next) {
  mode = next;
  closeThemedPickers();
  keyboard.clear();
  touchControls.clear();
  stage.classList.toggle("play-mode", next === "play");
  watchButton.classList.toggle("active", next === "watch");
  playButton.classList.toggle("active", next === "play");
  watchConfig.hidden = next !== "watch";
  playConfig.hidden = next !== "play";
  controlsHelp.hidden = next !== "play";
  touchControls.setAvailable(next === "play");
  syncInstantTurnButton();
  // A mode switch changes who tank 1 even is, so treat it as a fresh match.
  matchScore = [0, 0];
  streak.current = 0;
  saveStreak();
  newGame();
}

function syncInstantTurnButton() {
  const s = t();
  instantTurnButton.classList.toggle("active", instantTurn);
  instantTurnButton.textContent = instantTurn ? s.instantTurnOn : s.instantTurnOff;
  instantTurnButton.setAttribute("aria-label", s.instantTurnAria);
  instantTurnButton.setAttribute("aria-pressed", String(instantTurn));
}

function toggleInstantTurn() {
  instantTurn = !instantTurn;
  try { localStorage.setItem(INSTANT_TURN_STORAGE_KEY, instantTurn ? "1" : "0"); } catch { /* optional */ }
  syncInstantTurnButton();
  instantTurnButton.blur();
}

function syncMouseAimButton() {
  const s = t();
  const on = mouseMode !== AIM_MODE_OFF;
  mouseAimButton.classList.toggle("active", on);
  mouseAimButton.textContent = mouseMode === AIM_MODE_DRIVE ? s.mouseDrive
    : mouseMode === AIM_MODE_AIM ? s.mouseAimOn : s.mouseAimOff;
  mouseAimButton.setAttribute("aria-label", s.mouseAimAria);
  mouseAimButton.setAttribute("aria-pressed", String(on));
}

function setMouseMode(next) {
  if (next === mouseMode) return;
  mouseMode = next;
  mouseAim.setMode(next);
  try { localStorage.setItem(MOUSE_AIM_STORAGE_KEY, next); } catch { /* optional */ }
  syncMouseAimButton();
}

/** One button, three states: off, cursor aims, cursor aims and drives. */
function toggleMouseAim() {
  setMouseMode(MOUSE_MODES[(MOUSE_MODES.indexOf(mouseMode) + 1) % MOUSE_MODES.length]);
}

function syncPickupsButton() {
  const s = t();
  pickupsButton.classList.toggle("active", pickupsEnabled);
  pickupsButton.textContent = pickupsEnabled ? s.pickupsOn : s.pickupsOff;
  pickupsButton.setAttribute("aria-label", s.pickupsAria);
  pickupsButton.setAttribute("aria-pressed", String(pickupsEnabled));
}

/** The engine clears the floor when this goes off, and starts its own spawn
 *  clock when it goes on; both take effect on the handle that is running. */
function togglePickups() {
  pickupsEnabled = !pickupsEnabled;
  try { localStorage.setItem(PICKUPS_STORAGE_KEY, pickupsEnabled ? "1" : "0"); } catch { /* optional */ }
  if (wasm !== null && handle !== null) {
    wasm.kf_set_pickups_enabled(handle, pickupsEnabled ? 1 : 0);
  }
  syncPickupsButton();
  pickupsButton.blur();
}

function updateScoreboard() {
  if (handle === null) return;
  const s = t();
  for (let i = 0; i < 2; i++) {
    const label = seatDisplayName(i);
    if (nameLabels[i].textContent !== label) nameLabels[i].textContent = label;
    const score = String(matchScore[i]);
    if (scoreLabels[i].textContent !== score) scoreLabels[i].textContent = score;
  }
  let text = frozen ? s.roundOver(currentRound) : s.round(currentRound);
  const pause = opponentDriver ? opponentDriver.pause : 0;
  if (mode === "play" && pause > 0 && !frozen) {
    text += ` · ${s.openingDelayCountdown(pause / C.FPS)}`;
  }
  if (paused) text += ` · ${s.paused}`;
  if (roundline.textContent !== text) roundline.textContent = text;
  const streakText = s.streakLine(streak.current, streak.longest);
  if (streakline.textContent !== streakText) streakline.textContent = streakText;
}

/**
 * Hand the opponent's action to the engine before kf_step consumes this
 * frame's controls.
 *
 * Play mode routes through OpponentDriver, which owns the actuation delay and
 * the opening pause; Watch mode has neither, and can drive both seats straight
 * from the policy.
 */
function driveHybridSeats() {
  if (hybridPolicy === null) return null;
  if (mode === "play") {
    const decision = opponentDriver ? opponentDriver.decide(wasm, handle) : null;
    if (decision) opponentDriver.apply(wasm, handle, decision.action);
    return decision;
  }
  for (const seat of hybridSeats) {
    const { observation, mask, dodge } = readObservation(wasm, handle, seat);
    wasm.kf_set_hybrid_action(handle, seat, hybridPolicy.act(observation, mask, dodge));
  }
  return null;
}

/** Re-arm the `?weapon=` debug loadout; fresh tanks each round drop it. */
function applyForcedWeapon() {
  if (FORCED_WEAPON === null) return;
  const human = MODES[mode].humanTank;
  if (human !== null) wasm.kf_set_weapon(handle, human, FORCED_WEAPON);
}

function tick() {
  // kf_step drives any attached Laika/MPC agent internally, so unlike
  // killfield's JS loop this only needs to push human input and any Hybrid
  // seat's chosen action before stepping.
  const human = MODES[mode].humanTank;
  let humanInput = null;
  if (human !== null) {
    // Movement is the share of this 40 ms frame each key was really held, so a
    // tap that falls between two ticks still registers instead of being lost.
    // Fire is passed straight through by sampleWindowStrengths and its edges
    // are applied authoritatively by syncImmediateHumanFire(), so a released
    // trigger is never resurrected by the window.
    if (POLICY_PILOT && hybridPolicy) {
      const own = readObservation(wasm, handle, human);
      humanInput = policyActionToInput(hybridPolicy.act(own.observation, own.mask, own.dodge));
      wasm.kf_set_input(handle, human, humanInput.forward, humanInput.backup,
        humanInput.turnLeft, humanInput.turnRight, humanInput.fire, 1);
    } else {
      const strengths = keyboard.sampleWindowStrengths(STEP_MS);
      const pose = previousRenderState?.tanks[human] ?? null;
      const rotation = pose?.rotation ?? 0;
      // The cursor owns the heading only while it is over the canvas; off it,
      // the wheel/keyboard path keeps its own turn keys.
      const aim = cursorAimOffset(pose);
      // Drive mode scales its throttle ramp by the round's cell size, so the
      // feel is the same on a cramped 4x4 as on a 12x10.
      const applied = aim !== null
        ? mouseAim.applyTo(wasm, handle, human, strengths, aim, rotation, instantTurn,
          renderBuffer()[2])
        : touchControls.applyTo(wasm, handle, human, strengths, rotation, instantTurn);
      humanInput = applied.input;
      if (applied.snappedRotation !== null && previousRenderState?.tanks[human]) {
        // Physics and presentation both snap in the same frame.
        previousRenderState.tanks[human].rotation = applied.snappedRotation;
      }
    }
  }
  driveHybridSeats();
  roundFrames += 1;
  const flags = wasm.kf_step(handle);
  if (opponentDriver) opponentDriver.afterStep(wasm, handle, flags);
  playSoundsForFlags(flags);
  const buf = renderBuffer();
  currentRound = buf[9];
  frozen = buf[14] > 0.5;
  if (flags & 1) { roundFrames = 0; applyForcedWeapon(); } // new_round
  if (flags & 64) applyRoundEnd(buf[15]); // round_end
  // The policy pilot drives the human seat at 25x to self-test; ?target=N
  // stops it once it has taken N rounds in a row.
  if (POLICY_PILOT && streak.current >= POLICY_PILOT_TARGET && !paused) togglePause();
}

let last = performance.now();
let accumulator = 0;

function predictHumanForRender(buf, alpha) {
  const human = MODES[mode].humanTank;
  if (human === null || paused || frozen || buf[14] > 0.5) return null;
  const nWalls = buf[5] | 0;
  const o = HEADER + nWalls * 4 + human * TANK_SLOTS;
  if (buf[o + 3] < 0.5) return null;
  const pose = { x: buf[o], y: buf[o + 1], rotation: buf[o + 2] };
  // Same fixed one-frame window the authoritative tick uses. Scaling it by
  // alpha instead would shrink the averaging interval as the frame drains and
  // make the predicted direction flicker on and off near the threshold.
  const strengths = keyboard.sampleWindowStrengths(STEP_MS);
  const input = touchControls.resolveMovement(strengths, pose.rotation);
  if (!(input.forward || input.backup || input.turnLeft || input.turnRight)) {
    return { tank: human, pose };
  }
  wasm.kf_predict_human_pose(
    handle, human, input.forward, input.backup, input.turnLeft, input.turnRight, scratchPtr,
  );
  const predicted = new Float32Array(wasm.memory.buffer, scratchPtr, 3);
  return {
    tank: human,
    pose: interpolatePredictedPose(pose, {
      x: predicted[0], y: predicted[1], rotation: predicted[2],
    }, alpha),
  };
}

function syncImmediateHumanFire() {
  const pressed = keyboardFirePressed || touchFirePressed || mouseFirePressed;
  if (pressed === immediateFirePressed) return;
  immediateFirePressed = pressed;
  const human = MODES[mode].humanTank;
  if (wasm === null || handle === null || human === null) return;
  // A release is always safe and must not be lost while paused/frozen, or the
  // next press could inherit a latched trigger. Only creation is gated.
  if (pressed && (paused || frozen)) return;
  if (wasm.kf_set_fire_immediate(handle, human, pressed ? 1 : 0)) {
    sounds.playEvent(["fire"]);
  }
}

function frame(now) {
  const budget = simulationBudget(
    accumulator, now - last, STEP_MS, MAX_CATCHUP_MS,
  );
  last = now;
  if (paused) {
    // Don't let the gap pile up while paused, or unpausing would fast-forward.
    accumulator = 0;
  } else {
    const steps = POLICY_PILOT && mode === "play" ? POLICY_PILOT_STEPS_PER_FRAME : budget.steps;
    for (let i = 0; i < steps; i++) {
      previousRenderState = captureRenderState(renderBuffer());
      tick();
    }
    accumulator = budget.remainder;
  }
  const renderAlpha = paused ? 1 : Math.min(1, accumulator / STEP_MS);
  const buf = renderBuffer();
  const localPlayer = predictHumanForRender(buf, renderAlpha);
  draw(buf, activeTankColors(), previousRenderState, renderAlpha, localPlayer,
    aimOverlayFor(localPlayer), laserPreviewFor(localPlayer, buf));
  updateScoreboard();
  requestAnimationFrame(frame);
}

function togglePause() {
  paused = !paused;
  syncPauseButton();
  updateScoreboard();
}

// Drawn rather than typed so the glyph is identical (not an emoji-presentation
// variant) on iOS and desktop alike.
const PAUSE_ICON =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">' +
  '<rect x="3" y="2.5" width="3.6" height="11" rx="0.7" fill="currentColor"/>' +
  '<rect x="9.4" y="2.5" width="3.6" height="11" rx="0.7" fill="currentColor"/>' +
  "</svg>";
const PLAY_ICON =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">' +
  '<path d="M4 2.6 13.2 8 4 13.4Z" fill="currentColor"/>' +
  "</svg>";
const SOUND_ON_ICON =
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path d="M2 6h3l3-3v10l-3-3H2Z" fill="currentColor"/>' +
  '<path d="M10 5.2c1.6 1.4 1.6 4.2 0 5.6M12 3.5c3 2.5 3 6.5 0 9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>' +
  "</svg>";
const SOUND_OFF_ICON =
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">' +
  '<path d="M2 6h3l3-3v10l-3-3H2Z" fill="currentColor"/>' +
  '<path d="m10 6 4 4m0-4-4 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>' +
  "</svg>";

function syncPauseButton() {
  const s = t();
  pauseButton.innerHTML = paused ? PLAY_ICON : PAUSE_ICON;
  pauseButton.setAttribute("aria-label", paused ? s.pauseExit : s.pauseEnter);
}

function syncSoundButton() {
  soundButton.innerHTML = sounds.enabled ? SOUND_ON_ICON : SOUND_OFF_ICON;
  soundButton.setAttribute("aria-label", sounds.enabled ? t().soundMute : t().soundUnmute);
}

function toggleSound() {
  sounds.setEnabled(!sounds.enabled);
  syncSoundButton();
}

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function setPseudoFullscreen(active) {
  stage.classList.toggle("pseudo-fullscreen", active);
  document.body.style.overflow = active ? "hidden" : "";
  syncFullscreenButton();
}

function syncOrientationHint() {
  const fullscreen = fullscreenElement() === stage || stage.classList.contains("pseudo-fullscreen");
  const portrait = window.matchMedia?.("(orientation: portrait)").matches
    ?? window.innerHeight > window.innerWidth;
  orientationHint.hidden = !(fullscreen && portrait);
}

async function preferLandscape() {
  if (!screen.orientation?.lock) return;
  try {
    await screen.orientation.lock("landscape");
  } catch {
    // iOS and some embedded browsers only support physical device rotation.
  } finally {
    syncOrientationHint();
  }
}

function releaseOrientationLock() {
  try { screen.orientation?.unlock?.(); } catch { /* optional platform feature */ }
}

async function toggleFullscreen() {
  if (fullscreenElement()) {
    releaseOrientationLock();
    await (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    return;
  }
  if (stage.classList.contains("pseudo-fullscreen")) {
    releaseOrientationLock();
    setPseudoFullscreen(false);
    return;
  }
  const request = stage.requestFullscreen || stage.webkitRequestFullscreen;
  if (!request) {
    setPseudoFullscreen(true);
    await preferLandscape();
    return;
  }
  try {
    await request.call(stage);
    if (fullscreenElement() !== stage) setPseudoFullscreen(true);
    await preferLandscape();
  } catch {
    setPseudoFullscreen(true);
    await preferLandscape();
  }
}

function syncFullscreenButton() {
  const active = fullscreenElement() === stage || stage.classList.contains("pseudo-fullscreen");
  fullscreenButton.textContent = active ? "⤢" : "⛶";
  fullscreenButton.setAttribute("aria-label", active ? t().fullscreenExit : t().fullscreenEnter);
  renderer.resize();
  syncOrientationHint();
}

function toggleLanguage() {
  lang = lang === "en" ? "zh" : "en";
  saveLang(lang);
  applyLanguage();
}

// -------------------------------------------------------------------- boot

async function boot() {
  const [wasmBytes, hybrid] = await Promise.all([
    fetch("kf_engine.wasm?v=d9df06c2").then((res) => res.arrayBuffer()),
    HybridPolicy.load("assets/hybrid.json?v=a1ab1f63", "assets/hybrid.bin?v=96169340"),
  ]);
  // Hashed before instantiation so a record names the exact binaries it is
  // reproducible against, rather than a version string someone could bump.
  const wasmResult = await WebAssembly.instantiate(wasmBytes, {});
  wasm = wasmResult.instance.exports;
  hybridPolicy = hybrid;
  scratchPtr = wasm.kf_scratch_ptr();
  if (wasm.kf_hybrid_schema_version() !== HYBRID_OBS_SCHEMA
      || wasm.kf_hybrid_observation_len() !== HYBRID_OBS_DIM + HYBRID_BULLET_SLOTS) {
    throw new Error("Hybrid observation layout mismatch between engine and viewer");
  }
  initialiseThemedPickers();

  fullscreenButton.addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", syncFullscreenButton);
  document.addEventListener("webkitfullscreenchange", syncFullscreenButton);
  screen.orientation?.addEventListener?.("change", syncOrientationHint);

  keyboard.onReroll = newGame;
  keyboard.onPause = togglePause;
  keyboard.onFireChange = (pressed) => {
    keyboardFirePressed = pressed;
    syncImmediateHumanFire();
  };
  touchControls.onFireChange = (pressed) => {
    touchFirePressed = pressed;
    syncImmediateHumanFire();
  };
  mouseAim.onFireChange = (pressed) => {
    mouseFirePressed = pressed;
    syncImmediateHumanFire();
  };
  mouseAimButton.addEventListener("click", () => {
    toggleMouseAim();
    mouseAimButton.blur();
  });
  pickupsButton.addEventListener("click", togglePickups);
  rerollButton.addEventListener("click", () => { newGame(); rerollButton.blur(); });
  resetScoreButton.addEventListener("click", () => { resetScore(); resetScoreButton.blur(); });
  instantTurnButton.addEventListener("click", () => toggleInstantTurn());
  pauseButton.addEventListener("click", () => { togglePause(); pauseButton.blur(); });
  soundButton.addEventListener("click", () => { toggleSound(); soundButton.blur(); });
  controllerSelects.forEach((select) => select.addEventListener("change", newGame));
  playOpponentSelect.addEventListener("change", newGame);
  forwardAlignmentInput.addEventListener("input", () => {
    touchControls.setForwardAlignmentDegrees(forwardAlignmentInput.value);
    syncForwardAlignmentControl();
  });
  reactionDelaySelect.addEventListener("change", () => {
    reactionDelayFrames = normaliseReactionDelay(reactionDelaySelect.value);
    try {
      localStorage.setItem(REACTION_DELAY_STORAGE_KEY, String(reactionDelayFrames));
    } catch { /* optional */ }
    if (handle !== null && mode === "play") wasm.kf_set_mpc_delay(handle, 0, reactionDelayFrames);
    if (opponentDriver) opponentDriver.delayFrames = reactionDelayFrames;
  });
  openingDelayInput.addEventListener("input", () => {
    openingDelaySeconds = normaliseOpeningDelay(openingDelayInput.value);
    try {
      localStorage.setItem(OPENING_DELAY_STORAGE_KEY, String(openingDelaySeconds));
    } catch { /* optional */ }
    syncOpeningDelayControl();
    if (opponentDriver) {
      opponentDriver.openingDelayFrames = opponentDriver.opponent === "laika"
        ? 0 : openingDelayFrameCount();
    }
  });
  watchButton.addEventListener("click", () => setMode("watch"));
  playButton.addEventListener("click", () => setMode("play"));
  langToggle.addEventListener("click", toggleLanguage);
  window.addEventListener("resize", () => {
    renderer.resize();
    syncOrientationHint();
  });

  // Web Audio must be resumed from a user gesture. Capturing both pointer and
  // keyboard makes watch mode and keyboard-only play behave the same way.
  window.addEventListener("pointerdown", () => sounds.unlock(), { once: true, capture: true });
  window.addEventListener("keydown", () => sounds.unlock(), { once: true, capture: true });

  // A hook for poking the engine and the policy from the console. It exposes
  // no capability a reader of this file does not already have.
  window.__kf = {
    get wasm() { return wasm; },
    get handle() { return handle; },
    get policy() { return hybridPolicy; },
  };

  // The pilot drives the human seat, so it needs Play mode to have one.
  setMode(POLICY_PILOT ? "play" : "watch");
  applyLanguage();
  requestAnimationFrame(frame);
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<pre style="color:#a13a3a;padding:16px">Failed to load: ${err}\n\n`
    + `Must be served over HTTP (not file://), with kf_engine.wasm and `
    + `assets/hybrid.{json,bin} next to index.html.\n`
    + `Run: bash viewer/build.sh, then: cd viewer && python3 -m http.server 8000</pre>`);
});
