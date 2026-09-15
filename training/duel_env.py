"""ctypes wrapper over the Rust vectorised duel environment.

Shared by the trainer and the benchmark so there is exactly one place that
knows the ABI. Every buffer is a zero-copy `numpy` view into Rust memory and
stays valid until the next call that mutates the environment, so anything kept
across a `step` has to be copied.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

# Mirrored from engine/src/duel_obs.rs. Checked against the engine's own
# reported values at construction rather than trusted.
OBS_SCHEMA_VERSION = 25
OBS_DIM = 1064
BULLET_SLOTS = 10
ACTIONS = 18

MAP_W, MAP_H, MAP_CHANNELS = 12, 10, 7
MAP_DIM = MAP_W * MAP_H * MAP_CHANNELS          # 840
BULLET_OFFSET = 900
BULLET_DIM = 10
BULLET_BLOCK = BULLET_SLOTS * BULLET_DIM        # 100
# Everything that is neither the map grid nor the bullet rows.
SCALAR_DIM = OBS_DIM - MAP_DIM - BULLET_BLOCK   # 88

# Slots the actor's gates read straight out of the observation rather than
# through the trunk. `score::dodge_safety`'s per-movement survival outlook is
# already in here — the browser slices it out and hands it to the policy
# separately, but it is the same nine numbers, so the trainer just reads them
# in place. Mirrored from duel_obs.rs; the engine's reported OBS_DIM is the
# check that they still line up.
DODGE_OFFSET = 1018
DODGE_DIM = 9
IDLE_STREAK_INDEX = 1027
# The fire gate's five inputs: ammo fraction, predicted hit, predicted self
# hit, and the shot's time of flight.
AMMO_INDEX = 863
HIT_INDEX = 890
SUICIDE_INDEX = 891
ETA_INDEX = 893

OUTCOME_NAMES = {0: "running", 1: "win", 2: "loss", 3: "double", 4: "draw"}
OPPONENT_NAMES = {0: "laika", 1: "mpc", 2: "frozen"}
# `Weapon::code` in engine/src/pickups.rs. 0 means no drill.
DRILL_WEAPONS = {"none": 0, "gatling": 1, "shotgun": 2, "shield": 3, "laser": 4,
                 "homing": 5}

def _default_library() -> Path:
    """cargo names the cdylib per platform; pick the one that is there."""
    root = Path("engine/target/release")
    for name in ("kf_engine.dll", "libkf_engine.so", "libkf_engine.dylib"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "libkf_engine.dylib"


DEFAULT_LIBRARY = _default_library()


class DuelVec:
    def __init__(self, count: int, seed: int, weights=(1.0, 0.0, 0.0),
                 threads: int = 0, library: Path = DEFAULT_LIBRARY,
                 pickups: bool = False, drill: str = "none"):
        """`weights` is (laika, mpc, frozen); it is normalised, not required to
        sum to one. A frozen slot publishes tank 1's observation and expects an
        action back — see `obs_opponent` and `needs_action`.

        `pickups` turns the weapon crates on. Off by default, because that is
        the game every published benchmark was measured on and the only setting
        under which a schema-25 checkpoint is comparable to its ancestor.

        `drill` arms the opponent with a weapon every round, crates not
        involved. Learning to *use* a crate is an exploration problem with
        almost no gradient — a live tank crosses one about twice in 48,000
        frames of random play. Learning to survive what a crate produces is
        not: the threat arrives whether or not the policy goes looking."""
        self.count = count
        self.lib = ctypes.CDLL(str(Path(library).resolve()))

        for name in ("kf_duel_obs_dim", "kf_duel_bullet_slots", "kf_duel_action_count",
                     "kf_duel_frames", "kf_duel_grace_frames",
                     "kf_duel_obs_schema_version"):
            getattr(self.lib, name).restype = ctypes.c_uint32

        native = (
            int(self.lib.kf_duel_obs_dim()),
            int(self.lib.kf_duel_bullet_slots()),
            int(self.lib.kf_duel_action_count()),
            int(self.lib.kf_duel_obs_schema_version()),
        )
        expected = (OBS_DIM, BULLET_SLOTS, ACTIONS, OBS_SCHEMA_VERSION)
        # The gates read fixed indices, so a layout change that moved any of
        # them would silently feed the policy the wrong channel. Schema 25
        # appends past IDLE_STREAK_INDEX, so it is no longer last — what has
        # to hold is that the dodge block still runs straight into it and that
        # everything still lands inside the observation.
        if DODGE_OFFSET + DODGE_DIM != IDLE_STREAK_INDEX or IDLE_STREAK_INDEX >= OBS_DIM:
            raise RuntimeError(
                "the gate offsets no longer line up with the observation; "
                "duel_obs.rs moved them and duel_env.py was not updated"
            )
        if max(AMMO_INDEX, HIT_INDEX, SUICIDE_INDEX, ETA_INDEX) >= OBS_DIM:
            raise RuntimeError("a fire-gate index falls outside the observation")
        if native != expected:
            raise RuntimeError(
                f"engine/python schema mismatch: {native} != {expected}. "
                "Rebuild the engine (cargo build --release) or fix duel_env.py."
            )
        self.actions = ACTIONS
        self.episode_frames = int(self.lib.kf_duel_frames())
        self.grace_frames = int(self.lib.kf_duel_grace_frames())

        self.lib.kf_duel_new_drill.argtypes = [ctypes.c_uint32] * 8
        self.lib.kf_duel_new_drill.restype = ctypes.c_void_p
        self.lib.kf_duel_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.lib.kf_duel_reset_done.argtypes = [ctypes.c_void_p]
        self.lib.kf_duel_free.argtypes = [ctypes.c_void_p]

        total = sum(max(0.0, w) for w in weights) or 1.0
        permille = [int(round(1000 * max(0.0, w) / total)) for w in weights]
        self.weights = tuple(w / total for w in weights)
        # threads=0 lets the engine ask the OS how many cores it has.
        self.pickups = bool(pickups)
        if drill not in DRILL_WEAPONS:
            raise ValueError(f"drill must be one of {sorted(DRILL_WEAPONS)}, got {drill!r}")
        self.drill = drill
        self.handle = self.lib.kf_duel_new_drill(
            count, seed, *permille, threads, int(self.pickups), DRILL_WEAPONS[drill],
        )

        self.obs = self._view("kf_duel_obs", ctypes.c_float, (count, OBS_DIM))
        self.masks = self._view("kf_duel_masks", ctypes.c_uint8, (count, BULLET_SLOTS))
        self.obs_opponent = self._view("kf_duel_opponent_obs", ctypes.c_float,
                                       (count, OBS_DIM))
        self.masks_opponent = self._view("kf_duel_opponent_masks", ctypes.c_uint8,
                                         (count, BULLET_SLOTS))
        self.needs_action = self._view("kf_duel_needs_action", ctypes.c_uint8, (count,))
        self.rewards = self._view("kf_duel_rewards", ctypes.c_float, (count,))
        self.dones = self._view("kf_duel_dones", ctypes.c_uint8, (count,))
        self.terminals = self._view("kf_duel_terminals", ctypes.c_uint8, (count,))
        self.outcomes = self._view("kf_duel_outcomes", ctypes.c_uint8, (count,))
        self.opponents = self._view("kf_duel_opponents", ctypes.c_uint8, (count,))
        self.frames = self._view("kf_duel_episode_frames", ctypes.c_uint32, (count,))
        self.action_changes = self._view("kf_duel_action_changes", ctypes.c_uint32, (count,))
        self.shots = self._view("kf_duel_shots", ctypes.c_uint32, (count,))
        self.hits = self._view("kf_duel_hits", ctypes.c_uint32, (count,))
        # Frames this round the opponent's beam was lined up on the policy, and
        # whether the policy's death was a laser. Win rate barely moves on
        # either, which is exactly why the drill is scored on these instead.
        self.threat_frames = self._view("kf_duel_threat_frames", ctypes.c_uint32, (count,))
        self.laser_deaths = self._view("kf_duel_laser_deaths", ctypes.c_uint32, (count,))

    def _view(self, name, ctype, shape):
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.POINTER(ctype)
        return np.ctypeslib.as_array(function(self.handle), shape=shape)

    def step(self, actions, opponent_actions=None):
        actions = np.ascontiguousarray(actions, np.uint16)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        if opponent_actions is None:
            theirs = ctypes.cast(None, u16)
        else:
            buf = np.ascontiguousarray(opponent_actions, np.uint16)
            theirs = buf.ctypes.data_as(u16)
            self._opponent_buffer = buf  # keep alive across the call
        self.lib.kf_duel_step(self.handle, actions.ctypes.data_as(u16), theirs)

    def reset_done(self):
        self.lib.kf_duel_reset_done(self.handle)

    def close(self):
        if getattr(self, "handle", None):
            self.lib.kf_duel_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
