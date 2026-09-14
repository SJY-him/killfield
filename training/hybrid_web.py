"""Rebuild the deployed Hybrid actor in PyTorch from the files the browser loads.

The checkpoint the viewer ships (`viewer/assets/hybrid.bin` + `hybrid.json`) is
schema 24 with gated dodge/ammo heads. The training code that produced it was
never published: `upstream/rl`'s tip is schema 20, 1010 dims, and its
`ppo_models.py` contains no `dodge_delta`, `ammo_delta` or
`idle_logit_penalty` at all. So there is no `ActorCritic` in this repository
that can load `v17b_gs_league_u352.pt`, and nothing to continue training from.

What *is* published is the forward pass itself. `viewer/src/hybrid.js` is a
complete, exact reimplementation of it — the author measured it against
PyTorch at a maximum logit error of 9.54e-7 — and `hybrid.json` records every
tensor's name, shape and offset into the flat `f32` blob. Between them the
architecture is fully determined, and this module is that reconstruction.

`training/export_hybrid_web.py` is the inverse of this file and pins the
layout: the `.0`/`.2`/`.5` suffixes in the tensor names are `nn.Sequential`
indices, which is what fixes where the activations sit.

## What this recovers, and what it cannot

Recovered: the shared trunk, the map/bullet/scalar encoders, the actor head
and both residual gates — everything the browser needs to *act*.

Not recovered, because the export never wrote them:

  * the critic. `FIXED_KEYS` in the exporter stops at `actor.bias`, so the
    value head is gone and would have to be re-initialised and re-warmed.
  * the optimiser. Adam's moments live in `resume.pt`, not in the export, so a
    continued run restarts the moment estimates from zero.

Neither is fatal for fine-tuning, but both mean this is a warm start, not a
resumption: expect the first updates to be noisy while the critic catches up.

## Verification

`load_deployed_actor().verify()` runs the fixture the exporter emitted
alongside the weights (`hybrid-parity.json`: one observation, bullet mask and
dodge vector, with the logits the original PyTorch model produced) and reports
the largest disagreement. Anything near 1e-6 means this file reproduces the
deployed network; a large error means it does not, and the difference is a bug
here rather than a quirk of the checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

# Mirrored from engine/src/duel_obs.rs via viewer/src/hybrid.js.
MAP_W, MAP_H, MAP_C = 12, 10, 7
MAP_DIM = MAP_W * MAP_H * MAP_C          # 840
BULLET_OFFSET = 900
BULLET_SLOTS = 10
BULLET_DIM = 10
BULLET_BLOCK = BULLET_SLOTS * BULLET_DIM  # 100
OBS_DIM = 1028
ACTIONS = 18

# The scalar head reads everything that is neither the map grid nor the bullet
# rows: the gap between them, then the tail past them.
SCALAR_HEAD = BULLET_OFFSET - MAP_DIM               # 60
SCALAR_TAIL = OBS_DIM - BULLET_OFFSET - BULLET_BLOCK  # 28
SCALAR_DIM = SCALAR_HEAD + SCALAR_TAIL              # 88

# Individual observation slots the fire gate reads directly. These indices are
# hardcoded in hybrid.js too; they are the ammo fraction, the predicted-hit
# flag, the predicted-self-hit flag, the shot's time of flight, and the idle
# streak.
AMMO_INDEX = 863
HIT_INDEX = 890
SUICIDE_INDEX = 891
ETA_INDEX = 893
IDLE_STREAK_INDEX = 1027

DEFAULT_WEIGHTS = Path("viewer/assets/hybrid.bin")
DEFAULT_MANIFEST = Path("viewer/assets/hybrid.json")
DEFAULT_PARITY = Path("viewer/assets/hybrid-parity.json")


class HybridActor(nn.Module):
    """The deployed actor: schema 24, 1028 observations, 18 actions.

    Only the gated form is built. The exporter can emit an ungated checkpoint
    (flat `dodge_scale` / `ammo_scale` scalars instead of residual gates), but
    the deployed one is gated on both, and silently accepting either is exactly
    the failure `export_hybrid_web.py` warns about: a partial `state_dict` load
    that leaves half the architecture at its initialisation.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sequential indices are load-bearing: they are the tensor names.
        self.map = nn.Sequential(
            nn.Conv2d(MAP_C, 16, 3, stride=1, padding=1),   # 0
            nn.ReLU(),                                      # 1
            nn.Conv2d(16, 32, 3, stride=2, padding=1),      # 2
            nn.ReLU(),                                      # 3
            nn.Flatten(),                                   # 4
            nn.Linear(32 * 6 * 5, 128),                     # 5
            nn.Tanh(),                                      # 6
        )
        self.bullets = nn.Sequential(
            nn.Linear(BULLET_DIM, 32), nn.ReLU(),           # 0, 1
            nn.Linear(32, 32), nn.ReLU(),                   # 2, 3
        )
        self.scalars = nn.Sequential(nn.Linear(SCALAR_DIM, 128), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(128 + 128 + 32 + 32, 256), nn.Tanh())
        self.actor = nn.Linear(256, ACTIONS)

        # Residual gates: a warm-started constant plus a correction the trunk
        # can steer. `alpha_old` is a buffer rather than a parameter in the
        # export's key list, but it is a plain tensor either way.
        self.dodge_alpha_old = nn.Parameter(torch.zeros(()))
        self.dodge_delta = nn.Sequential(
            nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.ammo_alpha_old = nn.Parameter(torch.zeros(4))
        self.ammo_delta = nn.Sequential(
            nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 4),
        )
        self.idle_logit_penalty = nn.Parameter(torch.zeros(()))

    def features(self, obs: Tensor, mask: Tensor) -> Tensor:
        """The shared representation, before any head reads it."""
        batch = obs.shape[0]

        # hybrid.js's `toChw`: the observation stores the grid as
        # [y][x][channel] with y over MAP_W and x over MAP_H, and the conv
        # expects [channel][y][x]. Getting this transpose wrong still runs and
        # still trains — it just silently scrambles the maze.
        grid = obs[:, :MAP_DIM].reshape(batch, MAP_W, MAP_H, MAP_C)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        map_features = self.map(grid)

        rows = obs[:, BULLET_OFFSET:BULLET_OFFSET + BULLET_BLOCK]
        rows = rows.reshape(batch, BULLET_SLOTS, BULLET_DIM)
        encoded = self.bullets(rows)
        keep = mask.to(encoded.dtype).unsqueeze(-1)
        count = keep.sum(dim=1).clamp(min=1.0)
        mean = (encoded * keep).sum(dim=1) / count
        # Masked-out slots must not win the max, and an empty mask pools to
        # zero rather than to -inf — same two cases hybrid.js handles.
        peak = encoded.masked_fill(keep == 0, float("-inf")).max(dim=1).values
        peak = torch.where(mask.any(dim=1, keepdim=True), peak, torch.zeros_like(peak))

        scalars = torch.cat(
            (obs[:, MAP_DIM:BULLET_OFFSET],
             obs[:, BULLET_OFFSET + BULLET_BLOCK:]),
            dim=1,
        )
        scalars = self.scalars(scalars)
        return self.trunk(torch.cat((map_features, scalars, mean, peak), dim=1))

    def forward(self, obs: Tensor, mask: Tensor, dodge: Tensor) -> Tensor:
        features = self.features(obs, mask)
        logits = self.actor(features)

        dodge_scale = self.dodge_alpha_old + self.dodge_delta(features).squeeze(-1)
        ammo_scale, shot_quality, ammo_lock, suicide_scale = (
            self.ammo_alpha_old + self.ammo_delta(features)
        ).unbind(dim=1)

        ammo = obs[:, AMMO_INDEX]
        hit = obs[:, HIT_INDEX]
        suicide = obs[:, SUICIDE_INDEX]
        eta = (obs[:, ETA_INDEX] * 3.0).clamp(0.0, 1.0)
        hit_soon = hit * (1.0 - eta)
        fire_bias = (
            ammo_scale * ammo
            + shot_quality * hit_soon
            - ammo_lock * (1.0 - ammo) ** 2 * (1.0 - hit_soon)
            - suicide_scale * suicide
        )

        # Action index is [movement, fire]: every pair of logits shares a
        # movement, and the odd one of the pair is the one that fires.
        logits = logits + dodge_scale.unsqueeze(1) * dodge.repeat_interleave(2, dim=1)
        fire = torch.zeros_like(logits)
        fire[:, 1::2] = fire_bias.unsqueeze(1)
        logits = logits + fire

        idle = ((obs[:, IDLE_STREAK_INDEX] * 25.0 - 8.0) / 17.0).clamp(0.0, 1.0)
        penalty = (idle * self.idle_logit_penalty).unsqueeze(1)
        logits = logits.index_add(
            1, torch.tensor([8, 9], device=logits.device), -penalty.expand(-1, 2),
        )
        return logits


@dataclass
class Parity:
    max_logit_error: float
    action_matches: bool
    expected_action: int
    actual_action: int


class DeployedActor:
    """A `HybridActor` with the browser's weights in it, and its own receipt."""

    def __init__(self, model: HybridActor, manifest: dict, parity_path: Path):
        self.model = model
        self.manifest = manifest
        self._parity_path = parity_path

    def verify(self) -> Parity:
        """Replay the fixture the exporter wrote next to the weights."""
        fixture = json.loads(Path(self._parity_path).read_text())
        obs = torch.tensor([fixture["obs"]], dtype=torch.float32)
        mask = torch.tensor([fixture["mask"]], dtype=torch.bool)
        dodge = torch.tensor([fixture["dodge"]], dtype=torch.float32)
        expected = torch.tensor([fixture["logits"]], dtype=torch.float32)
        with torch.inference_mode():
            logits = self.model(obs, mask, dodge)
        return Parity(
            max_logit_error=float((logits - expected).abs().max()),
            action_matches=int(logits.argmax(1)) == int(fixture["action"]),
            expected_action=int(fixture["action"]),
            actual_action=int(logits.argmax(1)),
        )


def load_deployed_actor(
    weights: Path = DEFAULT_WEIGHTS,
    manifest: Path = DEFAULT_MANIFEST,
    parity: Path = DEFAULT_PARITY,
) -> DeployedActor:
    """Build the actor and fill it from the flat blob the browser downloads."""
    meta = json.loads(Path(manifest).read_text())
    if meta["schema"] != 24 or meta["observation"] != OBS_DIM or meta["actions"] != ACTIONS:
        raise ValueError(
            f"manifest is schema {meta['schema']}/{meta['observation']}/{meta['actions']}, "
            f"not 24/{OBS_DIM}/{ACTIONS}"
        )
    gated = meta.get("gated", {})
    if not (gated.get("dodge") and gated.get("ammo")):
        raise ValueError(
            "this loader only rebuilds the gated architecture; the manifest says "
            f"dodge={gated.get('dodge')} ammo={gated.get('ammo')}"
        )

    blob = np.fromfile(Path(weights), dtype="<f4")
    if blob.size != meta["floats"]:
        raise ValueError(f"{weights} holds {blob.size} floats, manifest says {meta['floats']}")

    model = HybridActor()
    state = model.state_dict()
    loaded = {}
    for name, spec in meta["tensors"].items():
        if name not in state:
            raise KeyError(f"manifest tensor {name!r} has nowhere to go in HybridActor")
        flat = blob[spec["offset"]:spec["offset"] + spec["length"]]
        value = torch.from_numpy(flat.reshape(spec["shape"]).copy())
        if value.shape != state[name].shape:
            raise ValueError(
                f"{name}: manifest shape {tuple(value.shape)} != "
                f"model shape {tuple(state[name].shape)}"
            )
        loaded[name] = value

    missing = sorted(set(state) - set(loaded))
    if missing:
        raise KeyError(f"the export has no weights for {missing}")
    model.load_state_dict(loaded)
    model.eval()
    return DeployedActor(model, meta, Path(parity))


if __name__ == "__main__":
    actor = load_deployed_actor()
    total = sum(p.numel() for p in actor.model.parameters())
    print(f"checkpoint : {actor.manifest['checkpoint']}")
    print(f"schema     : {actor.manifest['schema']} / {actor.manifest['observation']} dims")
    print(f"parameters : {total:,} (manifest says {actor.manifest['floats']:,} floats)")
    result = actor.verify()
    print(f"max |logit - expected| : {result.max_logit_error:.3e}")
    print(f"argmax                 : {result.actual_action} "
          f"(expected {result.expected_action}) "
          f"{'OK' if result.action_matches else 'MISMATCH'}")
