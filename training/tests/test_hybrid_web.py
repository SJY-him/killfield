"""Hold the PyTorch reconstruction against the forward pass that ships.

`hybrid_web.py` rebuilds the deployed actor from `hybrid.bin` + `hybrid.json`,
and its own `verify()` checks one point: the fixture the exporter wrote. That
fixture has a single fixed bullet mask and a single observation, so it says
nothing about the branches around it — an empty mask, a full mask, a batch of
more than one, or the clamps on the fire gate.

This runs the same cases through `viewer/src/hybrid.js` in node and compares.
That implementation is the one the browser actually executes and was measured
against the original PyTorch model at 9.54e-7, so agreeing with it to f32
rounding is the strongest statement available without the lost checkpoint.

    python training/tests/test_hybrid_web.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from hybrid_web import (  # noqa: E402
    AMMO_INDEX, BULLET_SLOTS, ETA_INDEX, HIT_INDEX, IDLE_STREAK_INDEX,
    SUICIDE_INDEX, load_deployed_actor,
)

# f32 through two different op orders. The repo's own JS/PyTorch check allows
# 2e-5 on the same comparison; hold this to the same bar.
TOLERANCE = 2e-5


def build_cases(rng: np.random.Generator, obs_dim: int) -> list[dict]:
    """Random observations, plus the mask and gate corners worth pinning."""
    cases: list[dict] = []

    def case(mask: list[bool], obs: np.ndarray | None = None) -> dict:
        values = rng.uniform(-1, 1, size=obs_dim).astype(np.float32) if obs is None else obs
        return {
            "obs": values.tolist(),
            "mask": [bool(m) for m in mask],
            "dodge": rng.uniform(-1, 1, size=9).astype(np.float32).tolist(),
        }

    # No bullets at all: mean stays zero and the max pool must not return -inf.
    cases.append(case([False] * BULLET_SLOTS))
    # Every slot live, and exactly one live, are the two pooling extremes.
    cases.append(case([True] * BULLET_SLOTS))
    for slot in (0, BULLET_SLOTS - 1):
        mask = [False] * BULLET_SLOTS
        mask[slot] = True
        cases.append(case(mask))

    # The fire gate reads five slots directly and clamps two of them. Drive
    # each past both ends of its clamp.
    for ammo, hit, suicide, eta, idle in (
        (0.0, 0.0, 0.0, -5.0, -5.0),
        (1.0, 1.0, 1.0, 5.0, 5.0),
        (0.5, 1.0, 0.0, 0.0, 0.32),
        (0.0, 0.0, 1.0, 0.9, 1.0),
    ):
        obs = rng.uniform(-1, 1, size=obs_dim).astype(np.float32)
        obs[AMMO_INDEX], obs[HIT_INDEX] = ammo, hit
        obs[SUICIDE_INDEX], obs[ETA_INDEX] = suicide, eta
        obs[IDLE_STREAK_INDEX] = idle
        cases.append(case(list(rng.random(BULLET_SLOTS) > 0.5), obs))

    # And a spread of ordinary frames.
    for _ in range(24):
        cases.append(case(list(rng.random(BULLET_SLOTS) > 0.5)))
    return cases


def browser_logits(cases: list[dict]) -> np.ndarray:
    script = Path(__file__).with_name("cross_check_hybrid.mjs")
    done = subprocess.run(
        ["node", str(script)],
        input=json.dumps({"cases": cases}),
        capture_output=True, text=True, cwd=ROOT,
    )
    if done.returncode != 0:
        raise RuntimeError(f"node failed:\n{done.stderr}")
    return np.array(json.loads(done.stdout)["logits"], dtype=np.float64)


def main() -> int:
    actor = load_deployed_actor(
        ROOT / "viewer/assets/hybrid.bin",
        ROOT / "viewer/assets/hybrid.json",
        ROOT / "viewer/assets/hybrid-parity.json",
    )

    fixture = actor.verify()
    print(f"exporter fixture     max error {fixture.max_logit_error:.3e} "
          f"argmax {fixture.actual_action} "
          f"{'OK' if fixture.action_matches else 'MISMATCH'}")
    if not fixture.action_matches or fixture.max_logit_error > TOLERANCE:
        print("FAIL: the reconstruction disagrees with the exporter's own fixture")
        return 1

    rng = np.random.default_rng(20260914)
    cases = build_cases(rng, actor.model.obs_dim)
    expected = browser_logits(cases)

    obs = torch.tensor([c["obs"] for c in cases], dtype=torch.float32)
    mask = torch.tensor([c["mask"] for c in cases], dtype=torch.bool)
    dodge = torch.tensor([c["dodge"] for c in cases], dtype=torch.float32)
    with torch.inference_mode():
        actual = actor.model(obs, mask, dodge).numpy().astype(np.float64)

    errors = np.abs(actual - expected)
    worst = int(errors.max(axis=1).argmax())
    agree = int((actual.argmax(1) == expected.argmax(1)).sum())
    print(f"browser cross-check  {len(cases)} cases, batched")
    print(f"  max |logit error|  {errors.max():.3e}  (worst case {worst})")
    print(f"  mean |logit error| {errors.mean():.3e}")
    print(f"  argmax agreement   {agree}/{len(cases)}")

    # A batch of one is a different code path through the pooling than a batch
    # of many; the deployed policy only ever runs one at a time.
    with torch.inference_mode():
        singles = np.array([
            actor.model(obs[i:i + 1], mask[i:i + 1], dodge[i:i + 1])[0].numpy()
            for i in range(len(cases))
        ], dtype=np.float64)
    batch_gap = np.abs(singles - actual).max()
    print(f"  batched vs single  {batch_gap:.3e}")

    ok = errors.max() <= TOLERANCE and agree == len(cases) and batch_gap <= TOLERANCE
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
