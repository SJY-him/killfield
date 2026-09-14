"""The trainer's ActorCritic must be the network the viewer ships.

`duel_ppo.ActorCritic` is what the trainer optimises and
`export_hybrid_web.py` is what writes the browser's weights out of it, so the
two agreeing is what makes a warm start a continuation rather than a fresh
start wearing the old weights' name.

Nothing here raises if it breaks. `load_state_dict(strict=False)` takes the
tensors it recognises and leaves the rest at their initialisation, so a
mismatch trains happily and only shows up as a model that plays worse than the
one it claims to continue. This checks the logits instead.

    python training/tests/test_actor_critic_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

import torch.nn as nn  # noqa: E402

from duel_ppo import ActorCritic  # noqa: E402
from hybrid_web import fill_from_export  # noqa: E402
from widen_obs import OLD_SCALAR_DIM  # noqa: E402

WEIGHTS = ROOT / "viewer/assets/hybrid.bin"
MANIFEST = ROOT / "viewer/assets/hybrid.json"
PARITY = ROOT / "viewer/assets/hybrid-parity.json"
TOLERANCE = 2e-5


def main() -> int:
    fixture = json.loads(PARITY.read_text())
    obs = torch.tensor([fixture["obs"]], dtype=torch.float32)
    mask = torch.tensor([fixture["mask"]], dtype=torch.bool)
    dodge = torch.tensor([fixture["dodge"]], dtype=torch.float32)
    expected = torch.tensor([fixture["logits"]], dtype=torch.float32)

    model = ActorCritic(dodge_gate=True, ammo_gate=True)
    before = model.critic.weight.detach().clone()
    meta = fill_from_export(model, WEIGHTS, MANIFEST, allow_missing=("critic",))
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    critic_size = model.critic.weight.numel() + model.critic.bias.numel()
    print(f"checkpoint      {meta['checkpoint']} · schema {meta['schema']} "
          f"/ {meta['observation']} dims")
    print(f"parameters      {total:,} = export {meta['floats']:,} + critic {critic_size}")

    with torch.inference_mode():
        logits, value = model(obs, mask, dodge)
    explicit_error = float((logits - expected).abs().max())

    # The trainer never passes dodge: it slices those nine numbers out of the
    # observation itself, which is what the browser hands over too —
    # `opponent.js` builds its dodge argument as a subarray view of the very
    # same observation buffer. Both routes therefore have to agree.
    #
    # The exporter's fixture cannot show that on its own: it drew obs and dodge
    # as two independent random vectors, so there the explicit argument is
    # deliberately *not* what sits at DODGE_OFFSET. Put the fixture's dodge
    # where the runtime would have put it, and the two routes must become the
    # same computation exactly, not merely to within f32 noise.
    from duel_env import DODGE_DIM, DODGE_OFFSET  # noqa: PLC0415

    aligned = obs.clone()
    aligned[:, DODGE_OFFSET:DODGE_OFFSET + DODGE_DIM] = dodge
    with torch.inference_mode():
        sliced, _ = model(aligned, mask)
        explicit_aligned, _ = model(aligned, mask, dodge)
    slice_error = float((sliced - explicit_aligned).abs().max())

    print(f"explicit dodge  max error {explicit_error:.3e}")
    print(f"sliced dodge    max error vs explicit {slice_error:.3e}")
    print(f"argmax          {int(logits.argmax(1))} (expected {fixture['action']})")
    print(f"critic          value {float(value):+.4f}, head left at init: "
          f"{bool(torch.equal(model.critic.weight, before))}")

    # An ungated model must refuse the gated export rather than load part of it.
    refused = False
    try:
        narrow = ActorCritic(dodge_gate=False, ammo_gate=False)
        narrow.scalars[0] = nn.Linear(OLD_SCALAR_DIM, 128)
        fill_from_export(narrow, WEIGHTS, MANIFEST, allow_missing=("critic",))
    except KeyError:
        refused = True
    print(f"ungated model   refuses the gated export: {refused}")

    ok = (
        explicit_error <= TOLERANCE
        # Not a tolerance: aligned, the two routes are the same arithmetic.
        and slice_error == 0.0
        and total == meta["floats"] + critic_size
        and int(logits.argmax(1)) == int(fixture["action"])
        and torch.equal(model.critic.weight, before)
        and refused
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
