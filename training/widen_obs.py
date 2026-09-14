"""Carry a schema-24 checkpoint into schema 25 without disturbing it.

Schema 25 appends 36 channels for the weapon crates — both seats' loadouts,
the two crates on the floor, and the laser threat that no bullet channel can
represent. They go strictly after schema 24's last index, which is the whole
point: every older channel keeps the index it had, so the only weights that
need to change are the ones that read the observation directly.

Exactly one layer does: `scalars.0`, whose input is
`cat(obs[MAP_DIM:BULLET_OFFSET], obs[BULLET_OFFSET+100:])`. The new channels
land at the end of that second slice, so widening means appending zero columns
to `scalars.0.weight` — 36 of them — and leaving every other tensor alone. A
zero column contributes nothing, so the widened network computes exactly what
its ancestor did until training moves those columns off zero.

`duel_obs.rs`'s `the_appended_channels_are_inert_without_crates` is the other
half of the guarantee: with crates disabled the new channels are all zero
anyway, so a widened checkpoint plays a crate-free game identically no matter
what those columns eventually hold.

    python training/widen_obs.py viewer/assets/hybrid.bin --output outputs/v17b_s25.pt
    python training/widen_obs.py outputs/run/s11/final.pt --output outputs/run/s11/final_s25.pt

The first form reads the flat export the browser loads; the second reads a
training checkpoint. Either way the output is a `.pt` that `duel_ppo.py
--init-from` accepts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duel_env import OBS_DIM, SCALAR_DIM  # noqa: E402

# What schema 24 was. Hardcoded rather than imported, because the point of this
# tool is to bridge two layouts and it has to know the old one after the code
# has moved on to the new one.
OLD_OBS_DIM = 1028
OLD_SCALAR_DIM = 88
WIDENED_KEY = "scalars.0.weight"


def widen_state_dict(state: dict) -> tuple[dict, int]:
    """Pad the one layer that reads the observation. Returns (state, columns)."""
    if WIDENED_KEY not in state:
        raise KeyError(
            f"{WIDENED_KEY} is missing; this does not look like a duel checkpoint"
        )
    weight = state[WIDENED_KEY]
    if weight.shape[1] == SCALAR_DIM:
        return state, 0  # already wide
    if weight.shape[1] != OLD_SCALAR_DIM:
        raise ValueError(
            f"{WIDENED_KEY} has {weight.shape[1]} inputs, expected schema 24's "
            f"{OLD_SCALAR_DIM} or schema 25's {SCALAR_DIM}"
        )
    extra = SCALAR_DIM - OLD_SCALAR_DIM
    padding = torch.zeros(weight.shape[0], extra, dtype=weight.dtype)
    state = dict(state)
    state[WIDENED_KEY] = torch.cat((weight, padding), dim=1)
    return state, extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path,
                        help="a training .pt, or viewer/assets/hybrid.bin")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from duel_ppo import ARCH, ActorCritic  # noqa: PLC0415

    if args.source.suffix == ".bin":
        from hybrid_web import fill_from_export  # noqa: PLC0415

        # Build at the old width, fill, then widen — the export's tensors are
        # schema-24 shaped and would not fit a schema-25 model.
        model = ActorCritic()
        model.scalars[0] = torch.nn.Linear(OLD_SCALAR_DIM, 128)
        meta = fill_from_export(
            model, args.source, args.source.with_suffix(".json"),
            allow_missing=("critic",),
        )
        state = model.state_dict()
        origin = f"{args.source} ({meta['checkpoint']}, schema {meta['schema']})"
    else:
        payload = torch.load(args.source, map_location="cpu", weights_only=False)
        state = payload["model"] if "model" in payload else payload
        origin = str(args.source)

    state, columns = widen_state_dict(state)
    if columns == 0:
        print(f"{origin} is already schema 25; nothing to do")
        return 0

    # Prove it fits before writing anything.
    model = ActorCritic()
    model.load_state_dict(state, strict=False)
    missing = [k for k in model.state_dict() if k not in state]
    print(f"source     {origin}")
    print(f"widened    {WIDENED_KEY}: {OLD_SCALAR_DIM} -> {SCALAR_DIM} inputs "
          f"({columns} zero columns)")
    print(f"obs        {OLD_OBS_DIM} -> {OBS_DIM}")
    if missing:
        print(f"left fresh {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "arch": ARCH, "widened_from": origin}, args.output)
    print(f"wrote      {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
