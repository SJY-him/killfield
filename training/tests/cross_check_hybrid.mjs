// Evaluate the browser's Hybrid forward pass over a batch of cases supplied on
// stdin, so `test_hybrid_web.py` can hold its PyTorch reconstruction against
// the implementation that actually ships. Reads {cases:[{obs,mask,dodge}]} and
// writes {logits:[[...]]} — no assertions here; the Python side judges.
import fs from "node:fs";
import { HybridPolicy } from "../../viewer/src/hybrid.js";

const asset = (name) => new URL(`../../viewer/assets/${name}`, import.meta.url);
const manifest = JSON.parse(fs.readFileSync(asset("hybrid.json")));
const bytes = fs.readFileSync(asset("hybrid.bin"));
const weights = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const policy = new HybridPolicy(manifest, weights);

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const logits = input.cases.map(({ obs, mask, dodge }) => Array.from(
  policy.logits(Float32Array.from(obs), mask, Float32Array.from(dodge)),
));
process.stdout.write(JSON.stringify({ logits }));
