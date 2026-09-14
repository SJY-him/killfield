#!/usr/bin/env bash
# Build the engine for the browser and drop the module next to the page.
set -euo pipefail
cd "$(dirname "$0")"

# GNU sed takes `-i`; BSD/macOS sed needs `-i ''`. Probing beats guessing from
# `uname`, because Git Bash on Windows ships GNU sed while macOS does not.
if sed --version >/dev/null 2>&1; then
  sed_i() { sed -i -E "$@"; }
else
  sed_i() { sed -i '' -E "$@"; }
fi

# coreutils has sha256sum; macOS ships shasum instead.
if command -v sha256sum >/dev/null 2>&1; then
  stamp() { sha256sum "$1" | cut -c1-8; }
else
  stamp() { shasum -a 256 "$1" | cut -c1-8; }
fi

cargo build --release --target wasm32-unknown-unknown --manifest-path ../engine/Cargo.toml
cp ../engine/target/wasm32-unknown-unknown/release/kf_engine.wasm .
wasm_stamp="$(stamp kf_engine.wasm)"
style_stamp="$(stamp style.css)"
hybrid_manifest_stamp="$(stamp assets/hybrid.json)"
hybrid_weights_stamp="$(stamp assets/hybrid.bin)"
sed_i \
  "s/fetch\(\"kf_engine\.wasm(\?v=[0-9A-Za-z._-]+)?\"\)/fetch(\"kf_engine.wasm?v=$wasm_stamp\")/" \
  viewer.js
# The weights are fetched by URL at runtime rather than imported, so they keep
# their own content stamps. Everything under src/ shares one stamp below.
sed_i \
  "s#assets/hybrid\.json\?v=[0-9A-Za-z._-]+#assets/hybrid.json?v=$hybrid_manifest_stamp#; \
   s#assets/hybrid\.bin\?v=[0-9A-Za-z._-]+#assets/hybrid.bin?v=$hybrid_weights_stamp#" \
  viewer.js
# viewer.js imports a dozen modules under src/ by bare path, and only i18n.js
# and hybrid.js were ever stamped. A rebuild could therefore leave the browser
# holding a stale mouse-aim.js or opponent.js and fail with "does not provide
# an export named ...", which points at the source rather than at the cache.
# One stamp covering every module busts them together; over-invalidating a few
# unchanged files costs nothing locally and avoids having to hash them in
# dependency order. (serve.py also sends no-store, which stops the problem
# arising at all; this keeps a plain `python -m http.server` honest too.)
src_stamp="$(cat src/*.js | if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi | cut -c1-8)"
sed_i "s#(from \")\./src/([a-z0-9-]+\.js)(\?v=[0-9A-Za-z._-]+)?\"#\1./src/\2?v=$src_stamp\"#g" viewer.js
viewer_stamp="$(stamp viewer.js)"
sed_i \
  "s/style\.css(\?v=[0-9A-Za-z._-]+)?/style.css?v=$style_stamp/; \
   s/viewer\.js(\?v=[0-9A-Za-z._-]+)?/viewer.js?v=$viewer_stamp/" \
  index.html
printf 'kf_engine.wasm  %s  (wasm=%s viewer=%s)\n' \
  "$(du -h kf_engine.wasm | cut -f1)" "$wasm_stamp" "$viewer_stamp"
echo 'serve with:  python viewer/serve.py     (caching off; Ctrl-C to stop)'
