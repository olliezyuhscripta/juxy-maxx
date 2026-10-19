#!/usr/bin/env bash
# Point the OpenCode "juxy" provider at a new Colab tunnel URL.
#
#   ./set_endpoint.sh https://foo-bar-baz.trycloudflare.com
#   (the /v1 suffix is optional — it is normalised either way)
set -euo pipefail

URL="${1:-}"
if [ -z "$URL" ]; then
  echo "usage: $0 https://<name>.trycloudflare.com" >&2
  exit 2
fi

CFG="${OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

# normalise once, up front, so the health check and the config agree
URL="${URL%/}"
if [[ "$URL" == */v1 ]]; then URL="${URL%/v1}"; fi

python3 - "$URL" "$CFG" <<'PY'
import json, sys

raw, cfg_path = sys.argv[1], sys.argv[2]
url = raw.strip().rstrip("/")
if url.endswith("/v1"):
    url = url[:-3]

cfg = json.load(open(cfg_path))
juxy = cfg.setdefault("provider", {}).setdefault(
    "juxy",
    {"npm": "@ai-sdk/openai-compatible", "name": "Juxy (Colab T4)", "options": {}, "models": {}},
)
juxy["options"]["baseURL"] = url + "/v1"
json.dump(cfg, open(cfg_path, "w"), indent=2)
open(cfg_path, "a").write("\n")
print("baseURL  ->", url + "/v1")
print("apiKey   ->", juxy["options"].get("apiKey", "(unset)"))
PY

echo -n "health   -> "
curl -s --max-time 20 "$URL/health" || echo "UNREACHABLE"
echo
echo "Restart OpenCode (or reload the config) to pick up the change."
