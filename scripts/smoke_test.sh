#!/usr/bin/env bash
# Smoke test a built image: does it come up, and is it serving the model we gated?
#
# Used by both CI and CD so the two cannot drift apart. CD runs it against the exact
# image it is about to push, which is the last check before a tag becomes a release.
#
# Usage: scripts/smoke_test.sh <image-ref> [artifacts-dir] [port]
set -euo pipefail

IMAGE="${1:?usage: smoke_test.sh <image-ref> [artifacts-dir] [port]}"
ARTIFACTS_DIR="${2:-artifacts}"
PORT="${3:-8000}"
CONTAINER="smoke-$$"

# The version the quality gate signed off on. Asserting the container reports this
# exact string is what proves the tested model is the shipped model — without it the
# test only proves the image serves *a* model.
EXPECTED_VERSION=$(python -c "import json;print(json.load(open('${ARTIFACTS_DIR}/metrics.json'))['model_version'])")
echo "expecting model_version=${EXPECTED_VERSION}"

cleanup() {
  docker logs "${CONTAINER}" 2>&1 | tail -30 || true
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "${CONTAINER}" -p "${PORT}:8000" "${IMAGE}" >/dev/null

for _ in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT}/ready" >/dev/null; then break; fi
  sleep 2
done
curl -sf "http://localhost:${PORT}/ready" >/dev/null \
  || { echo "ERROR: container never became ready" >&2; exit 1; }

health=$(curl -sf "http://localhost:${PORT}/health")
echo "health: ${health}"

grep -q '"model_loaded":true' <<<"${health}" \
  || { echo "ERROR: model not loaded" >&2; exit 1; }

grep -q "\"model_version\":\"${EXPECTED_VERSION}\"" <<<"${health}" \
  || { echo "ERROR: served model is not the gated model (expected ${EXPECTED_VERSION})" >&2; exit 1; }

# Behaviour, not just liveness: an unmistakable billing ticket must route to billing.
prediction=$(curl -sf -X POST "http://localhost:${PORT}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"text":"i was charged twice for my subscription this month"}')
echo "prediction: ${prediction}"

grep -q '"label":"billing"' <<<"${prediction}" \
  || { echo "ERROR: unmistakable billing ticket did not route to billing" >&2; exit 1; }

echo "smoke test passed: ${IMAGE} serving ${EXPECTED_VERSION}"
