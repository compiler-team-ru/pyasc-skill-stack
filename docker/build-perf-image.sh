#!/usr/bin/env bash
# Build (and optionally push) the docker_full perf image `pyasc-sim-perf`.
#
# Stages a build context with the vendored reference operator repos
# (ops-math, ops-nn), the evaluation pyasc tree (pyasc-v2-eval, with its built
# native extension), and the host CANN simulator dirs (dav_3510 +
# Ascend950PR_9599), then builds docker/Dockerfile.perf on top of the standard
# pyasc-sim base image.
#
# MUST run on the host that has the private ops-* clones + pyasc-v2-eval and a
# working CANN install (GitHub-hosted runners cannot reach them). The output is
# pushed to ghcr.io and CI only `docker pull`s it.
#
# Usage:
#   docker/build-perf-image.sh                 # build only
#   docker/build-perf-image.sh --push          # build + push
#
# Env overrides:
#   WORKSPACE_ROOT   parent dir holding ops-math/ops-nn/pyasc-v2-eval
#                    (default: parent of this repo)
#   ASCEND_HOME_PATH CANN install root (default: /usr/local/Ascend/cann-9.0.0)
#   BASE_IMAGE       base pyasc-sim image (default: ghcr.io/<owner>/pyasc-sim:py3.11)
#   IMAGE            output tag (default: ghcr.io/<owner>/pyasc-sim-perf:py3.11)
#   OWNER            ghcr owner used to derive defaults (default: aloschilov)
set -euo pipefail

PUSH=0
[ "${1:-}" = "--push" ] && PUSH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OWNER="${OWNER:-aloschilov}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/${OWNER}/pyasc-sim:py3.11}"
# ARM64-ONLY: the camodel sims + pyasc native extension are arm64, so this
# image only runs on arm64 (CI's perf-gate uses ubuntu-24.04-arm). The tag
# carries an explicit -arm64 suffix to keep it distinct from the amd64 base.
IMAGE="${IMAGE:-ghcr.io/${OWNER}/pyasc-sim-perf:py3.11-arm64}"

SIMROOT="$ASCEND_HOME_PATH/tools/simulator"

echo "== build-perf-image =="
echo "  workspace : $WORKSPACE_ROOT"
echo "  cann      : $ASCEND_HOME_PATH"
echo "  base      : $BASE_IMAGE"
echo "  image     : $IMAGE"

for d in ops-math ops-nn pyasc-v2-eval; do
    [ -d "$WORKSPACE_ROOT/$d" ] || { echo "ERROR: missing $WORKSPACE_ROOT/$d" >&2; exit 1; }
done
[ -f "$WORKSPACE_ROOT/pyasc-v2-eval/python/asc/_C"/libpyasc.cpython-311-*-linux-gnu.so ] 2>/dev/null \
    || echo "WARN: pyasc-v2-eval native extension (.so) not found — gen side may fail to import asc"
[ -d "$SIMROOT/dav_3510/lib" ] || echo "WARN: dav_3510 sim not on host CANN; ref side will fail unless the base image ships it"

CTX="$(mktemp -d)"
# Tolerant cleanup: prior in-repo ops builds can leave root-owned files
# (e.g. _installed_opp/.../uninstall.sh) that a non-root rm cannot remove.
trap 'rm -rf "$CTX" 2>/dev/null || sudo rm -rf "$CTX" 2>/dev/null || true' EXIT
echo "  staging build context at $CTX ..."

# Vendored sources without .git / build outputs (keeps the image lean-ish and
# avoids copying root-owned install artifacts from prior local ref builds).
for d in ops-math ops-nn pyasc-v2-eval; do
    rsync -a --delete \
        --exclude='.git' \
        --exclude='build/' \
        --exclude='_installed_opp/' \
        --exclude='_build_cache/' \
        --exclude='**/__pycache__/' \
        "$WORKSPACE_ROOT/$d/" "$CTX/$d/"
done

# Host CANN simulator dirs — overlaid by the Dockerfile only if the base image
# is missing them.
mkdir -p "$CTX/sim"
for s in dav_3510 Ascend950PR_9599; do
    if [ -d "$SIMROOT/$s" ]; then
        rsync -a "$SIMROOT/$s/" "$CTX/sim/$s/"
    fi
done

cp "$SCRIPT_DIR/Dockerfile.perf" "$CTX/Dockerfile.perf"

echo "  building image ..."
docker build \
    -f "$CTX/Dockerfile.perf" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -t "$IMAGE" \
    "$CTX"

echo "  built: $IMAGE"
if [ "$PUSH" -eq 1 ]; then
    echo "  pushing $IMAGE ..."
    docker push "$IMAGE"
fi
echo "done."
