# pyasc-sim (self-contained): CANN 9.0 + Python 3.11 + pyasc (with asc2)
# built from the pyasc v2 source inside the image.
#
# This is the default image for running pyasc-skill-stack on a clean machine:
# no local pyasc checkout is needed on the host, and no overlay is copied in.
# For the alternative overlay-based variant (graft a host-side pyasc build into
# a prebuilt CANN image), see docker/Dockerfile.overlay.
#
# Build:
#   docker build -f docker/Dockerfile -t pyasc-sim:latest docker/
#
# Usage (mount the skill-stack repo as /workspace and drop into a shell):
#   docker run --rm -it \
#     -v "$(pwd)":/workspace -w /workspace \
#     pyasc-sim:latest
#
# Inside the container:
#   source "$CANN_HOME/set_env.sh"
#   export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
#   python3.11 -c "import asc, asc2, torch; print('pyasc OK')"
#   opencode
#
# The Ascend950PR_9599 simulator is required for matmul (cube unit). torch is
# installed (CPU-only) because asc2.matmul requires torch.Tensor inputs on the
# simulator -- numpy arrays produce zeros for matmul.
#
# Architecture support:
#   This image builds on both x86_64 and aarch64 Linux hosts. The base
#   `ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11` image is multi-arch
#   (linux/amd64 and linux/arm64), and the LLVM prebuilt archive used during
#   the pyasc build is auto-selected by `uname -m` at build time.
#
# Build args:
#   PYASC_GIT_URL         pyasc source repository
#                         (default: https://gitcode.com/compiler-team/pyasc.git)
#   PYASC_GIT_REV         SHA on origin/v2 to pin (default: 7b85554a, the Q1 baseline).
#                         Bump in lock-step with the host-side pyasc-v2-eval
#                         checkout used by the harness so docker and host runs stay
#                         apples-to-apples.
#   PYASC_GIT_REF         optional non-SHA ref (refs/merge-requests/N/head, branch
#                         name, etc). When set, takes precedence over PYASC_GIT_REV.
#                         Leave empty for the canonical pinned-SHA build.
#   LLVM_ARCHIVE_X86_64   prebuilt LLVM tarball URL for x86_64 hosts
#   LLVM_ARCHIVE_AARCH64  prebuilt LLVM tarball URL for aarch64 hosts
#   LLVM_ARCHIVE          optional override; if non-empty, used regardless of arch

FROM ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11

ENV CANN_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=${CANN_HOME}
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH}

ARG PYASC_GIT_URL=https://gitcode.com/compiler-team/pyasc.git
# Pin to a specific SHA on origin/v2 so every image build is reproducible.
# Bump this together with the host-side pyasc-v2-eval checkout when the
# evaluation owner is ready to refresh the baseline. Leaving PYASC_GIT_REF
# as a legacy override hook for callers that still need a non-SHA ref.
ARG PYASC_GIT_REV=7b85554a2bcc67805f90c30b4e262a6209767839
ARG PYASC_GIT_REF=
ARG LLVM_ARCHIVE_X86_64=https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz
ARG LLVM_ARCHIVE_AARCH64=https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-arm64.tar.gz
ARG LLVM_ARCHIVE=

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential ccache clang lld \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_INSTALL_PREFIX=/opt/llvm
# Download then extract (instead of piping curl into tar). On slow networks a
# mid-stream stall would otherwise corrupt the tar stream and fail with exit 2.
RUN set -eux ; \
    if [ -z "${LLVM_ARCHIVE}" ]; then \
        case "$(uname -m)" in \
            x86_64)  LLVM_ARCHIVE="${LLVM_ARCHIVE_X86_64}"  ;; \
            aarch64) LLVM_ARCHIVE="${LLVM_ARCHIVE_AARCH64}" ;; \
            *) echo "Unsupported arch: $(uname -m)" >&2 ; exit 1 ;; \
        esac ; \
    fi ; \
    echo "Using LLVM archive: ${LLVM_ARCHIVE}" ; \
    mkdir -p "${LLVM_INSTALL_PREFIX}" ; \
    curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors \
         -o /tmp/llvm.tar.gz "${LLVM_ARCHIVE}" ; \
    echo "Downloaded $(stat -c '%s' /tmp/llvm.tar.gz) bytes" ; \
    tar -xzf /tmp/llvm.tar.gz --strip-components=1 -C "${LLVM_INSTALL_PREFIX}" ; \
    rm -f /tmp/llvm.tar.gz

RUN pip install --no-cache-dir \
        "numpy<2.0" \
        "pytest>=7.0" \
        pyyaml \
        scipy \
        pybind11 \
        setuptools \
        setuptools_scm \
        wheel

RUN pip install --no-cache-dir \
        torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

ENV PYASC_SRC=/opt/pyasc
# Default path pins PYASC_GIT_REV (a SHA on origin/v2). The PYASC_GIT_REF
# fallback exists for ad-hoc callers that still need refs/merge-requests/N/head
# or similar; it must resolve to something git fetch understands.
RUN git clone "${PYASC_GIT_URL}" "${PYASC_SRC}" \
 && cd "${PYASC_SRC}" \
 && if [ -n "${PYASC_GIT_REF}" ]; then \
        git fetch origin "+${PYASC_GIT_REF}:pyasc-ref" \
     && git checkout pyasc-ref ; \
    else \
        git fetch origin v2 \
     && git checkout "${PYASC_GIT_REV}" ; \
    fi \
 && git --no-pager log -1 --format='pyasc revision: %H (%s)'

ENV PYASC_SETUP_CLANG_LLD=1 \
    PYASC_SETUP_CCACHE=1
RUN cd "${PYASC_SRC}" \
 && pip install --no-cache-dir -r requirements-build.txt \
 && pip install --no-cache-dir --no-build-isolation .

# Patch upstream casing bug: asc.language.tile.unary_ops.rms_norm calls
# `create_asctile_RmsnormOp` (lowercase n) but the IR builder exposes
# `create_asctile_RmsNormOp` (capital N). Fix in-place so asc2.rms_norm works.
RUN PYASC_SITE=$(python3 -c "import asc, os; print(os.path.dirname(asc.__file__))") \
 && sed -i 's/create_asctile_RmsnormOp/create_asctile_RmsNormOp/g' \
        "${PYASC_SITE}/language/tile/unary_ops.py" \
 && grep -q "create_asctile_RmsNormOp" "${PYASC_SITE}/language/tile/unary_ops.py" \
 && echo "rms_norm casing patched"

RUN python3 -c "import numpy; print(f'numpy: {numpy.__version__}')" && \
    python3 -c "import torch; print(f'torch: {torch.__version__}')" && \
    python3 -c "import asc, asc2; print('asc, asc2: OK')" || \
    echo "WARN: import failed during build (expected if CANN runtime not fully initialized). Will work at container runtime."

WORKDIR /workspace
CMD ["/bin/bash"]
