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
#   export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
#   python3.11 -c "import asc, asc2; print('pyasc OK')"
#   opencode
#
# Build args:
#   PYASC_GIT_URL   pyasc source repository (default: upstream gitcode mirror)
#   PYASC_GIT_REF   git ref to check out (default: v2 branch via MR 85)
#   LLVM_ARCHIVE    prebuilt LLVM tarball URL used by the pyasc build

FROM ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11

ENV CANN_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=${CANN_HOME}
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH}

ARG PYASC_GIT_URL=https://gitcode.com/cann/pyasc.git
ARG PYASC_GIT_REF=refs/merge-requests/85/head
ARG LLVM_ARCHIVE=https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential ccache clang lld \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_INSTALL_PREFIX=/opt/llvm
RUN mkdir -p "${LLVM_INSTALL_PREFIX}" \
 && curl -fsSL "${LLVM_ARCHIVE}" | tar -xz --strip-components=1 -C "${LLVM_INSTALL_PREFIX}"

RUN pip install --no-cache-dir \
        "numpy<2.0" \
        "pytest>=7.0" \
        pyyaml \
        scipy \
        pybind11 \
        setuptools \
        setuptools_scm \
        wheel

ENV PYASC_SRC=/opt/pyasc
RUN git clone "${PYASC_GIT_URL}" "${PYASC_SRC}" \
 && cd "${PYASC_SRC}" \
 && git fetch origin "+${PYASC_GIT_REF}:v2" \
 && git checkout v2

ENV PYASC_SETUP_CLANG_LLD=1 \
    PYASC_SETUP_CCACHE=1
RUN cd "${PYASC_SRC}" \
 && pip install --no-cache-dir -r requirements-build.txt \
 && pip install --no-cache-dir --no-build-isolation .

RUN python3 -c "import numpy; print(f'numpy: {numpy.__version__}')" && \
    python3 -c "import asc, asc2; print('asc, asc2: OK')" || \
    echo "WARN: import failed during build (expected if CANN runtime not fully initialized). Will work at container runtime."

WORKDIR /workspace
CMD ["/bin/bash"]
