# pyasc-sim: CANN 9.0 + Python 3.11 + pyasc (with asc2) for simulator verification.
#
# Build:
#   docker build -f docker/Dockerfile -t pyasc-sim:py3.11 docker/
#
# Usage:
#   docker run --rm -v "$(pwd)":/workspace -w /workspace \
#     pyasc-sim:py3.11 \
#     python3.11 tests/tools/run_and_verify.py golden/kernels/abs_f16.py --mode simulator
#
# Push to ghcr.io:
#   docker tag pyasc-sim:py3.11 ghcr.io/aloschilov/pyasc-sim:py3.11
#   docker push ghcr.io/aloschilov/pyasc-sim:py3.11

FROM ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH}

RUN pip install --no-cache-dir \
    "pyasc>=1.1.0" \
    "numpy<2.0" \
    "pytest>=7.0" \
    pyyaml \
    scipy

# asc2 and asc.language.tile are from the pyasc dev branch (not yet in PyPI).
# The native extension must also be replaced with the dev build that includes
# TileLocation and other asc2 IR types.
ARG SITE_PKG=/usr/local/python3.11.15/lib/python3.11/site-packages
COPY pyasc-overlay/asc_C/libpyasc.cpython-311-x86_64-linux-gnu.so ${SITE_PKG}/asc/_C/libpyasc.cpython-311-x86_64-linux-gnu.so
COPY pyasc-overlay/asc2/ ${SITE_PKG}/asc2/
COPY pyasc-overlay/asc_language_tile/ ${SITE_PKG}/asc/language/tile/
COPY pyasc-overlay/asc_changed/codegen/ ${SITE_PKG}/asc/codegen/
COPY pyasc-overlay/asc_changed/language/core/ ${SITE_PKG}/asc/language/core/
COPY pyasc-overlay/asc_changed/lib/runtime/ ${SITE_PKG}/asc/lib/runtime/
COPY pyasc-overlay/asc_changed/runtime/ ${SITE_PKG}/asc/runtime/

RUN python3 -c "import numpy; print(f'numpy: {numpy.__version__}')" && \
    python3 -c "import asc2; print('asc2: OK')" || \
    echo "WARN: import failed during build (expected if CANN runtime not fully initialized). Will work at container runtime."

WORKDIR /workspace
CMD ["/bin/bash"]
