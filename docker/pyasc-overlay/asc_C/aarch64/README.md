# Overlay native extension (aarch64)

Drop a host-built `libpyasc.cpython-311-aarch64-linux-gnu.so` here when you
want to build the overlay image on an aarch64 Linux host.

The file is the native extension produced by a normal `pip install .` of
`pyasc` v2 on aarch64 Linux against CPython 3.11; it must be ABI-compatible
with the Python 3.11 inside the
`ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11` base image (which is
multi-arch and ships an arm64 variant).

This `.so` is intentionally not tracked in git (see `.gitignore` rule
`docker/pyasc-overlay/asc_C/**/*.so`) — each developer supplies their own.

The Dockerfile selects this directory automatically when `uname -m` reports
`aarch64`; see `docker/Dockerfile.overlay`.
