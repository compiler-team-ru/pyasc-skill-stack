# Overlay native extension (x86_64)

Drop a host-built `libpyasc.cpython-311-x86_64-linux-gnu.so` here when you want
to build the overlay image on an x86_64 Linux host.

The file is the native extension produced by a normal `pip install .` of `pyasc`
v2 on x86_64 Linux against CPython 3.11; it must be ABI-compatible with the
Python 3.11 inside the `ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11`
base image.

This `.so` is intentionally not tracked in git (see `.gitignore` rule
`docker/pyasc-overlay/asc_C/**/*.so`) — each developer supplies their own.

The Dockerfile selects this directory automatically when `uname -m` reports
`x86_64`; see `docker/Dockerfile.overlay`.
