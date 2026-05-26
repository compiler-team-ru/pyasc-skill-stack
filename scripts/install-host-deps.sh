#!/usr/bin/env bash
#
# install-host-deps.sh -- idempotent bootstrap for pyasc-skill-stack hosts.
#
# Checks whether the apt packages, Docker Engine, and the `opencode` CLI are
# present on this Ubuntu host. Prints OK / MISS per item, shows the exact
# commands it would run, and asks once for confirmation before doing anything.
#
# Flags:
#   -y, --yes           Skip the confirmation prompt (non-interactive / CI use).
#       --skip-docker   Do not install Docker Engine even if missing.
#       --skip-opencode Do not install the opencode CLI even if missing.
#   -h, --help          Show this help.
#
# Supported hosts: Ubuntu 22.04 / 24.04, on x86_64 or aarch64.
# Anything else is reported as unsupported and the script exits 0 without
# touching the system. The script intentionally does NOT install the CANN
# Toolkit (see docs/cann-setup.md) and does NOT manage pyenv-installed
# Pythons.

set -euo pipefail

ASSUME_YES=0
SKIP_DOCKER=0
SKIP_OPENCODE=0

APT_REQUIRED=(
    build-essential
    ca-certificates
    clang
    lld
    ccache
    curl
    git
    python3.10
    python3.10-venv
    python3-pip
)

usage() {
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes)        ASSUME_YES=1 ;;
        --skip-docker)   SKIP_DOCKER=1 ;;
        --skip-opencode) SKIP_OPENCODE=1 ;;
        -h|--help)       usage ; exit 0 ;;
        *) echo "Unknown argument: $1" >&2 ; usage >&2 ; exit 2 ;;
    esac
    shift
done

note()  { printf '  %s\n' "$*" ; }
ok()    { printf '  OK:   %s\n' "$*" ; }
miss()  { printf '  MISS: %s\n' "$*" ; }
hdr()   { printf '\n== %s ==\n' "$*" ; }

require_supported_arch() {
    case "$(uname -m)" in
        x86_64|aarch64) ;;
        *) echo "Unsupported architecture: $(uname -m). Only x86_64 and aarch64 are supported." >&2
           exit 0 ;;
    esac
}

require_supported_distro() {
    if [ ! -r /etc/os-release ]; then
        echo "Cannot read /etc/os-release; this script targets Ubuntu hosts." >&2
        exit 0
    fi
    . /etc/os-release
    if [ "${ID:-}" != "ubuntu" ]; then
        echo "Detected distro: ${PRETTY_NAME:-unknown}. This script targets Ubuntu; skipping." >&2
        exit 0
    fi
    case "${VERSION_ID:-}" in
        22.04|24.04) ;;
        *) echo "Detected Ubuntu ${VERSION_ID:-?}; only 22.04 and 24.04 are tested. Proceeding anyway." >&2 ;;
    esac
}

dpkg_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'
}

MISSING_APT=()
check_apt_packages() {
    hdr "apt packages"
    for pkg in "${APT_REQUIRED[@]}"; do
        if dpkg_installed "$pkg"; then
            ok "$pkg"
        else
            miss "$pkg"
            MISSING_APT+=("$pkg")
        fi
    done
}

DOCKER_NEEDED=0
check_docker() {
    hdr "Docker Engine"
    if [ "$SKIP_DOCKER" -eq 1 ]; then
        note "skipped (--skip-docker)"
        return
    fi
    if command -v docker >/dev/null 2>&1; then
        ok "docker ($(docker --version 2>/dev/null | head -1))"
    else
        miss "docker"
        DOCKER_NEEDED=1
    fi
}

OPENCODE_NEEDED=0
check_opencode() {
    hdr "opencode CLI"
    if [ "$SKIP_OPENCODE" -eq 1 ]; then
        note "skipped (--skip-opencode)"
        return
    fi
    if command -v opencode >/dev/null 2>&1; then
        ok "opencode ($(opencode --version 2>/dev/null | head -1))"
    elif [ -x "$HOME/.opencode/bin/opencode" ]; then
        ok "opencode ($HOME/.opencode/bin/opencode) -- not on PATH; add \$HOME/.opencode/bin to PATH"
    else
        miss "opencode"
        OPENCODE_NEEDED=1
    fi
}

show_plan() {
    hdr "Planned actions"
    if [ ${#MISSING_APT[@]} -gt 0 ]; then
        note "sudo apt-get update"
        note "sudo apt-get install -y ${MISSING_APT[*]}"
    fi
    if [ "$DOCKER_NEEDED" -eq 1 ]; then
        note "install Docker Engine from https://download.docker.com/linux/ubuntu (official repo)"
    fi
    if [ "$OPENCODE_NEEDED" -eq 1 ]; then
        note "curl -fsSL https://opencode.ai/install | bash"
    fi
}

prompt_or_yes() {
    if [ ${#MISSING_APT[@]} -eq 0 ] \
       && [ "$DOCKER_NEEDED" -eq 0 ] \
       && [ "$OPENCODE_NEEDED" -eq 0 ]; then
        printf '\nNothing to install. Host already meets prerequisites.\n'
        exit 0
    fi
    show_plan
    if [ "$ASSUME_YES" -eq 1 ]; then
        printf '\n--yes given; proceeding.\n'
        return
    fi
    printf '\nProceed with the above? [y/N] '
    read -r reply || reply=""
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Aborted." ; exit 1 ;;
    esac
}

install_apt_packages() {
    [ ${#MISSING_APT[@]} -gt 0 ] || return 0
    hdr "Installing apt packages"
    sudo apt-get update
    sudo apt-get install -y "${MISSING_APT[@]}"
}

install_docker() {
    [ "$DOCKER_NEEDED" -eq 1 ] || return 0
    hdr "Installing Docker Engine"
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    . /etc/os-release
    local arch
    arch="$(dpkg --print-architecture)"
    echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                            docker-buildx-plugin docker-compose-plugin
    note "to run docker without sudo, add yourself to the 'docker' group:"
    note "    sudo usermod -aG docker \"\$USER\" && newgrp docker"
}

install_opencode() {
    [ "$OPENCODE_NEEDED" -eq 1 ] || return 0
    hdr "Installing opencode CLI"
    curl -fsSL https://opencode.ai/install | bash
    if ! command -v opencode >/dev/null 2>&1; then
        note "opencode installed under \$HOME/.opencode/bin; add it to PATH, e.g.:"
        note "    echo 'export PATH=\"\$HOME/.opencode/bin:\$PATH\"' >> ~/.bashrc"
    fi
}

final_check() {
    hdr "Final state"
    local errors=0
    for pkg in "${APT_REQUIRED[@]}"; do
        if dpkg_installed "$pkg"; then ok "$pkg"; else miss "$pkg"; errors=$((errors + 1)); fi
    done
    if [ "$SKIP_DOCKER" -eq 0 ]; then
        if command -v docker >/dev/null 2>&1; then
            ok "docker"
        else
            miss "docker"; errors=$((errors + 1))
        fi
    fi
    if [ "$SKIP_OPENCODE" -eq 0 ]; then
        if command -v opencode >/dev/null 2>&1 || [ -x "$HOME/.opencode/bin/opencode" ]; then
            ok "opencode"
        else
            miss "opencode"; errors=$((errors + 1))
        fi
    fi
    if [ "$errors" -gt 0 ]; then
        printf '\n%d item(s) still missing. Review the output above.\n' "$errors" >&2
        exit 1
    fi
    printf '\nAll host prerequisites are present.\n'
}

main() {
    require_supported_arch
    require_supported_distro
    check_apt_packages
    check_docker
    check_opencode
    prompt_or_yes
    install_apt_packages
    install_docker
    install_opencode
    final_check
}

main "$@"
