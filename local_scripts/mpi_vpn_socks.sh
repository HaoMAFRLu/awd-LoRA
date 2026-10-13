#!/usr/bin/env bash
# Keep the system VPN intact and use a userspace MPI tunnel for cluster SSH.
set -euo pipefail

mpi_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mpi_ocproxy="$mpi_repo_root/data/local_tools/ocproxy_1.60-1build2/usr/bin/ocproxy"
mpi_action="${1:-connect}"
if (( $# > 0 )); then
    shift
fi

mpi_usage() {
    printf 'Usage: bash %s [connect [OPENCONNECT_OPTIONS...] | ssh [REMOTE_COMMAND...] | --help]\n' "${BASH_SOURCE[0]}"
    printf 'Run connect in one terminal, complete MPI authentication, and leave it running.\n'
    printf 'Run ssh in another terminal. Neither command needs sudo.\n'
}

case "$mpi_action" in
    connect)
        if (( EUID == 0 )); then
            printf 'Run this userspace VPN without sudo.\n' >&2
            exit 1
        fi
        if [[ ! -x "$mpi_ocproxy" ]]; then
            printf 'The locally extracted ocproxy executable is missing: %s\n' "$mpi_ocproxy" >&2
            exit 1
        fi
        for mpi_tool in openconnect nc; do
            if ! command -v "$mpi_tool" >/dev/null; then
                printf 'Required command is missing: %s\n' "$mpi_tool" >&2
                exit 1
            fi
        done
        if nc -z -w 1 127.0.0.1 11080 >/dev/null 2>&1; then
            printf 'Port 127.0.0.1:11080 already has a listener; no new tunnel was started.\n' >&2
            exit 1
        fi
        # OpenConnect invokes this command through a shell. Quote its path.
        printf -v mpi_proxy_command '%q -D 127.0.0.1:11080' "$mpi_ocproxy"
        printf 'Starting the MPI SOCKS tunnel on 127.0.0.1:11080.\n'
        printf 'Complete MPI login below and leave this terminal running.\n'
        # This IP passed HTTPS certificate validation on 2026-10-06. Pin only
        # its DNS resolution, retaining hostname-based certificate validation.
        exec openconnect "$@" \
            --protocol=anyconnect --no-dtls --no-proxy \
            --resolve=vpngw.tuebingen.mpg.de:192.124.25.4 \
            --script-tun --script="$mpi_proxy_command" \
            https://vpngw.tuebingen.mpg.de/
        ;;
    ssh)
        if ! nc -z -w 1 127.0.0.1 11080 >/dev/null 2>&1; then
            printf 'No proxy is listening on 127.0.0.1:11080. Start connect first.\n' >&2
            exit 1
        fi
        exec ssh \
            -i "${HOME}/.ssh/mpi_cluster_ed25519" \
            -o IdentitiesOnly=yes \
            -o ConnectTimeout=20 \
            -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=5 \
            -o TCPKeepAlive=yes \
            -o 'ProxyCommand=nc -X 5 -x 127.0.0.1:11080 %h %p' \
            hma2@login.cluster.is.localnet "$@"
        ;;
    -h|--help|help)
        mpi_usage
        ;;
    *)
        mpi_usage >&2
        exit 2
        ;;
esac
