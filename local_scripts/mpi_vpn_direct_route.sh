#!/usr/bin/env bash
# Temporary host routing for the current Wi-Fi network; leave Cisco connected.
# Requires a VPN policy that permits this route. Cisco can remove it immediately.
set -euo pipefail

mpi_gateway=192.124.25.4
mpi_hostname=vpngw.tuebingen.mpg.de
mpi_wifi_gateway=192.168.71.1
mpi_wifi_interface=wlp0s20f3
mpi_route=("$mpi_gateway/32" via "$mpi_wifi_gateway" dev "$mpi_wifi_interface")

mpi_remove_route() {
    local mpi_existing mpi_delete_error
    mpi_existing="$(ip -4 route show exact "$mpi_gateway/32")" || return 1
    if [[ -z "$mpi_existing" ]]; then
        printf 'The temporary MPI host route is already absent; no cleanup is needed.\n'
        return 0
    fi
    if mpi_delete_error="$(ip -4 route del "${mpi_route[@]}" 2>&1)"; then
        printf 'Removed the temporary MPI gateway host route.\n'
        return 0
    fi
    # A VPN agent may remove the route between the query and the delete.
    mpi_existing="$(ip -4 route show exact "$mpi_gateway/32")" || return 1
    if [[ -z "$mpi_existing" ]]; then
        printf 'Another network service already removed the temporary MPI host route.\n'
        return 0
    fi
    printf 'The requested route could not be removed: %s\n' "$mpi_delete_error" >&2
    return 1
}

if (( EUID != 0 )); then
    printf 'Run this script with sudo in your own terminal.\n' >&2
    exit 1
fi
if [[ "${1:-}" == --restore && $# == 1 ]]; then
    mpi_remove_route
    exit 0
fi
if (( $# != 0 )); then
    printf 'Usage: sudo bash %s [--restore]\n' "$0" >&2
    exit 2
fi

# Refuse to replace an existing host route or operate on a different network.
if [[ -n "$(ip -4 route show exact "$mpi_gateway/32")" ]]; then
    printf 'An MPI gateway host route already exists; no changes made.\n' >&2
    exit 1
fi
if [[ "$(ip -4 route show default dev "$mpi_wifi_interface")" != *"via $mpi_wifi_gateway "* ]]; then
    printf 'The Wi-Fi gateway changed; no changes made.\n' >&2
    exit 1
fi
if [[ -z "$(ip -4 route show default dev cscotun0)" ]]; then
    printf 'The expected Cisco VPN default route is absent; no changes made.\n' >&2
    exit 1
fi

mpi_added=0
mpi_keep=0
mpi_cleanup() {
    if (( mpi_added && ! mpi_keep )); then
        if ! mpi_remove_route; then
            printf 'Could not remove the temporary route; inspect: ip -4 route get %s\n' "$mpi_gateway" >&2
        fi
    fi
}
trap mpi_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ip -4 route add "${mpi_route[@]}"
mpi_added=1
ip -4 route get "$mpi_gateway"
if [[ "$(ip -4 route get "$mpi_gateway")" != *"via $mpi_wifi_gateway dev $mpi_wifi_interface "* ]]; then
    printf 'The MPI host route was removed or overridden; the Wi-Fi route is not active.\n' >&2
    printf 'Check the VPN routing policy before trying again.\n' >&2
    exit 1
fi

# No credentials are sent; certificate validation remains enabled.
if ! curl -4 --interface "$mpi_wifi_interface" --noproxy '*' \
    --resolve "$mpi_hostname:443:$mpi_gateway" \
    --connect-timeout 10 --max-time 15 --silent --show-error --head \
    --output /dev/null "https://$mpi_hostname/"; then
    printf 'MPI HTTPS is unreachable directly over Wi-Fi.\n' >&2
    exit 1
fi
if [[ "$(ip -4 route get "$mpi_gateway")" != *"via $mpi_wifi_gateway dev $mpi_wifi_interface "* ]]; then
    printf 'Another service changed the MPI gateway route during the test.\n' >&2
    exit 1
fi
mpi_keep=1
printf 'MPI gateway HTTPS works directly over Wi-Fi. The temporary host route is retained.\n'

# Reconnect only the single OpenConnect process connected to this MPI gateway.
mapfile -t mpi_processes < <(pgrep -x openconnect || true)
if (( ${#mpi_processes[@]} == 1 )) && \
    ss -Htnp dst "$mpi_gateway" | grep -Fq "pid=${mpi_processes[0]},"; then
    if kill -USR2 "${mpi_processes[0]}"; then
        printf 'Requested MPI OpenConnect reconnection; Cisco was left running.\n'
    else
        printf 'Reconnect MPI OpenConnect in its original terminal.\n' >&2
    fi
else
    printf 'Reconnect MPI OpenConnect in its original terminal.\n'
fi
printf 'Check the MPI terminal and retry cluster SSH after reconnection.\n'
printf 'To undo this host route: sudo bash %q --restore\n' "$0"
