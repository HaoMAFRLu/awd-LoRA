#!/usr/bin/env bash
set -euo pipefail

HOST="hma2@login.cluster.is.localnet"

exec ssh \
  -i "${HOME}/.ssh/mpi_cluster_ed25519" \
  -o IdentitiesOnly=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=5 \
  -o TCPKeepAlive=yes \
  "$HOST" \
  "$@"
