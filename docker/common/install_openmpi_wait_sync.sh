#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENMPI_SOURCE_ARCHIVE=/opt/hpcx/sources/openmpi5-gitclone.tar.gz
OPENMPI_PREFIX=/opt/hpcx/ompi5
# Backport of OpenMPI v5.0.x commit d054029e8a9eb60f887a7c69e2c284d88152650b
# against HPC-X 2.50's OpenMPI v5.0.10rc2-gb99be7132e source archive.
OPENMPI_PATCH="${SCRIPT_DIR}/patches/openmpi/d054029e-request-add-wait-sync-memory-barriers.diff"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Skipping the OpenMPI wait-sync backport outside ARM64 images"
    exit 0
fi

active_openmpi_prefix="$(readlink -f /usr/local/mpi 2>/dev/null || true)"
if [[ "${active_openmpi_prefix}" != "${OPENMPI_PREFIX}" ]]; then
    echo "Skipping the OpenMPI 5 wait-sync backport: active prefix is ${active_openmpi_prefix:-unknown}"
    exit 0
fi

if [[ ! -f "${OPENMPI_SOURCE_ARCHIVE}" ]]; then
    echo "HPC-X OpenMPI source archive not found: ${OPENMPI_SOURCE_ARCHIVE}" >&2
    exit 1
fi

source_dir="$(mktemp -d /tmp/openmpi-wait-sync.XXXXXX)"
trap 'rm -rf "${source_dir}"' EXIT

tar -xzf "${OPENMPI_SOURCE_ARCHIVE}" --strip-components=1 -C "${source_dir}"
cd "${source_dir}"
if git apply --reverse --check --no-index "${OPENMPI_PATCH}"; then
    echo "OpenMPI source already contains the wait-sync barriers"
    exit 0
fi
git apply --check --no-index "${OPENMPI_PATCH}"
git apply --no-index "${OPENMPI_PATCH}"

unset PMIX_VERSION
./configure \
    --prefix="${OPENMPI_PREFIX}" \
    --with-libevent=internal \
    --enable-mpi1-compatibility \
    --without-xpmem \
    --with-cuda=/usr/local/cuda \
    --with-slurm \
    --with-platform=contrib/platform/mellanox/optimized \
    --with-hcoll=/opt/hpcx/hcoll \
    --with-ucx=/opt/hpcx/ucx \
    --with-ucc=/opt/hpcx/ucc
make -j"$(nproc)"
make install

"${OPENMPI_PREFIX}/bin/ompi_info" --version
