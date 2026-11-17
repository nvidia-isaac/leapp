#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# All callers run on Linux x86_64. Keep release binaries pinned and verified.
set -euo pipefail
case "${1:-}" in
  pixi)
    url=https://github.com/prefix-dev/pixi/releases/download/v0.84.0/pixi-x86_64-unknown-linux-musl.tar.gz
    checksum=f09fcce2b8a2a11d2db5a63863a19ce6c52822083262ea6769c2d6464588b743
    strip_components=0
    ;;
  cargo-deny)
    url=https://github.com/EmbarkStudios/cargo-deny/releases/download/0.20.2/cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz
    checksum=9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f
    strip_components=1
    ;;
  *) echo "Usage: $0 {pixi|cargo-deny}" >&2; exit 2 ;;
esac

download=$(mktemp -d "$RUNNER_TEMP/leapp-tool-XXXXXX")
trap 'rm -rf "$download"' EXIT
curl --fail --location --silent --show-error --retry 3 "$url" -o "$download/tool.tar.gz"
printf '%s  %s\n' "$checksum" "$download/tool.tar.gz" | sha256sum --check
bin_dir="${LEAPP_CI_ROOT:-$RUNNER_TEMP/leapp-runtime-tools}/bin"
mkdir -p "$bin_dir"
tar -xzf "$download/tool.tar.gz" -C "$download" --strip-components="$strip_components"
install -m 755 "$download/$1" "$bin_dir/$1"
echo "$bin_dir" >> "$GITHUB_PATH"
"$bin_dir/$1" --version
