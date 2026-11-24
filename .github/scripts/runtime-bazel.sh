#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
runtime_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../packages/leapp-rt" && pwd)
task_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/leapp-bazel-XXXXXX")
cleanup() {
  (cd "$runtime_root" && bazel --output_user_root="$task_dir/cache" shutdown) || true
  if [[ -d "$task_dir/consumer" ]]; then
    (cd "$task_dir/consumer" && bazel --output_user_root="$task_dir/cache" shutdown) || true
  fi
  # Bazel makes generated tree-artifact directories read-only.
  find "$task_dir" -type d -exec chmod u+w {} +
  rm -rf "$task_dir"
}
trap cleanup EXIT

cd "$runtime_root"
buildifier -mode=check MODULE.bazel BUILD.bazel bazel/{BUILD.bazel,defs.bzl} \
  bindings/{common,cxx}/BUILD.bazel tests/bazel/{MODULE,BUILD}.bazel
bazel --output_user_root="$task_dir/cache" test //... \
  --symlink_prefix=/ --test_output=errors

# This root module has no LEAPP setup macros, toolchain registration, or bazelrc.
# Archive only the module's source files, so it cannot reach the checkout's
# parent directories, ignored build outputs, or pre-existing environments.
git -C "$runtime_root" ls-files --cached --others --exclude-standard -z |
  tar -C "$runtime_root" --null -T - -czf "$task_dir/runtime.tar.gz"
cp -R tests/bazel "$task_dir/consumer"
printf '\narchive_override(module_name = "leapp_rt", urls = ["file://%s/runtime.tar.gz"])\n' \
  "$task_dir" >> "$task_dir/consumer/MODULE.bazel"
cd "$task_dir/consumer"
bazel --output_user_root="$task_dir/cache" test //... \
  --symlink_prefix=/ --test_output=errors
