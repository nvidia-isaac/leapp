#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Auto Buffer Tracking: GRU Policy Export

Demonstrates LEAPP's zero-intrusion buffer tracking for stateful neural nets.
The model uses standard PyTorch ``register_buffer`` + reassignment — no LEAPP
imports or protocol methods are needed inside the model.

``annotate.module()`` registers the model for buffer tracking.  LEAPP detects
which buffers were mutated during forward and exports them as state tensor I/O
with automatic feedback connections.

Usage::

    python examples/stateful_gru_export.py

Important:
    Use ``export_with="onnx-torchscript"`` for models containing ``nn.GRU`` or
    ``nn.LSTM``. The default dynamo-based ONNX exporter decomposes RNN ops into
    primitives that produce invalid ONNX Slice nodes.
"""

import os
import shutil
import torch
import torch.nn as nn
import leapp
from leapp import annotate


# ── Model: pure PyTorch, no LEAPP awareness ─────────────────────────────────

class GRUPolicy(nn.Module):
    """Policy with GRU hidden state stored as a registered buffer.

    The forward pass reassigns ``self.h_state = h_out``, which LEAPP's
    buffer tracker detects as a state mutation.
    """

    def __init__(self, obs_dim=16, hidden_dim=32, action_dim=8):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=1, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, action_dim),
        )
        # Hidden state as registered buffer — standard PyTorch pattern
        self.register_buffer("h_state", torch.zeros(1, 1, hidden_dim))

    def forward(self, obs):
        gru_out, h_out = self.gru(obs.unsqueeze(0), self.h_state)
        self.h_state = h_out  # reassignment — detected by LEAPP
        return self.mlp(gru_out.squeeze(0))


# ── Export: generic call-site, works with any model that uses register_buffer ─

def main():
    output_dir = "stateful_gru_export"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    model = GRUPolicy()
    model.eval()
    obs = torch.randn(1, 16)

    # Start LEAPP tracing
    leapp.start("stateful_gru", save_path=output_dir)

    # Register regular inputs
    obs_traced = annotate.input_tensors("policy", {"obs": obs})

    # Register module: injects TracedTensors into buffers, mutations
    # are auto-detected when output_tensors() compiles the graph
    annotate.module("policy", model)
    action = model(obs_traced)

    # Output — state outputs (h_state_out) are added automatically
    annotate.output_tensors("policy", {"action": action},
                            export_with="onnx-torchscript")

    leapp.stop()
    leapp.compile_graph(visualize=False)

    # Verify the ONNX model
    onnx_path = os.path.join(output_dir, "stateful_gru", "policy.onnx")
    if os.path.exists(onnx_path):
        import onnx
        m = onnx.load(onnx_path)
        init_names = {i.name for i in m.graph.initializer}
        print("\nONNX model I/O:")
        print("  Inputs:")
        for inp in m.graph.input:
            if inp.name not in init_names:
                dims = [d.dim_value or d.dim_param
                        for d in inp.type.tensor_type.shape.dim]
                print(f"    {inp.name}: {dims}")
        print("  Outputs:")
        for out in m.graph.output:
            dims = [d.dim_value or d.dim_param
                    for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {dims}")

    # Clean up
    shutil.rmtree(output_dir)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
