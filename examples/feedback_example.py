"""Feedback connection example for LEAPP.

This example demonstrates inferred graph-level feedback by running the same
two-node pipeline twice in one trace session.

Pipeline:
  [policy_step] -> [feedback_update]
        ^               |
        |_______________|

Run:
    python examples/feedback_example.py
"""

import os
import torch
import leapp
from leapp import annotate

GRAPH_NAME = "sample_feedback_graph"


def mix_with_feedback(data: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
    """Small helper to make the first node a bit more substantial."""
    centered = data - 0.5
    return torch.tanh(centered + 0.25 * feedback)


def compute_action(hidden: torch.Tensor) -> torch.Tensor:
    return torch.clamp(hidden * 2.0, min=-1.0, max=1.0)


def blend_feedback(hidden: torch.Tensor, previous_feedback: torch.Tensor) -> torch.Tensor:
    return 0.8 * previous_feedback + 0.2 * hidden


def main():
    leapp.start(name=GRAPH_NAME)

    policy_memory = torch.tensor([0.0])

    for step in range(2):  # needed for inferred cross-node feedback detection
        policy_inputs = annotate.input_tensors("policy_step", {
            "observation_scalar": torch.tensor([1.0 + step]),
            "policy_memory_in": policy_memory,
        })
        policy_context = mix_with_feedback(policy_inputs[0], policy_inputs[1])
        control_action = compute_action(policy_context)
        annotate.output_tensors(
            "policy_step",
            {"policy_context": policy_context, "control_action": control_action},
            export_with="jit",
        )

        feedback_inputs = annotate.input_tensors("feedback_update", {
            "policy_context": policy_context,
            "policy_memory_prev": policy_memory,
        })
        policy_memory = blend_feedback(feedback_inputs[0], feedback_inputs[1])
        annotate.output_tensors(
            "feedback_update",
            {"policy_memory_out": policy_memory},
            export_with="jit",
        )

    leapp.stop()
    leapp.compile_graph(visualize=True)

    out_dir = GRAPH_NAME
    print(f"\nExported to {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
