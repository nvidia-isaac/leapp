#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Label joint tensors and export a timed chunk of position targets.

Run from the repository root::

    python examples/tensor_semantics.py
"""

from pathlib import Path

import torch
import yaml

import leapp
from leapp import GraphConfigs, InferenceManager, TemporalAxis, TensorSemantics, annotate
from leapp.utils.enums import InputKindEnum, OutputKindEnum


GRAPH_NAME = "tensor_semantics_example"
NODE_NAME = "joint_controller"
JOINT_NAMES = ["hip", "knee", "ankle"]
PERIOD_MS = 20


def main():
    leapp.start(name=GRAPH_NAME)
    # Label measured state and commands, and name the joints in tensor order.
    # These descriptions become metadata in the exported YAML.
    position, velocity, command = annotate.input_tensors(NODE_NAME, [
        TensorSemantics(
            "joint_pos", torch.tensor([0.1, -0.2, 0.3]),
            kind=InputKindEnum.JOINT_POSITION,
            element_names=JOINT_NAMES,
        ),
        TensorSemantics(
            "joint_vel", torch.tensor([0.2, 0.0, -0.1]),
            kind=InputKindEnum.JOINT_VELOCITY,
            element_names=JOINT_NAMES,
        ),
        TensorSemantics(
            "joint_command", torch.tensor([0.4, 0.0, -0.1]),
            kind=InputKindEnum.COMMAND_JOINT_POSITION,
            element_names=JOINT_NAMES,
        ),
    ])
    # Blend from the current position toward a velocity-adjusted command
    # to produce four successive sets of joint targets.
    target = command - 0.05 * velocity
    actions = torch.stack([
        position + (target - position) * fraction
        for fraction in (0.25, 0.5, 0.75, 1.0)
    ])
    # The [4, 3] output has time along the first axis and joints along the
    # second. TemporalAxis describes the nominal spacing between rows.
    annotate.output_tensors(
        NODE_NAME,
        TensorSemantics(
            "actions",
            actions,
            kind=OutputKindEnum.JOINT_POSITION,
            element_names=[TemporalAxis(period_ms=PERIOD_MS), JOINT_NAMES],
        ),
        export_with="jit",
    )
    leapp.stop()
    # Graph frequency describes intended calls per second, independently of
    # the spacing within an action chunk. Neither label schedules execution.
    leapp.compile_graph(
        graph_configs=GraphConfigs(frequency=50), visualize=False, validate=True,
    )

    # Inspect the labels that downstream consumers will find in the bundle.
    yaml_path = Path(GRAPH_NAME) / f"{GRAPH_NAME}.yaml"
    with yaml_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    description = config["models"][NODE_NAME]

    # Replay the computation; semantic labels do not change its arithmetic.
    manager = InferenceManager(str(yaml_path))
    inputs = manager.get_mock_input()
    for name, values in {
        "joint_pos": [0.1, -0.2, 0.3],
        "joint_vel": [0.2, 0.0, -0.1],
        "joint_command": [0.4, 0.0, -0.1],
    }.items():
        key = f"{NODE_NAME}/{name}"
        inputs[key] = inputs[key].new_tensor(values)
    output = manager.run_policy(inputs)[f"{NODE_NAME}/actions"]

    print("Input semantics:", description["inputs"])
    print("Output semantics:", description["outputs"])
    print("Graph frequency (Hz):", config["pipeline"]["configs"]["frequency"])
    print("Action chunk:", output)
    print("Nominal row offsets (ms):",
          [row * PERIOD_MS for row in range(output.shape[0])])


if __name__ == "__main__":
    main()
