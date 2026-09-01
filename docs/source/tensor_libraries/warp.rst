====================
Warp (EXPERIMENTAL)
====================

.. warning::

   Warp support in LEAPP is experimental. Use it with caution.

.. warning::

   Warp features require additional installation. See
   :doc:`installation`.

LEAPP traces NVIDIA Warp work on arrays handed to
:func:`~leapp.annotate.input_tensors`. Those arrays come back as
``TracedWpArray``, and Warp launches applied to them are captured into the
same exported graph as the surrounding PyTorch code. A pipeline can keep
simulation-side conditioning in Warp instead of rewriting it in torch
before it can be exported.

For any pipeline that traces Warp,

.. container:: leapp-highlight

   Warp features require at least two passes to capture correctly: a
   discovery pass, then a capture pass of the same Warp control-flow path
   before ``leapp.stop()``.

Each captured Warp region becomes a ``leapp::warp_runner`` operation in an
ONNX or PT2 artifact. Export Warp-containing nodes with
``export_with="onnx"`` or ``export_with="pt2"``. See
:doc:`warp_how_it_works` for the capture model and :doc:`warp_limitations`
for what it rules out.

Example: a simulated robot policy
=================================

The simulator yields joint state as Warp arrays. Conditioning stays in Warp.
The learned policy stays in torch. Commands go back through Warp before they
are written to the sim.

.. code-block:: python

   import warp as wp
   import torch
   import leapp
   from leapp import annotate

   JOINT_LIMIT = 1.0
   VEL_SCALE = 4.0
   ACTION_SCALE = 0.25

   @wp.kernel
   def scale_and_clip(
       src: wp.array(dtype=wp.float32),
       dst: wp.array(dtype=wp.float32),
       scale: float,
       limit: float,
   ):
       i = wp.tid()
       dst[i] = wp.clamp(src[i] * scale, -limit, limit)

   class Policy(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.net = torch.nn.Linear(12, 6)

       def forward(self, obs: torch.Tensor) -> torch.Tensor:
           return torch.tanh(self.net(obs))

   def get_robot_state(sim):
       return {
           "joint_pos": sim.joint_positions,
           "joint_vel": sim.joint_velocities,
       }

   def set_robot_command(sim, command):
       sim.set_joint_targets(command)

   def preprocess(frame: dict) -> torch.Tensor:
       pos, vel = annotate.input_tensors("preprocess", {
           "joint_pos": frame["joint_pos"],
           "joint_vel": frame["joint_vel"],
       })

       pos_n = wp.empty_like(pos)
       vel_n = wp.empty_like(vel)
       wp.launch(scale_and_clip, dim=pos.size,
                 inputs=[pos, pos_n, 1.0, 5.0], device=pos.device)
       wp.launch(scale_and_clip, dim=vel.size,
                 inputs=[vel, vel_n, 1.0 / VEL_SCALE, 1.0], device=vel.device)

       obs = torch.cat([wp.to_torch(pos_n), wp.to_torch(vel_n)])
       annotate.output_tensors("preprocess", {"obs": obs}, export_with="onnx")
       return obs

   def run_policy(policy: Policy, obs: torch.Tensor) -> torch.Tensor:
       traced_obs = annotate.input_tensors("policy", {"obs": obs})
       action = policy(traced_obs.unsqueeze(0)).squeeze(0)
       annotate.output_tensors("policy", {"action": action}, export_with="jit")
       return action

   def postprocess(action: torch.Tensor):
       cmd_in = annotate.input_tensors(
           "postprocess", {"action": wp.from_torch(action)})
       command = wp.empty_like(cmd_in)
       wp.launch(scale_and_clip, dim=cmd_in.size,
                 inputs=[cmd_in, command, ACTION_SCALE, JOINT_LIMIT],
                 device=cmd_in.device)
       annotate.output_tensors(
           "postprocess", {"command": command}, export_with="onnx")
       return command

   def main(sim):
       policy = Policy().eval()

       leapp.start(name="warp_robot_pipeline")
       for _ in range(2):
           frame = get_robot_state(sim)
           obs = preprocess(frame)
           action = run_policy(policy, obs)
           command = postprocess(action)
           set_robot_command(sim, command)
       leapp.stop()
       leapp.compile_graph()

The loop runs the annotated path twice: once to discover Warp segments, once
to capture them. ``wp.to_torch()`` and ``wp.from_torch()`` are traced
conversions. The exported bundle wires the three nodes together:

.. code-block:: yaml

   pipeline:
     data_flow:
       preprocess/obs: [policy/obs]
       policy/action: [postprocess/action]
     inputs:
       preprocess: [joint_pos, joint_vel]
     outputs:
       postprocess: [command]

.. note::

   Warp tracing requires ``leapp.start(..., global_patching=True)``, which is
   the default. Starting with ``global_patching=False`` disables the Warp
   session patches, and conversions then return untraced values.

.. toctree::
   :hidden:

   installation
   warp_how_it_works
   warp_limitations
