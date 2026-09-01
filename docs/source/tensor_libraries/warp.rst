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

.. role:: strong-underline
   :class: leapp-strong-underline

For any pipeline that traces Warp, :strong-underline:`Warp features require
at least two passes to capture correctly: a discovery pass, then a capture
pass of the same Warp control-flow path before` ``leapp.stop()``.

Each captured Warp region becomes a ``leapp::warp_runner`` operation in an
ONNX or PT2 artifact. Export Warp-containing nodes with
``export_with="onnx"`` or ``export_with="pt2"``. See
:doc:`warp_how_it_works` for the capture model and :doc:`warp_limitations`
for what it rules out.

Example: a simulated robot policy
=================================

The simulator yields joint state as Warp arrays, so conditioning stays in
Warp. The learned policy stays in torch.

.. code-block:: python

   import warp as wp
   import torch
   import leapp
   from leapp import annotate

   VEL_SCALE = 4.0

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

   def preprocess(joint_pos, joint_vel) -> torch.Tensor:
       pos, vel = annotate.input_tensors("preprocess", {
           "joint_pos": joint_pos,
           "joint_vel": joint_vel,
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
       annotate.output_tensors("policy", {"action": action}, export_with="onnx")
       return action

   def main(sim):
       policy = Policy().eval().cuda()

       leapp.start(name="warp_robot_pipeline")
       for _ in range(2):  # discovery pass, then capture pass
           pos, vel = sim.joint_state()
           obs = preprocess(pos, vel)
           run_policy(policy, obs)
       leapp.stop()
       leapp.compile_graph()

The maintained copy of this example is ``examples/warp_robot_pipeline.py``.

The loop runs the annotated path twice: once to discover Warp segments, once
to capture them. ``wp.to_torch()`` is a traced conversion, not a trace break.

Nothing separates the two launches, so both belong to the same segment. This
is the graph LEAPP records for ``preprocess``:

.. code-block:: text

   %joint_pos               = placeholder[target=joint_pos]
   %joint_vel               = placeholder[target=joint_vel]
   %warp_segment_0_bundle   = get_attr[target=_warp_segment_0_bundle]
   # Both wp.launch calls are condensed into this single node.
   %warp_segment_0          = call_function[target=torch.ops.leapp.warp_runner.default](args = ([%joint_pos, %joint_vel], {...boundary metadata...}, %warp_segment_0_bundle))
   %warp_segment_0_output_5 = call_function[target=operator.getitem](args = (%warp_segment_0, 5))
   %warp_segment_0_output_7 = call_function[target=operator.getitem](args = (%warp_segment_0, 7))
   %cat                     = call_function[target=torch.cat](args = ([%warp_segment_0_output_5, %warp_segment_0_output_7],))
   return cat

The kernels themselves are not in the graph. ``_warp_segment_0_bundle`` is the
captured APIC archive, the elided argument is the shape and dtype metadata for
the arrays crossing the boundary, and the two ``getitem`` calls select the
buffers the segment wrote. Only ``torch.cat``, which ran outside Warp, is
recorded as an ordinary torch operation.

.. note::

   Warp tracing requires ``leapp.start(..., global_patching=True)``, which is
   the default. Starting with ``global_patching=False`` disables the Warp
   session patches, and conversions then return untraced values.

.. toctree::
   :hidden:

   installation
   warp_how_it_works
   warp_limitations
