=======
PyTorch
=======

PyTorch is the path LEAPP is built around. Tensors handed to
:func:`~leapp.annotate.input_tensors` come back as ``TracedTensor``, a
``torch.Tensor`` subclass that shares storage with the tensor you passed in,
and every torch operation applied to one is recorded into the graph that gets
exported.

Nothing is translated along the way. :doc:`NumPy <numpy>` tracing looks each
call up in a table of torch equivalents and :doc:`Warp <warp>` hands whole
regions to APIC capture, but a torch call is recorded as itself. See
:doc:`torch_how_it_works` for the interception model and
:doc:`torch_limitations` for what it cannot represent.

Example: a two-node policy pipeline
===================================

Normalization and the policy are separate nodes, so they can be exported with
different backends and deployed as separate models.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   STATE_MEAN = torch.zeros(6)
   STATE_STD = torch.full((6,), 0.5)

   class Policy(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.net = torch.nn.Linear(9, 6)

       def forward(self, obs: torch.Tensor) -> torch.Tensor:
           return torch.tanh(self.net(obs))

   def preprocess(state, velocity):
       state, velocity = annotate.input_tensors("preprocess", {
           "state": state,
           "velocity": velocity,
       })

       state_norm = torch.clamp((state - STATE_MEAN) / STATE_STD, -5.0, 5.0)
       obs = torch.cat([state_norm, velocity])

       annotate.output_tensors("preprocess", {"obs": obs}, export_with="jit")
       return obs

   def run_policy(policy, obs):
       traced_obs = annotate.input_tensors("policy", {"obs": obs})
       action = policy(traced_obs.unsqueeze(0)).squeeze(0)
       annotate.output_tensors("policy", {"action": action}, export_with="onnx")
       return action

   def main():
       policy = Policy().eval()

       leapp.start(name="torch_pipeline")
       obs = preprocess(torch.zeros(6), torch.zeros(3))
       run_policy(policy, obs)
       leapp.stop()
       leapp.compile_graph()

Each ``input_tensors`` and ``output_tensors`` pair marks one node. ``obs``
leaves ``preprocess`` as a published output and arrives at ``policy`` as an
input, which is what connects the two nodes in the exported pipeline.

This is the graph LEAPP records for ``preprocess``:

.. code-block:: text

   %state             = placeholder[target=state]
   %velocity          = placeholder[target=velocity]
   %_tensor_constant0 = get_attr[target=_tensor_constant0]
   %sub               = call_function[target=torch.sub](args = (%state, %_tensor_constant0))
   %_tensor_constant1 = get_attr[target=_tensor_constant1]
   %div               = call_function[target=torch.div](args = (%sub, %_tensor_constant1))
   %clamp             = call_function[target=torch.clamp](args = (%div, -5.0, 5.0))
   %cat               = call_function[target=torch.cat](args = ([%clamp, %velocity],))
   return cat

Every operation appears under its own torch name, and the two traced inputs
are placeholders. ``STATE_MEAN`` and ``STATE_STD`` never passed through
``input_tensors()``, so they are frozen into the graph as ``get_attr``
constants instead.

.. toctree::
   :hidden:

   torch_how_it_works
   torch_limitations
