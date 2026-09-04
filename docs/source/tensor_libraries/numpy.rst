=====
NumPy
=====

LEAPP traces NumPy code. Arrays handed to :func:`~leapp.annotate.input_tensors`
come back as ``TracedNpArray``, and the NumPy operations applied to them are
recorded into the same graph as the surrounding PyTorch code. A pipeline can
keep the NumPy preprocessing it already has instead of rewriting it in torch
before it can be exported.

The exported artifact is always a torch graph. NumPy is a frontend to that
graph, not a second export backend. See :doc:`how_it_works` for the
interception model and :doc:`limitations` for the cases that follow from that.

Example: a dataset-driven policy pipeline
=========================================

Datasets in the LeRobot format yield frames as NumPy arrays, so normalization
runs in NumPy and only the policy itself needs torch. LEAPP traces straight
through the handover.

.. code-block:: python

   import numpy as np
   import torch
   import leapp
   from leapp import annotate

   # Normalization statistics that ship with the dataset.
   STATE_MEAN = np.zeros(6, dtype=np.float32)
   STATE_STD = np.full(6, 0.5, dtype=np.float32)

   class Policy(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.net = torch.nn.Linear(9, 6)

       def forward(self, obs: torch.Tensor) -> torch.Tensor:
           return torch.tanh(self.net(obs))

   def preprocess(frame: dict) -> np.ndarray:
       """Runs on the NumPy arrays the dataset yields."""
       state, velocity = annotate.input_tensors("preprocess", {
           "state": frame["observation.state"],
           "velocity": frame["observation.velocity"],
       })

       state_norm = np.clip((state - STATE_MEAN) / STATE_STD, -5.0, 5.0)
       obs = np.concatenate([state_norm, velocity])

       annotate.output_tensors("preprocess", {"obs": obs}, export_with="jit")
       return obs

   def run_policy(policy: Policy, obs: np.ndarray) -> np.ndarray:
       traced_obs = annotate.input_tensors("policy", {"obs": obs})

       action = policy(torch.from_numpy(traced_obs).unsqueeze(0))
       action = action.squeeze(0).numpy()

       annotate.output_tensors("policy", {"action": action}, export_with="onnx")
       return action

   def main():
       policy = Policy().eval()
       frame = {
           "observation.state": np.zeros(6, dtype=np.float32),
           "observation.velocity": np.zeros(3, dtype=np.float32),
       }

       leapp.start(name="lerobot_pipeline")
       obs = preprocess(frame)
       run_policy(policy, obs)
       leapp.stop()
       leapp.compile_graph()

``torch.from_numpy()`` and ``.numpy()`` are traced conversions, not trace
breaks: they keep the recorded chain intact and, when shape and dtype are
unchanged, they also keep the node boundary that connects ``preprocess`` to
``policy``. The exported bundle wires the two nodes together:

.. code-block:: yaml

   pipeline:
     data_flow:
       preprocess/obs: [policy/obs]
     inputs:
       preprocess: [state, velocity]
     outputs:
       policy: [action]

.. note::

   Conversion between NumPy and torch relies on the patches LEAPP installs for
   the duration of a tracing session. Starting with
   ``leapp.start(..., global_patching=False)`` disables them, and conversions
   then return untraced values.

.. toctree::
   :hidden:

   how_it_works
   limitations
