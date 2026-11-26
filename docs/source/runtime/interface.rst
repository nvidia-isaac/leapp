Interface
=========

The runtime separates bundle inspection, session execution, and tensor ownership.
A ``Bundle`` is an in-memory representation of the LEAPP YAML, useful for
inspection without parsing YAML yourself. A ``Session`` loads the models for
execution and owns one sequence of feedback state. Tensor wrappers borrow caller storage or
retain result allocations through DLPack.

All three language interfaces use the same qualified ``node/port`` names and
synchronous completion contract. Inspect a bundle before allocating buffers,
open a session once, and reuse it for successive inference calls.

.. toctree::
   :maxdepth: 1

   bundle
   session_functions
   memory
