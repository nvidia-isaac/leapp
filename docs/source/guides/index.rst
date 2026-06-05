======
Guides
======

These guides cover the working surfaces of LEAPP in more depth. Read them
in order or jump to the topic you need.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Node patterns
      :link: nodes
      :link-type: doc
      :class-card: sd-rounded-3

      Distributed inputs, method shorthand, and nested data structures.

   .. grid-item-card:: Export configuration
      :link: export
      :link-type: doc
      :class-card: sd-rounded-3

      Backend selection, TorchScript vs ONNX, and bringing pre-compiled
      models into the graph.

   .. grid-item-card:: State capture and feedback
      :link: graph
      :link-type: doc
      :class-card: sd-rounded-3

      Explicit state tensors, module buffer tracking, and automatic feedback
      detection for recurrent or re-entered graph state.

   .. grid-item-card:: Buffers and constant tensors
      :link: buffers
      :link-type: doc
      :class-card: sd-rounded-3

      Registered module buffers, static outputs, embedded constants, and
      manual tensor-copy tag preservation.

   .. grid-item-card:: Debugging
      :link: debugging
      :link-type: doc
      :class-card: sd-rounded-3

      Dry-run modes, selective non-traced nodes, verbose logs, and full
      FX graph inspection.

   .. grid-item-card:: Runtime and validation
      :link: runtime
      :link-type: doc
      :class-card: sd-rounded-3

      Per-node validation, cached re-entry examples, and the Python
      ``InferenceManager`` for end-to-end smoke tests.

   .. grid-item-card:: Semantic data annotation
      :link: semantics
      :link-type: doc
      :class-card: sd-rounded-3

      Use ``TensorSemantics`` to describe what tensors represent and name
      individual tensor elements.


.. toctree::
   :hidden:
   :maxdepth: 1

   nodes
   export
   graph
   buffers
   debugging
   runtime
   semantics
