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

      Distributed inputs, explicit recurrent state, module buffer tracking,
      static outputs, and nested data structures.

   .. grid-item-card:: Export configuration
      :link: export
      :link-type: doc
      :class-card: sd-rounded-3

      Backend selection, TorchScript vs ONNX, dry-run modes, and bringing
      pre-compiled models into the graph.

   .. grid-item-card:: Graph operations
      :link: graph
      :link-type: doc
      :class-card: sd-rounded-3

      Cycle detection, feedback connections, and preserving tracing
      continuity with ``mirror_leapp_tags``.

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
   runtime
   semantics
