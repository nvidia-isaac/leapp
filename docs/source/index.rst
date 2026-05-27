=====
LEAPP
=====

**Lightweight Export Annotations for Policy Pipelines**

LEAPP is a Python package for tracing and exporting multi-step PyTorch
computational graphs. Annotate your existing code with lightweight markers,
and LEAPP captures the graph structure, exports each stage as an individual
model, and generates a deployment-ready YAML specification.

What is LEAPP?
==============

LEAPP is designed for pipelines that chain multiple PyTorch models or
processing stages together --- where you need to export the whole pipeline,
not just a single model.

LEAPP works by tracing real execution of your code. It records which tensors
flow between stages, exports each stage independently, and writes a YAML that
describes how to wire them back together at inference time. Unlike other
solutions, LEAPP is non-invasive and eliminates the need for a separate
export-specific implementation of your processing.

Features
========

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Multiple export formats
      :class-card: sd-rounded-3

      TorchScript and ONNX, with multiple backend variants per format.

   .. grid-item-card:: Flexible structuring
      :class-card: sd-rounded-3

      Define complex node boundaries or multi-node graph structures
      with a small set of annotations.

   .. grid-item-card:: Lightweight
      :class-card: sd-rounded-3

      Minimal insertions --- no rewrites of existing model code required.
      Annotations safely no-op when not exporting.

   .. grid-item-card:: YAML specification
      :class-card: sd-rounded-3

      Complete metadata for deployment and downstream frameworks.

   .. grid-item-card:: BYOM
      :class-card: sd-rounded-3

      Bring Your Own Model --- integrate pre-compiled models into the graph
      without recompiling.

   .. grid-item-card:: Graph visualization
      :class-card: sd-rounded-3

      Automatic diagrams of your pipeline.


Get started
===========

* Install LEAPP and walk through your first pipeline in
  :doc:`getting_started`.
* Dig into specific workflows in :doc:`guides/index` --- node patterns,
  export options, feedback graphs, semantic data, and runtime validation.
* Browse the full :doc:`api/index`.

Installation
============

.. code-block:: bash

   pip install leapp


.. toctree::
   :caption: Contents
   :hidden:
   :maxdepth: 2

   getting_started
   guides/index
   api/index
