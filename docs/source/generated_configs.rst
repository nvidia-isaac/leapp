===================================
Understanding the Generated Configs
===================================

LEAPP is designed so generated configs and model artifacts are transparent:
you can inspect every file and implement your own runtime when needed.
``leapp.compile_graph()`` writes a directory containing one YAML file, one
compiled model artifact per exported node, optional feedback initial values,
and an optional PNG visualization. The YAML is the runtime contract: it tells a
deployment system which models to load, which external values to provide, how
node outputs feed later node inputs, and how robot-facing tensors should be
named and interpreted.

.. note::

   Understanding this section is only needed for debugging and implementing
   your own runtime. To run an exported bundle, use the
   :doc:`leapp_runtime` page.

Bundle layout
=============

For ``leapp.start(name="sample_pipeline")``, the output directory looks like:

.. code-block:: text

   sample_pipeline/
     sample_pipeline.yaml
     obs_processor.pt
     policy.pt
     sample_pipeline.png
     sample_pipeline_initial_values.safetensors

The PNG is only present when visualization is enabled and supported. The
``*_initial_values.safetensors`` file is only present when the graph has
feedback state.

YAML structure
==============

.. code-block:: yaml

   models:
     obs_processor:
       inputs:
       - name: joint_pos
         dtype: float32
         shape: [6]
         type: tensor
         kind: state/joint/position
         element_names: [left_hip, left_knee, left_ankle, right_hip,
           right_knee, right_ankle]
       outputs:
       - name: obs_features
         dtype: float32
         shape: [18]
         type: tensor
       parameters:
         backend: jit
         model_path: obs_processor.pt
         sha256sum: ...
     policy:
       inputs:
       - name: obs_features
         dtype: float32
         shape: [18]
         type: tensor
       outputs:
       - name: joint_targets
         dtype: float32
         shape: [6]
         type: tensor
         kind: target/joint/position
         element_names: [left_hip, left_knee, left_ankle, right_hip,
           right_knee, right_ankle]
       parameters:
         backend: jit
         model_path: policy.pt

   pipeline:
     data_flow:
       obs_processor/obs_features: [policy/obs_features]
     feedback_flow: {}
     inputs:
       obs_processor: [joint_pos, joint_vel, orientation, cmd_vel]
     outputs:
       policy: [joint_targets]
     configs:
       frequency: 50

   system information:
     leapp version: 0.6.0
     leapp config version: '1.3'
     torch version: 2.9.1+cu128
     python version: 3.12.9
     cuda version: '12.8'
     os: Ubuntu 24.04

Field Reference
===============

``models``
----------

Per-node model contracts. Each key is a node name; each value identifies the
compiled artifact for that node and describes the tensor ports that the runtime
must provide and consume.

The ``models`` mapping is deterministically ordered. LEAPP guarantees:

* Entries appear in the order nodes were finalized during tracing
  (typically ``output_tensors()``, or when an annotated method returns).
* A runtime that executes nodes in this order runs producers before
  consumers for forward ``pipeline.data_flow`` edges. Cyclic dependencies
  are recorded separately in ``pipeline.feedback_flow``.

``models.<node>.inputs`` and ``models.<node>.outputs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   These entries define the tensor ports for a node. Required fields are
   ``name``, ``dtype``, ``shape``, and ``type``. A deployment library should
   use these fields to allocate buffers, validate incoming values, and preserve
   backend input and output ordering.

   Both lists are deterministically ordered. The order of ``inputs`` is the
   order the compiled model expects for its positional arguments, and the
   order of ``outputs`` is the order the model returns values.

``models.<node>.parameters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ``parameters`` contains the export metadata needed to load the node
   artifact.

   ``backend``
      Selects the runtime loader. For example, ``jit`` loads a TorchScript
      ``.pt`` file, ``pt2`` loads a ``torch.export`` ``.pt2`` file, and ONNX
      backends load ``.onnx`` files.

   ``model_path``
      Names the compiled artifact relative to the YAML file.

   ``md5sum`` and ``sha256sum``
      Hashes of the compiled artifact LEAPP wrote. A deployment library can
      use these fields to verify that the model file has not changed since
      export.

``pipeline``
------------

The pipeline section describes graph wiring and the external runtime boundary.
A deployment library uses it to decide which values must come from outside the
LEAPP graph, how node outputs feed later node inputs, and which outputs should
be returned or published.

``pipeline.data_flow``
~~~~~~~~~~~~~~~~~~~~~~

   Forward edges. Keys are ``node/output`` ports and values are lists of
   ``node/input`` ports. A deployment library should copy or bind each source
   output to every listed target input in dependency order.

``pipeline.feedback_flow``
~~~~~~~~~~~~~~~~~~~~~~~~~~

   State edges applied across ticks. Treat these like ``data_flow`` after a
   node runs, but feed their values into the next graph invocation rather than
   the current one. Before the first graph invocation, initialize feedback input
   buffers from the safetensors file named by ``pipeline.initial_values``.

``pipeline.inputs`` and ``pipeline.outputs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The external graph boundary. Inputs must be supplied by the embedding
   runtime; outputs are the values the runtime should publish, return, or pass
   to the surrounding controller.

``pipeline.initial_values``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Optional safetensors filename containing first-tick values for feedback
   inputs. Keys match target ports such as ``policy/hidden``.

``pipeline.configs``
~~~~~~~~~~~~~~~~~~~~

   Optional graph-level metadata from ``GraphConfigs``, such as execution
   ``frequency`` in Hz and project-specific runtime fields.

``system information``
----------------------

Export provenance. Use this for debugging compatibility, support reports, and
reproducibility. A deployment library should not need these fields for normal
graph execution unless it intentionally gates behavior on LEAPP, PyTorch,
Python, CUDA, or OS versions.

Semantic Fields
===============

Semantic fields are optional tensor-port fields. A deployment library can use
these fields to connect raw tensors to robot concepts, message formats, topic
names, controller APIs, coordinate frames, or UI labels.

``kind``
--------

   Describes the physical role of a tensor, such as ``state/joint/position``
   or ``target/joint/position``. A deployment library can use it to route
   values to the right subscriber, publisher, command interface, or adapter
   without relying only on tensor names. See
   :doc:`semantics/kind_element_names`.

   .. code-block:: yaml

      models:
        policy:
          inputs:
          - name: joint_pos
            dtype: float32
            shape: [6]
            type: tensor
            kind: state/joint/position
          outputs:
          - name: joint_targets
            dtype: float32
            shape: [6]
            type: tensor
            kind: target/joint/position

``element_names``
-----------------

   Describes the order and identity of elements along tensor axes, such as
   joint names or vector components. A deployment library can use it to reorder
   robot messages into model order, verify that expected joints are present,
   and label outputs. See :doc:`semantics/kind_element_names`.

   .. code-block:: yaml

      models:
        policy:
          inputs:
          - name: joint_pos
            dtype: float32
            shape: [6]
            type: tensor
            element_names: [[left_hip, left_knee, left_ankle,
              right_hip, right_knee, right_ankle]]
          outputs:
          - name: joint_targets
            dtype: float32
            shape: [6]
            type: tensor
            element_names: [[left_hip, left_knee, left_ankle,
              right_hip, right_knee, right_ankle]]

``__temporal_axis__`` and ``temporal_period_ms``
------------------------------------------------

   Mark one tensor axis as time-like and record sample spacing in milliseconds.
   A deployment library can use these fields to interpret action chunks,
   trajectories, history windows, or other batched time samples. See
   :doc:`semantics/temporal`.

   .. code-block:: yaml

      models:
        policy:
          outputs:
          - name: actions
            dtype: float32
            shape: [4, 3]
            type: tensor
            element_names:
            - __temporal_axis__
            - [hip, knee, ankle]
            temporal_period_ms: 100
