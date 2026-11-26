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

String models use ordinary ``onnx`` or ``pt2`` entries. A string input keeps its
name and is a ``uint8[1, capacity + 8]`` port: an eight-byte little-endian byte
length followed by the UTF-8 bytes. See :ref:`string-capture`.

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
         kind: joint/position
         axes:
         - kind: element
           names: [left_hip, left_knee, left_ankle, right_hip,
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
         kind: joint/position
         axes:
         - kind: element
           names: [left_hip, left_knee, left_ankle, right_hip,
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
     leapp version: 0.8.1
     leapp config version: '1.5'
     torch version: 2.9.1+cu128
     python version: 3.12.9
     cuda version: '12.8'
     os: Ubuntu 24.04

.. _yaml-minimum-contract:

Required and optional fields
============================

The tables below describe the fields emitted by LEAPP. **Always
present** means the key exists; its value may still be empty or ``null``.
Application-specific metadata should not override built-in fields.

Minimum YAML structure
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Location
     - Fields
     - Presence and meaning
   * - Top level
     - ``models``, ``pipeline``, ``system information``
     - Always present.
   * - Each model
     - ``inputs``, ``outputs``, ``parameters``
     - Always present. Input/output lists may be empty.
   * - Each tensor entry
     - ``name``, ``dtype``, ``shape``, ``type``
     - Always present. These describe the port independently of semantic
       annotations.
   * - Model ``parameters``
     - ``backend``, ``model_path``, ``md5sum``, ``sha256sum``, ``warp_segments``
     - Always present. Artifact paths and hashes may be ``null``, such as
       during a dry run.
   * - Model ``parameters``
     - Additional backend metadata
     - Backend-dependent; consumers must not assume these fields exist.
   * - ``pipeline``
     - ``data_flow``, ``feedback_flow``, ``inputs``, ``outputs``
     - Always present; mappings may be empty.
   * - ``pipeline``
     - ``configs``
     - Present when ``GraphConfigs`` is supplied; may be ``{}``.
   * - ``pipeline.configs``
     - ``frequency``
     - Present when supplied. Its absence means no execution frequency was
       declared.
   * - ``pipeline``
     - ``initial_values``
     - Present when a feedback-initialization file was written.

Optional tensor metadata
------------------------

A consumer must not assume semantic metadata is present merely because a
tensor has a name or a particular shape.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Field
     - When present
     - Meaning when absent
   * - ``kind``
     - A quantity or custom kind was supplied.
     - The quantity is unspecified.
   * - ``axes``
     - Axis metadata was supplied.
     - The tensor layout has no declared axis metadata.
   * - ``axes[i]``
     - Each entry corresponds to one dimension. An entry may be ``null`` or
       ``{}``.
     - A ``null`` entry leaves that dimension unspecified; ``{}`` declares
       no axis fields. When ``axes`` is supplied, an entry is required for
       every dimension.
   * - ``axes[i].kind``
     - An axis kind was supplied.
     - The axis category is unspecified.
   * - ``axes[i].names``
     - Labels were supplied.
     - Entries along that dimension are unnamed.
   * - ``axes[i].period_ms``
     - Required for a ``TIME`` axis; otherwise omitted.
     - No sample period is declared.
   * - ``axes[i].variable``
     - The dimension accepts a range of sizes. See
       :ref:`config-variable-dimensions`.
     - The dimension is fixed at the size in ``shape``.
   * - ``is_setpoint``
     - Emitted when explicitly set to ``true`` or ``false``.
     - Unspecified (``None`` in Python); no setpoint purpose is declared.
   * - ``spatial``
     - Emitted for applicable built-in spatial quantities or explicitly
       supplied spatial fields.
     - No explicit spatial metadata; applicable spatial fields default to
       global. See :ref:`spatial-yaml-fields` for helper-specific output.
   * - ``element_names``, ``temporal_period_ms``
     - Supplied through legacy annotations.
     - No corresponding legacy labels or timing metadata.
   * - Application-specific fields
     - Supplied through ``extra``, flattened into the entry.
     - No application-specific default is implied.

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
      ``.pt`` file, ``pt2`` loads a ``torch.export`` ``.pt2`` file, ``pte``
      loads an ExecuTorch ``.pte`` file, and ONNX backends load ``.onnx``
      files.

   ``model_path``
      Names the compiled artifact relative to the YAML file.

   ``md5sum`` and ``sha256sum``
      Hashes of the compiled artifact LEAPP wrote. A deployment library can
      use these fields to verify that the model file has not changed since
      export.

   ``data_path`` and ``data_sha256sum``
      Present for a ``pte`` node whose program keeps data in a separate
      ``.ptd`` file. They give that file's name, relative to the ``.pte``
      file, and its hash. A deployment library passes the file to
      ExecuTorch with the program.

   ``delegate``
      For a ``pte`` node, the delegate the program was built for: ``xnnpack``
      (CPU) or ``cuda``. A ``cuda`` program runs on the first visible GPU. The
      ``.pte`` file itself decides what runs; this entry does not limit it.

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

.. warning::

   ``Kind``, ``AxisKind``, ``ExpressionFrame``, ``Axis``, the quantity helpers,
   and the ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` fields
   are experimental. They are imported directly from ``leapp``. Their contract
   may evolve in future releases.

Semantic fields are optional tensor-port fields. The example above uses the
quantity and axis representation. See :doc:`semantics/legacy` for the
``element_names`` representation.

``kind`` and ``is_setpoint``
----------------------------

``kind`` describes a quantity, such as ``joint/position`` or ``frame/twist``.
The ``inputs`` and ``outputs`` lists determine direction. An input with
``is_setpoint: true`` carries a desired value that a controller aims to reach
or maintain. Explicit ``true`` and ``false`` values are preserved; ``None``
(the default) omits the field. Outputs accept ``False`` or ``None`` in Python
but reject ``True``. See :doc:`semantics/usage` for setpoints and
:doc:`semantics/kind_element_names` for quantities.

``axes``
--------

The list contains one mapping or ``null`` per tensor dimension. Each mapping
can describe the axis kind and ordered names. Time axes also include sample
spacing in milliseconds. For example, a tensor with shape ``[4, 3]`` can use:

.. code-block:: yaml

   axes:
   - {kind: time, period_ms: 100}
   - {kind: element, names: [hip, knee, ankle]}

See :ref:`semantics-time-axes`. Axis periods are independent of graph frequency.

.. _config-variable-dimensions:

Variable dimensions
-------------------

A dimension declared with ``Axis(variable=...)``, or inferred from one, has a
``variable`` mapping in its ``axes`` entry. ``shape`` still records the size of
the captured sample, so the two must be read together:

.. code-block:: yaml

   inputs:
   - name: tokens
     dtype: float32
     shape: [2, 8]
     type: tensor
     axes:
     - null
     - variable:
         symbol: length
         min: 4
         max: 32
         multiple_of: 2
   outputs:
   - name: pairs
     dtype: float32
     shape: [2, 4]
     type: tensor
     axes:
     - null
     - variable:
         symbol: __leapp_derived_7061697273_7061697273_1
         min: 2
         max: 16
         expression:
           formula:
             op: floordiv
             args: [length, 2]
         roots:
           length: {min: 4, max: 32, multiple_of: 2}

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``symbol``
     - Name of the dimension. Every axis in the bundle with the same symbol
       must have the same size in one ``run_policy()`` call. Symbols starting
       with ``__leapp_`` are generated.
   * - ``min``, ``max``
     - Inclusive bounds on the size.
   * - ``multiple_of``
     - Present when the size must be a multiple of this value.
   * - ``expression``
     - Present on a derived dimension. Either ``terms`` (symbol to integer
       coefficient) with an integer ``offset``, or a ``formula`` tree whose
       nodes are integers, symbols, or ``{op, args}`` with ``op`` one of
       ``add``, ``mul``, ``floordiv``, ``mod``.
   * - ``roots``
     - Present with ``expression``. The declared symbols the expression
       depends on, with their bounds.

Variable dimensions were added in config version ``1.5``. A bundle that uses
them also writes a ``shape contract version`` under ``system information``:
``2`` when any dimension uses ``multiple_of`` or a ``formula``, otherwise ``1``.
Bundles without variable dimensions have no ``shape contract version``.

A consumer must validate inputs against these fields before running a model.
``InferenceManager`` does this on every call, and refuses to load a bundle whose
axes declare ``variable`` without a supported ``shape contract version``. LEAPP
releases that predate config ``1.5`` ignore these fields.

``spatial``
-----------

Spatial metadata identifies a reference frame and coordinate basis. For example:

.. code-block:: yaml

   kind: frame/twist
   spatial:
     reference: base
     expressed_in: {selector: self}

A literal frame name uses ``expressed_in: {frame: world}``; a selector uses
``{selector: self}`` or ``{selector: reference}``. Null or missing spatial
fields mean global. Fields apply according to the quantity; see
:doc:`semantics/spatial`. Metadata does not transform tensor values.
