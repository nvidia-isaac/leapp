Bundle
======

A ``Bundle`` is an immutable, in-memory representation of a LEAPP YAML bundle.
It parses the YAML and exposes its model descriptions, input and output
contracts, and metadata through getters. Use it to inspect an exported bundle
without implementing YAML parsing yourself.
Opening a ``Bundle`` does not load models for execution or initialize inference
backends. A :doc:`Session <session_functions>` performs that work when opened.
Bundle inspection can be used on its own, without creating a session.

Opening still validates graph ports and connections, referenced artifacts,
optional SHA-256 hashes, and initial-value files, which can involve reading those
files. Keep model and initial-value files available until session construction
finishes; the bundle does not retain their bytes for later execution.

The runtime first looks for model files relative to the YAML file. If a file
is missing there, it tries the model path as written, resolving relative paths
from the application's working directory. Backend aliases are mapped to their
standard names. You can inspect bundles with custom backend names; the runtime
checks whether it supports them when you open a session. See
:doc:`../generated_configs` for details of the YAML format.

Inspect inputs and outputs
--------------------------

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import lrt

         # Bundle: parse the exported description without loading models.
         bundle = lrt.Bundle.open("sample_pipeline/sample_pipeline.yaml")

         # Inspect the public input contracts and model ports.
         for name, spec in bundle.pipeline_inputs().items():
             if spec is not None:
                 print(name, spec.dtype, spec.shape)
         print(bundle.model_io("control"))
         print(bundle.output_info("control", "joint_targets"))

   .. tab-item:: C++
      :sync: cpp

      With ``<leapp/bundle.h>`` and ``<iostream>`` included:

      .. code-block:: cpp

         // Bundle: parse the exported description without loading models.
         lrt::Bundle bundle;
         auto status = lrt::Bundle::Open("sample_pipeline/sample_pipeline.yaml", &bundle);
         if (!status.ok()) return status;

         // Inspect the public input contracts and model ports.
         for (const auto& [name, spec] : bundle.pipeline_inputs()) {
           if (!spec) {
             std::cout << name << ": descriptor unavailable\n";
             continue;
           }
           std::cout << name << ": [";
           for (std::size_t i = 0; i < spec->shape.size(); ++i) {
             if (i > 0) std::cout << ", ";
             std::cout << spec->shape[i];
           }
           std::cout << "]\n";
         }
         const auto* ports = bundle.model_io("control");
         const auto* targets = bundle.output_info("control", "joint_targets");

      Here ``spec`` is a ``std::optional<PortSpec>``, not a pointer. It is empty
      when a declared pipeline input does not resolve to a model port. Checking
      it before accessing ``shape`` handles that case. By comparison,
      ``input_info()`` and ``output_info()`` return pointers that can be ``nullptr``.

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         // Bundle: parse the exported description without loading models.
         let bundle = lrt::Bundle::open("sample_pipeline/sample_pipeline.yaml")?;

         // Inspect the public input contracts and model ports.
         for port in bundle.pipeline_inputs() {
             if let Some(spec) = bundle.input_info(&port.node, &port.port) {
                 println!("{}: {:?} {:?}", port.name(), spec.dtype, spec.shape);
             }
         }
         let ports = bundle.model_io("control");
         let targets = bundle.output_info("control", "joint_targets");


Getter reference
----------------

The getter names below apply to Python, C++, and Rust. Python uses method calls,
including ``pipeline_inputs()`` and ``metadata()``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Getter
     - Returned information
   * - ``yaml_path()``
     - Source YAML path
   * - ``model_names()``
     - Model names in declaration order
   * - ``pipeline_inputs()``, ``pipeline_outputs()``
     - Public ports in declaration order, using qualified names
   * - ``model_io(node)``
     - Unqualified input and output name lists, including internal ports
   * - ``input_info(node, port)``, ``output_info(node, port)``
     - Port descriptor: name, dtype, shape, tensor type, and metadata
   * - ``model_parameters(node)``
     - Exporter parameters, including original backend names and paths
   * - ``model_metadata(node)``
     - Model-level metadata
   * - ``pipeline_configs()``
     - Pipeline configuration values
   * - ``pipeline_metadata()``
     - Pipeline-level semantic metadata
   * - ``system_information()``
     - Export environment information
   * - ``metadata()``
     - Top-level metadata

``model_io`` returns inputs followed by outputs. It preserves declaration order
and duplicate names. A known model with no ports has two empty lists; a missing
model returns the language's missing-value representation. Port lookup returns
the first matching descriptor. A public input declaration can lack a resolved
descriptor during inspection; handle that case before allocating storage.

Language types and lifetimes
----------------------------

.. tab-set::

   .. tab-item:: Python
      :sync: python

      Getters return detached Python values. Public port getters return ordered
      dictionaries of qualified names to ``PortSpec`` values; an unresolved
      descriptor is ``None``. Missing model or port lookups return ``None``.
      ``model_io`` returns a pair of lists. Modifying returned values does not
      change the bundle. Metadata uses ``MetadataMapping.entries`` for
      unhashable or colliding mapping keys.

   .. tab-item:: C++
      :sync: cpp

      ``Bundle`` copies share the snapshot and cached inspection values. Getter
      references and pointers remain valid while a copy retains that snapshot.
      Public port maps are ordered vectors of name/optional-descriptor pairs;
      an unresolved descriptor is ``std::nullopt``. Missing model or port lookups
      return ``nullptr``. ``model_io`` returns a retained ``ModelIO`` pointer.
      ``Bundle::Open`` requires an empty destination and preserves it on failure.
      Metadata uses a typed ``Metadata`` variant; use ``Metadata::Find`` for
      string-key lookup.

   .. tab-item:: Rust
      :sync: rust

      Clones share the reference-counted immutable snapshot. Getter references
      borrow it. Public ports are ``PortRef`` slices; use ``input_info`` or
      ``output_info`` for descriptors. Missing lookups return ``None``.
      ``model_io`` returns ``Option<(Vec<&str>, Vec<&str>)>`` with borrowed names.
      Concurrent inspection is safe.

Metadata preserves ordinary YAML values, sequences, ordered mappings, and unknown
fields. Custom YAML tags are unsupported and fail during parsing. Declaration and merge
handling follow the bundle loader; do not convert arbitrary metadata mappings to
string-key dictionaries if their original key types matter.
