=====
LEAPP
=====

.. raw:: html

   <header class="leapp-hero">
     <div class="leapp-hero-copy">
       <h1>LEAPP</h1>
       <p>Lightweight Export Annotations<br />for Policy Pipelines</p>
     </div>
     <div class="leapp-hero-action">
       <a class="leapp-hero-button sd-btn sd-btn-primary" href="https://github.com/nvidia-isaac/leapp" aria-label="LEAPP repository on GitHub">
         <img src="_static/images/icons/github.png" alt="" aria-hidden="true" />
         GitHub repository
       </a>
       <a class="leapp-hero-button sd-btn sd-btn-primary" href="_downloads/leapp-integrate.zip" download="leapp-integrate.zip" title="Download the skill and supporting guides as a ZIP">
         <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
           <path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5" />
         </svg>
         Download agent skill
       </a>
     </div>
   </header>

Export the whole policy pipeline, not just the model weights.

LEAPP exports the policy code you already run into ONNX, TorchScript, or
PyTorch ExportedProgram models. Capture preprocessing, model inference, and
action processing together in **one model**, even when the computation spans
helper functions, class methods, and supported PyTorch and NumPy operations.
Choose separate models only when your deployment benefits from independent
stages.

LEAPP is designed for the full policy path, including the logic that
surrounds the learned model: normalization, feature transforms, recurrent
state, action chunking, unit conversion, clipping, and task-specific output
shaping. Without LEAPP, teams often rewrite that logic in another language,
maintain a separate export pipeline, or carry the full research environment
into deployment.

With LEAPP, you mark the input and output boundaries of the compute you already
run. Execute the annotated pipeline with sample inputs, and LEAPP compiles the
traced graph and validates exported outputs against captured Python outputs.
Your code does not need to implement an RL environment interface or fit into a
single ``forward()`` method. Isaac Lab also provides a native LEAPP export
integration; see :doc:`ecosystem`.

.. button-ref:: getting_started
   :ref-type: doc
   :color: primary

   Get started

.. button-ref:: guides/nodes
   :ref-type: doc
   :color: secondary

   Read usage guide

.. dropdown:: Using the agent skill

   The download includes ``SKILL.md`` and supporting guides for function patches,
   state, deployment semantics, and Warp. Extract the ZIP and give your agent
   ``leapp-integrate/SKILL.md``, or place the ``leapp-integrate`` folder in your
   agent's skills directory. Keep the ``references`` folder beside ``SKILL.md``
   so the linked guides remain available. No local LEAPP checkout is needed.

Is LEAPP a Fit?
===============

Use LEAPP when your deployment target needs more than a single model
checkpoint. It is a strong fit when you need to:

.. grid:: 1 1 2 3
   :gutter: 3
   :class-container: leapp-fit-grid

   .. grid-item-card:: Deploy Python policy code
      :class-card: leapp-fit-card sd-rounded-3

      Move Python-based policy code from research workflows into deployment.

   .. grid-item-card:: Capture the full pipeline
      :class-card: leapp-fit-card sd-rounded-3

      Keep preprocessing, postprocessing, constants, and tensor operations
      around the learned model.

   .. grid-item-card:: Reuse existing computation
      :class-card: leapp-fit-card sd-rounded-3

      Reuse supported computation directly from your application.

   .. grid-item-card:: Move between runtimes
      :class-card: leapp-fit-card sd-rounded-3

      Carry policy logic into another runtime, simulator, application, or
      deployment stack.

   .. grid-item-card:: Choose one model or several
      :class-card: leapp-fit-card sd-rounded-3

      Export the full computation as one model, or choose separate stages
      with independent export backends when needed.

   .. grid-item-card:: Describe tensor meaning
      :class-card: leapp-fit-card sd-rounded-3

      Help downstream systems connect inputs, state, commands, and outputs.

How LEAPP Fits In
=================

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      LEAPP traces the code path you actually execute. One annotated input/output
      boundary can capture the entire policy pipeline into one model file.
      Multiple boundaries let you export separate models instead. A YAML
      specification describes the model interfaces, semantic metadata, and
      any state feedback or connections between models.

      The result is an export bundle that can be inspected, validated, and
      handed to downstream systems without requiring those systems to
      understand the original training project.

   .. grid-item::

      .. raw:: html

         <div class="leapp-diagram leapp-export-diagram" aria-label="LEAPP export bundle diagram">
           <div class="leapp-diagram-stack">
             <div class="leapp-diagram-box leapp-box-annotation">annotate inputs</div>
             <div class="leapp-diagram-box leapp-box-code">Existing policy code</div>
             <div class="leapp-diagram-box leapp-box-annotation">annotate outputs</div>
             <div class="leapp-diagram-chevron">export bundle</div>
             <div class="leapp-diagram-output">
               <div class="leapp-artifact-grid">
                 <span>control.onnx</span>
                 <span>pipeline.yaml</span>
                 <span>visualization.png</span>
               </div>
             </div>
           </div>
         </div>

Supported computation
=====================

LEAPP traces :doc:`PyTorch <tensor_libraries/torch>` and supported
:doc:`NumPy <tensor_libraries/numpy>` operations, including conversions between
them. Experimental :doc:`Warp <tensor_libraries/warp>` support captures kernel
regions with an additional runtime adapter. :doc:`guides/graph` covers explicit
feedback state and automatic tracking of reassigned registered module buffers.

For unsupported module-level functions, :doc:`guides/patching` lets you register
a traceable replacement while keeping existing callers. The replacement must
preserve the behavior your application uses and rely on supported operations.

Capture follows the executed path: data-dependent Python branches and loops
are not exported as dynamic control flow. See the tensor-library limitations
and :doc:`guides/debugging` for validation guidance.


Documentation Map
=================

* Install LEAPP and walk through your first annotated policy pipeline in
  :doc:`getting_started`.
* See how LEAPP connects to Isaac Lab and Isaac ROS Deploy in
  :doc:`ecosystem`.
* Add graph-structure annotations with :doc:`guides/nodes`,
  :doc:`guides/export`, :doc:`guides/graph`, :doc:`guides/buffers`,
  :doc:`guides/patching`, and :doc:`guides/debugging`.
* Add semantic data annotations with :doc:`semantics/usage`.
* See how tracing works in :doc:`tensor_libraries/torch`, and how it extends to
  other tensor libraries in :doc:`tensor_libraries/numpy` and
  :doc:`tensor_libraries/warp`.
* Learn how to run an exported bundle in :doc:`generated_configs` and
  :doc:`leapp_runtime`.
* Browse the full :doc:`api/index`.


.. toctree::
   :caption: Getting started
   :hidden:
   :maxdepth: 2

   getting_started
   ecosystem

.. toctree::
   :caption: Graph Structure Annotations
   :hidden:
   :maxdepth: 2

   guides/nodes
   guides/export
   guides/graph
   guides/buffers
   guides/patching
   guides/debugging

.. toctree::
   :caption: Semantic Data Annotations
   :hidden:
   :maxdepth: 2

   semantics/usage
   semantics/kind_element_names
   semantics/temporal

.. toctree::
   :caption: Tensor Libraries
   :hidden:
   :maxdepth: 2

   tensor_libraries/torch
   tensor_libraries/numpy
   tensor_libraries/warp

.. toctree::
   :caption: Running the Exported Model
   :hidden:
   :maxdepth: 2

   leapp_runtime
   generated_configs

.. toctree::
   :caption: API reference
   :hidden:
   :maxdepth: 2

   api/index
