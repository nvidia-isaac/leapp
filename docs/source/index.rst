=====
LEAPP
=====

.. raw:: html

   <p class="leapp-github-link">
     <a href="https://github.com/nvidia-isaac/leapp" aria-label="LEAPP on GitHub">
       <img src="_static/images/icons/github.png" alt="" aria-hidden="true" />
       nvidia-isaac/leapp
     </a>
   </p>

**Lightweight Export Annotations for Policy Pipelines**

Export the whole policy pipeline, not just the model weights.

LEAPP helps teams move learned policies and multi-stage PyTorch pipelines
from research code into deployable artifacts. It traces real execution across
preprocessing, model inference, postprocessing, constants, and tensor library
glue, then exports per-node models and a YAML pipeline specification that
downstream runtimes can consume.

LEAPP is designed for the full policy path, including the logic that
surrounds the learned model: normalization, feature transforms, recurrent
state, action chunking, unit conversion, clipping, and task-specific output
shaping. Without LEAPP, teams often rewrite that logic in another language,
maintain a separate export pipeline, or carry the full research environment
into deployment.

With LEAPP, you mark the input and output boundaries of the compute you already
run. Execute the annotated pipeline with sample inputs, and LEAPP compiles the
traced graph into portable artifacts.

.. button-ref:: getting_started
   :ref-type: doc
   :color: primary

   Get started

.. button-ref:: guides/nodes
   :ref-type: doc
   :color: secondary

   Read usage guide

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

   .. grid-item-card:: Avoid specialized export-only implementations
      :class-card: leapp-fit-card sd-rounded-3

      Eliminate the need for a separate export-only implementation of the same
      computation.

   .. grid-item-card:: Move between runtimes
      :class-card: leapp-fit-card sd-rounded-3

      Carry policy logic into another runtime, simulator, application, or
      deployment stack.

   .. grid-item-card:: Export modular policy graphs
      :class-card: leapp-fit-card sd-rounded-3

      Represent a policy as connected nodes, so each stage can be inspected,
      exported, and wired independently.

   .. grid-item-card:: Describe tensor meaning
      :class-card: leapp-fit-card sd-rounded-3

      Help downstream systems connect inputs, state, commands, and outputs.

How LEAPP Fits In
=================

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      LEAPP traces the code path you actually execute. It records tensor flow
      between annotated stages, exports each stage independently, and writes a
      YAML specification describing how the exported pieces connect at
      inference time.

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
                 <span>node1.pt</span>
                 <span>node2.onnx</span>
                 <span>pipeline.yaml</span>
                 <span>visualization.png</span>
               </div>
             </div>
           </div>
         </div>


Documentation Map
=================

* Install LEAPP and walk through your first annotated policy pipeline in
  :doc:`getting_started`.
* See how LEAPP connects to Isaac Lab and Isaac ROS Deploy in
  :doc:`ecosystem`.
* Add graph-structure annotations with :doc:`guides/nodes`,
  :doc:`guides/export`, :doc:`guides/graph`, :doc:`guides/buffers`, and
  :doc:`guides/debugging`.
* Add semantic data annotations with :doc:`semantics/usage`.
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
   guides/debugging
   guides/runtime

.. toctree::
   :caption: Semantic Data Annotations
   :hidden:
   :maxdepth: 2

   semantics/usage
   semantics/kind_element_names
   semantics/temporal

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
