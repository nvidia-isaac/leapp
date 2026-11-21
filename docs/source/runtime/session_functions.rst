Session
========

A session loads the bundle's models and owns its feedback and default-input
state. Open it once and run it repeatedly. Calls are synchronous: inference,
required transfers, and output writes finish before return.

Lifecycle
---------

.. list-table::
   :header-rows: 1

   * - Operation
     - Python
     - C++
     - Rust
   * - Load models and capture initial state
     - ``Session.open(bundle, options)``
     - ``Session::Open(bundle, options, &session)``
     - ``Session::open(&bundle, options)``
   * - Run with allocated results
     - ``session.run(inputs)``
     - ``session.Run(inputs, &result)``
     - ``session.run(&inputs)``
   * - Run into caller buffers
     - ``session.run(inputs, outputs=buffers)``
     - ``session.Run(inputs, &result, buffers)``
     - ``session.run_into(&inputs, &mut buffers)``
   * - Restore captured initial state
     - ``session.reset()``
     - ``session.Reset()``
     - ``session.reset()``
   * - Create a session with fresh state
     - ``session.new_session()``
     - ``session.NewSession(&peer)``
     - ``session.new_session()``
   * - Release this session
     - ``session.close()``
     - ``session.Close()``
     - ``session.close()``
   * - Inspect the retained bundle
     - ``session.bundle()``
     - ``session.bundle()``
     - ``session.bundle()``

The retained bundle is available through ``session.bundle()`` even after closing
the session.

* **Python:** Use ``with Session.open(...) as session`` for automatic closure.
  Operations on one session serialize; loading, execution, reset, and close
  release the GIL.
* **C++:** ``Open`` and ``NewSession`` require empty destinations; assign a default
  instance to reuse a handle. Sessions are movable and noncopyable, and
  operations on one session serialize.
* **Rust:** Execution borrows the session mutably.

Input contract
--------------

Supply every declared public input using its qualified ``node/port`` name.
Dictionary order does not affect routing. The runtime handles internal
connections and feedback; supply only public input ports.

* **Shape and layout:** Match each port's dtype and shape, resolving dynamic
  dimensions consistently with the model. Inputs must be contiguous and use
  supported native-endian numeric dtypes. Byte offsets are honored.
* **Lifetime:** Keep input storage, descriptors, shape/stride arrays, and producer
  owners alive for the entire call. Do not modify or resize storage during it.
  With :ref:`runtime-cuda-graph-replay`, retain the bound allocations until close
  and update values in place between calls.
* **Device and conversions:** Direct binding requires inputs on the execution
  device. Output placement does not move inputs. Perform contiguous copies,
  dtype conversions, and transfers explicitly in the producing library.
* **Python protocol:** Inputs implement ``__dlpack__`` and ``__dlpack_device__``,
  including NumPy, PyTorch, CuPy, and returned outputs. Read-only inputs are
  accepted if the producer can export them; export errors propagate. The
  protocol supports legacy and DLPack 1.0 versioned capsules.
* **Python CUDA synchronization:** Producers hand work to a runtime-owned stream
  per device. After batch import, each participating stream is synchronized
  once. Leases stay alive through all waits, including failure cleanup.
* **C++ and Rust raw DLPack:** Complete producer work before the call, including
  work on nondefault CUDA streams. The caller guarantees sufficient storage;
  the descriptor cannot establish allocation bounds.

Allocated outputs
-----------------

Ordinary runs allocate fresh storage for each public output, in declaration
order. Repeated declarations share one named entry.

* **Lifetime:** Results own their allocations and metadata independently of the
  session. They survive later calls, reset, and closure.
* **Shared storage:** Exporting a result multiple times can share its allocation.
  Keep each consumer owner alive until all uses finish, including CUDA work.
* **Placement:** Outputs keep their backend storage location by default. Request
  transfers with the construction-time options in :doc:`session_options`.
* **Feedback isolation:** Outputs used as feedback are copied to their selected
  public location before publication; the default preserves backend placement.
  Mutating a public output cannot change private feedback or the reset snapshot.

These snippets use the bundle and input tensors from :doc:`getting_started`;
see :doc:`memory` for allocating inputs and reading returned tensors.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Session: load the models with default CPU execution options.
         session = lrt.Session.open(bundle, lrt.SessionOptions())

         # Execution: borrow the input arrays until this call returns.
         result = session.run(inputs)

         # Optional cleanup: close explicitly to release session resources now.
         session.close()
         # result retains its output allocations after session closure.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         // Session: load the models with default CPU execution options.
         lrt::Session session;
         auto status = lrt::Session::Open(bundle, lrt::SessionOptions{}, &session);
         if (!status.ok()) return 1;

         // Execution: pass named inputs and let the runtime allocate outputs.
         lrt::RunResult result;
         status = session.Run({{"control/joint_pos", &p}, {"control/joint_vel", &v},
                               {"control/orientation", &q}, {"control/cmd_vel", &c}},
                              &result);
         if (!status.ok()) return 1;

         // Cleanup: the session releases its resources when it leaves scope.
         // result owns its output allocations independently of the session.

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         // Session: load the models with default CPU execution options.
         let mut session = lrt::Session::open(&bundle, lrt::SessionOptions::default())?;

         // Execution: borrow the input tensors until this call returns.
         let result = session.run(&inputs)?;

         // Cleanup: the session releases its resources when dropped.
         // result owns its output allocations independently of the session.


Caller-owned output buffers
---------------------------

Supply writable output buffers to reuse allocations and allow direct backend
binding. The runtime borrows them until the call returns.

* **Buffer requirements:** Match each output's dtype and resolved shape, meet
  backend device/alignment requirements, and keep buffers disjoint from inputs
  and each other.
* **Result contents:** Supplied outputs are omitted from the returned result.
  Only outputs without supplied buffers allocate storage and appear there.
* **Placement:** Supplied buffers determine their own storage location.
  Construction-time output placement applies to allocated outputs.
* **Capacity:** A backend may reject a data-dependent output buffer if its
  required capacity exceeds the final shape.
* **Failure:** Supplied buffer contents are unspecified on error.
* **CUDA graph replay:** Supply every public output of each enabled node. Keep
  its CUDA buffer alive at the same address until close; ordinary calls only
  borrow buffers until return. See :ref:`runtime-cuda-graph-replay`.
* **Copies:** Reuse avoids result allocation but does not guarantee that backend
  operations or state updates avoid copies.

These snippets use the open session and input tensors from
:doc:`getting_started`. The C++ printing code also needs ``<iostream>``.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Output memory: allocate a writable buffer owned by NumPy.
         targets = np.empty(6, dtype=np.float32)

         # Execution: write the named output into the supplied buffer.
         remaining = session.run(inputs, outputs={"control/joint_targets": targets})

         # Read the supplied buffer; remaining has no entry for this output.
         print(targets)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         // Output memory: allocate a writable buffer and describe its shape.
         std::array<float, 6> target_values{};
         int64_t target_shape[] = {6};
         DLTensor output{target_values.data(), {kDLCPU, 0}, 1,
                         {kDLFloat, 32, 1}, target_shape, nullptr, 0};

         // Execution: write the named output into the supplied buffer.
         lrt::RunResult remaining;
         auto run_status = session.Run(
             {{"control/joint_pos", &p}, {"control/joint_vel", &v},
              {"control/orientation", &q}, {"control/cmd_vel", &c}},
             &remaining, {{"control/joint_targets", &output}});
         if (!run_status.ok()) return 1;

         // Read the supplied buffer; remaining has no entry for this output.
         for (std::size_t i = 0; i < target_values.size(); ++i) {
             std::cout << "joint_targets[" << i << "] = " << target_values[i] << '\n';
         }

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         use lrt::dlpack::OutputBuffer;
         use std::collections::HashMap;

         // Output memory: allocate a writable array owned by the caller.
         let mut targets = [0.0_f32; 6];
         {
             let mut outputs = HashMap::from([
                 ("control/joint_targets", OutputBuffer::cpu(&mut targets, &[6])?),
             ]);

             // Execution: borrow the output array exclusively until run_into returns.
             let remaining = session.run_into(&inputs, &mut outputs)?;
             assert!(remaining.get("control/joint_targets").is_none());
         }

         // The mutable borrow has ended; read the supplied array directly.
         println!("{targets:?}");


Reset and new sessions
----------------------

Each session owns its feedback state and captures its initial values when opened.

* **Reset:** Restore captured feedback and default inputs without rereading files,
  reloading models, or changing retained outputs. Fixed-shape feedback keeps
  its existing buffers. Other sessions are unaffected. With CUDA graph replay,
  reset retains the capture and requires the same input/output buffers.
* **New session:** Share loaded models and construction options, but start with
  independent feedback initialized to the captured values. The source session's
  current feedback is not copied. ONNX CUDA graph replay sessions reject this
  operation; open an independent session instead.

Mutable state inside a backend model may remain shared between sessions;
neither operation resets it.

Errors
------

Error reporting follows the language interface:

* **Python:** Bundle and execution errors expose ``code``, ``phase``, and
  ``message``; ``BundleError`` also includes ``path``. Argument and protocol
  errors use standard Python exceptions.
* **C++:** Operational failures return ``Status`` with code, phase, and message.
  Failed calls preserve result destinations. Allocation failures and Rust
  panics are not recoverable status errors.
* **Rust:** ``BundleError`` and ``RuntimeError`` expose code, phase, and a source
  chain.

Feedback rollback does not cover caller-owned output-buffer contents or mutable
state inside a backend model.
