Getting Started
===============

The original :doc:`Getting Started <../getting_started>` guide is a prerequisite
for this section. Please complete it first, then install the runtime and ONNX
Runtime using :doc:`installation`.

That guide generates ``sample_pipeline/`` in the directory where you run the
export. The bundle description is ``sample_pipeline/sample_pipeline.yaml`` and
its model is ``sample_pipeline/control.onnx``. The model normalizes joint
positions and velocities, projects gravity, runs the policy, and scales and
clips six joint targets.

The YAML declares four float32 inputs: ``control/joint_pos`` (6),
``control/joint_vel`` (6), ``control/orientation`` (4), and ``control/cmd_vel`` (3).
The public output is ``control/joint_targets`` (6).

Load and run
------------

Run the following applications from the directory containing ``sample_pipeline/``.
Keep ``sample_pipeline.yaml`` and its referenced artifacts together. Use the
qualified ``node/port`` names above when passing inputs and reading outputs.

The export uses random sample observations; these deployment examples instead
use zero joint state, the identity quaternion, and the same velocity command.
Their outputs have shape ``[6]`` and lie between -1 and 1; they need not equal
the export's sample values.

The comments group the steps covered in :doc:`bundle`,
:doc:`session_functions`, and :doc:`memory`.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import numpy as np
         import lrt

         # Bundle: parse the exported description without loading models.
         bundle = lrt.Bundle.open("sample_pipeline/sample_pipeline.yaml")

         # Input memory: NumPy owns these contiguous float32 arrays.
         inputs = {
             "control/joint_pos": np.zeros(6, dtype=np.float32),
             "control/joint_vel": np.zeros(6, dtype=np.float32),
             "control/orientation": np.array([1, 0, 0, 0], dtype=np.float32),
             "control/cmd_vel": np.array([0.5, 0, 0.1], dtype=np.float32),
         }

         # Session: load the models with default CPU execution options.
         session = lrt.Session.open(bundle, lrt.SessionOptions())

         # Execution: borrow the input arrays until this call returns.
         result = session.run(inputs)

         # Output memory: share the returned allocation through DLPack.
         targets = np.from_dlpack(result["control/joint_targets"])
         print(targets)

         # Optional cleanup: close explicitly to release session resources now.
         session.close()
         # targets retains its allocation after session closure.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include <leapp/runtime.h>
         #include <array>
         #include <iostream>

         int main() {
           // Bundle: parse the exported description without loading models.
           lrt::Bundle bundle;
           auto status = lrt::Bundle::Open("sample_pipeline/sample_pipeline.yaml", &bundle);
           if (!status.ok()) return 1;

           // Session: load the models with default CPU execution options.
           lrt::Session session;
           status = lrt::Session::Open(bundle, lrt::SessionOptions{}, &session);
           if (!status.ok()) return 1;

           // Input memory: these arrays own the float32 data.
           std::array<float, 6> pos{}, vel{};
           std::array<float, 4> quat{1, 0, 0, 0};
           std::array<float, 3> cmd{0.5f, 0, 0.1f};

           // DLTensor descriptors borrow the data and shape storage.
           // Keep all of them alive until Run returns.
           int64_t six = 6, four = 4, three = 3;
           DLTensor p{pos.data(), {kDLCPU, 0}, 1, {kDLFloat, 32, 1}, &six, nullptr, 0};
           DLTensor v{vel.data(), {kDLCPU, 0}, 1, {kDLFloat, 32, 1}, &six, nullptr, 0};
           DLTensor q{quat.data(), {kDLCPU, 0}, 1, {kDLFloat, 32, 1}, &four, nullptr, 0};
           DLTensor c{cmd.data(), {kDLCPU, 0}, 1, {kDLFloat, 32, 1}, &three, nullptr, 0};

           // Execution: pass named inputs and let the runtime allocate outputs.
           lrt::RunResult result;
           status = session.Run({{"control/joint_pos", &p}, {"control/joint_vel", &v},
                                 {"control/orientation", &q}, {"control/cmd_vel", &c}},
                                &result);
           if (!status.ok()) return 1;

           // Output memory: retain a managed lease on the returned tensor.
           lrt::ManagedTensor targets;
           status = result.GetTensor("control/joint_targets", &targets);
           if (!status.ok()) return 1;

           // Read the CPU float32 data, honoring the descriptor's byte offset.
           const auto& t = targets->dl_tensor;
           const auto* data = reinterpret_cast<const float*>(
               static_cast<const char*>(t.data) + t.byte_offset);
           for (int i = 0; i < 6; ++i) std::cout << data[i] << '\n';

           // Cleanup: the managed lease and session release resources on scope exit.
           return 0;
         }

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         use lrt::{Bundle, Session, SessionOptions};
         use lrt::dlpack::BorrowedTensor;
         use std::collections::HashMap;

         fn main() -> Result<(), Box<dyn std::error::Error>> {
             // Bundle: parse the exported description without loading models.
             let bundle = Bundle::open("sample_pipeline/sample_pipeline.yaml")?;

             // Session: load the models with default CPU execution options.
             let mut session = Session::open(&bundle, SessionOptions::default())?;

             // Input memory: these arrays own the float32 data.
             let pos = [0.0_f32; 6];
             let vel = [0.0_f32; 6];
             let quat = [1.0_f32, 0.0, 0.0, 0.0];
             let cmd = [0.5_f32, 0.0, 0.1];

             // Borrow the arrays as tensor views; keep them alive through run.
             let inputs = HashMap::from([
                 ("control/joint_pos", BorrowedTensor::cpu(&pos, &[6])?),
                 ("control/joint_vel", BorrowedTensor::cpu(&vel, &[6])?),
                 ("control/orientation", BorrowedTensor::cpu(&quat, &[4])?),
                 ("control/cmd_vel", BorrowedTensor::cpu(&cmd, &[3])?),
             ]);
             // Execution: pass named inputs and let the runtime allocate outputs.
             let result = session.run(&inputs)?;

             // Output memory: copy the managed tensor into an owned CPU vector.
             let targets = result.get("control/joint_targets").ok_or("missing targets")?;
             println!("{:?}", targets.to_cpu_vec::<f32>()?);

             // Cleanup: the result and session release resources when dropped.
             Ok(())
         }


Build the applications
----------------------

Save the chosen example as ``deploy.py``, ``deploy.cc``, or ``src/main.rs``.
See :doc:`installation` for installing each interface and its dependencies.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

         python deploy.py

   .. tab-item:: C++
      :sync: cpp

      Save this build definition as ``CMakeLists.txt``:

      .. code-block:: cmake

         cmake_minimum_required(VERSION 3.22)
         project(deploy LANGUAGES CXX)
         find_package(leapp-rt CONFIG REQUIRED)
         add_executable(deploy deploy.cc)
         target_compile_features(deploy PRIVATE cxx_std_20)
         target_link_libraries(deploy PRIVATE lrt::runtime)

      Configure, build, and run the application:

      .. code-block:: bash

         cmake -S . -B build -DCMAKE_PREFIX_PATH=<runtime-sdk-prefix>
         cmake --build build
         ./build/deploy

   .. tab-item:: Rust
      :sync: rust

      Add the dependency described in :doc:`installation` to ``Cargo.toml`` first.

      .. code-block:: bash

         cargo run --release
