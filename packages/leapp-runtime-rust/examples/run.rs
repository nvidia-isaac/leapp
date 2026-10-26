// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execute the generated policy fixture using public LEAPP names.
use leapp_runtime::dlpack::BorrowedTensor;
use leapp_runtime::{Bundle, EngineKind, Session, SessionOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: run <policy.yaml>")?;
    let bundle = Bundle::open(path)?;
    let options = SessionOptions {
        engine: EngineKind::Triton,
        ..Default::default()
    };
    let mut session = Session::open(&bundle, options)?;
    let observation = [1.0_f32, 2.0, 3.0, 4.0];
    let bias = [10.0_f32, 20.0, 30.0, 40.0];
    let result = session.run(&[
        (
            "preprocess/observation",
            BorrowedTensor::cpu(&observation, &[4])?,
        ),
        ("policy/bias", BorrowedTensor::cpu(&bias, &[4])?),
    ])?;
    let action = result.get("policy/action").ok_or("missing policy/action")?;
    println!("{:?}", action.to_cpu_vec::<f32>()?);
    Ok(())
}
