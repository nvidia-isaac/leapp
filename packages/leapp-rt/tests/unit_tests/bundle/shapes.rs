// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::{json, Value};

fn execution_shape(port: &PortSpec) -> Result<ExecutionShape> {
    Compiler::default().port(port)
}

fn port(variable: Value) -> PortSpec {
    serde_json::from_value(json!({"name":"x", "dtype":"float32", "shape":[4,6],
        "axes":[{"variable":variable},{}]}))
    .unwrap()
}

#[test]
fn bounds_change_execution_shape_without_changing_capture() {
    let spec = port(json!({"symbol":"rows","min":0,"max":8,"multiple_of":2}));
    let shape = execution_shape(&spec).unwrap();
    assert_eq!(spec.shape, [4, 6]);
    assert_eq!(shape.dims, [-1, 6]);
    for actual in [[0, 6], [2, 6], [4, 6], [8, 6]] {
        assert!(shape.matches(&actual));
    }
    for actual in [[-1, 6], [3, 6], [10, 6], [4, 3]] {
        assert!(!shape.matches(&actual));
    }
    assert!(!shape.matches(&[4]));
    assert!(!shape.matches(&[4, 6, 1]));
}

#[test]
fn malformed_and_unsupported_constraints_are_rejected() {
    for variable in [
        json!(true),
        json!({"symbol":"n","min":0}),
        json!({"symbol":"n","min":0,"max":8,"multiple_of":0}),
        json!({"symbol":"n","min":-1,"max":8}),
        json!({"symbol":"n","min":0,"max":2147483648i64}),
        json!({"symbol":"n","min":0,"max":3}),
        json!({"symbol":"","min":0,"max":8}),
        json!({"symbol":"n","min":0,"max":8,"multiple_of":3}),
        json!({"symbol":"n","min":0,"max":8,"expression":{"formula":"n*2"}}),
        json!({"symbol":"n","min":0,"max":8,"roots":{}}),
    ] {
        assert!(
            execution_shape(&port(variable.clone())).is_err(),
            "{variable}"
        );
    }
    for variable in [Value::Null, json!(false)] {
        let shape = execution_shape(&port(variable)).unwrap();
        assert_eq!(shape, ExecutionShape::from(vec![4, 6]));
    }
    for expression in [
        json!({"formula":{"op":"add","args":[1]}}),
        json!({"formula":{"op":"add","args":[1,2,3]}}),
        json!({"formula":{"op":"div","args":[4,1]}}),
        json!({"formula":1.5}),
        json!({"terms":{"n":1.5},"offset":0}),
        json!({"terms":{"n":1},"offset":"0"}),
    ] {
        let spec = port(json!({"symbol":"n","min":0,"max":8,"expression":expression}));
        assert!(execution_shape(&spec).is_err());
    }
}

#[test]
fn shared_and_derived_dimensions_use_invocation_local_values() {
    let mut compiler = Compiler::default();
    let root = json!({"symbol":"n","min":2,"max":16,"multiple_of":2});
    let input = compiler.port(&port(root.clone())).unwrap();
    let derived = port(json!({"symbol":"half","min":1,"max":8,
        "roots":{"n":{"min":2,"max":16,"multiple_of":2}},
        "expression":{"formula":{"op":"floordiv","args":["n",2]}}}));
    let output = compiler.port(&derived).unwrap();
    let symbols = &compiler.symbols;
    for size in [4, 8, 4] {
        let mut values = vec![None; symbols.len()];
        input.bind(&[size, 6], symbols, &mut values).unwrap();
        output.bind(&[size / 2, 6], symbols, &mut values).unwrap();
        assert!(input.bind(&[size + 2, 6], symbols, &mut values).is_err());
        assert!(output
            .bind(&[size / 2 + 1, 6], symbols, &mut values)
            .is_err());
    }
    let affine = port(json!({"symbol":"twice","min":0,"max":32,
        "roots":{"n":{"min":2,"max":16,"multiple_of":2}},
        "expression":{"terms":{"n":2},"offset":0}}));
    let twice = compiler.port(&affine).unwrap();
    let symbols = &compiler.symbols;
    for (size, valid) in [(4, true), (8, true), (6, false), (5, false), (36, false)] {
        let mut values = vec![None; symbols.len()];
        assert_eq!(twice.bind(&[size, 6], symbols, &mut values).is_ok(), valid);
    }
    let mut conflicting = root;
    conflicting["max"] = 32.into();
    assert!(compiler.port(&port(conflicting)).is_err());
}

#[test]
fn formulas_use_checked_integer_arithmetic() {
    for (formula, expected) in [
        (json!({"op":"floordiv","args":[-3,2]}), Some(-2)),
        (json!({"op":"mod","args":[-3,2]}), Some(1)),
        (
            json!({"op":"mul","args":[{"op":"add","args":[2,3]},4]}),
            Some(20),
        ),
        (json!({"op":"add","args":[i64::MAX,1]}), None),
        (json!({"op":"mul","args":[i64::MAX,2]}), None),
        (json!({"op":"floordiv","args":[4,0]}), None),
    ] {
        let expression =
            Expression::parse(&yaml_serde::to_value(formula).unwrap(), &Symbols::new()).unwrap();
        assert_eq!(expression.evaluate(&[]).ok(), expected);
    }
}

#[test]
fn shape_contract_versions_and_default_upper_bound() {
    let spec = port(json!({"symbol":"n","min":1,"max":2147483647}));
    assert!(execution_shape(&spec).is_ok());
    let mut description: BundleDescription = serde_json::from_value(json!({
        "models":{"node":{"inputs":[],"outputs":[],"parameters":{"backend":"onnx","model_path":"model.onnx"}}},
        "pipeline":{"inputs":{},"outputs":{}}, "system information":{}
    })).unwrap();
    description.models["node"].inputs.push(spec);
    assert!(compile(&mut description).is_err());
    for (version, valid) in [(0, false), (1, true), (2, true), (3, false)] {
        description
            .system_information
            .insert("shape contract version".into(), version.into());
        assert_eq!(compile(&mut description).is_ok(), valid);
    }
    description.models["node"].inputs[0] = port(json!({"symbol":"half","min":1,"max":8,
        "roots":{"n":{"min":2,"max":16,"multiple_of":2}},
        "expression":{"terms":{"n":1},"offset":-2}}));
    description
        .system_information
        .insert("shape contract version".into(), 1.into());
    assert!(compile(&mut description).is_err());
    description
        .system_information
        .insert("shape contract version".into(), 2.into());
    compile(&mut description).unwrap();
}
