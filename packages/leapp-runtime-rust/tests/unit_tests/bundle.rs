// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use leapp_runtime::{Bundle, DataType, ErrorCode, MetadataValue as Value, PortRef};
use safetensors::{tensor::TensorView, Dtype};
use std::{fs, path::PathBuf};
use tempfile::TempDir;

const YAML: &str = include_str!("../fixtures/pipeline.yaml");

struct Fixture {
    directory: TempDir,
    path: PathBuf,
    value: Value,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("model.bin"), b"abc").unwrap();
        let path = directory.path().join("pipeline.yaml");
        Self {
            directory,
            path,
            value: serde_yaml_ng::from_str(YAML).unwrap(),
        }
    }

    fn write(&self) {
        fs::write(&self.path, serde_yaml_ng::to_string(&self.value).unwrap()).unwrap();
    }

    fn open(&self) -> Bundle {
        self.write();
        Bundle::open(&self.path).unwrap()
    }

    fn fail(&self, phase: &str, message: &str) {
        self.write();
        let error = Bundle::open(&self.path).unwrap_err();
        assert_eq!(error.phase(), phase, "{error}");
        assert!(
            error.message().contains(message),
            "{error}: missing {message:?}"
        );
        assert_eq!(error.path(), self.path);
    }

    fn initial(&mut self, name: &str, dtype: Dtype, shape: Vec<usize>, bytes: &[u8]) {
        let tensor = TensorView::new(dtype, shape, bytes).unwrap();
        safetensors::serialize_to_file(
            [(name, tensor)],
            None,
            &self.directory.path().join("initial.safetensors"),
        )
        .unwrap();
        self.value["pipeline"]["initial_values"] = Value::String("initial.safetensors".into());
    }
}

fn value(text: &str) -> Value {
    serde_yaml_ng::from_str(text).unwrap()
}

#[test]
fn inspection_preserves_order_and_metadata() {
    let f = Fixture::new();
    let b = f.open();
    assert_eq!(b.model_names(), ["z_observation", "a_policy"]);
    assert_eq!(b.yaml_path(), f.path);
    assert_eq!(b.pipeline_inputs()[0].name(), "z_observation/joint_pos");
    assert_eq!(
        b.pipeline_outputs()
            .iter()
            .map(PortRef::name)
            .collect::<Vec<_>>(),
        [
            "a_policy/next_state",
            "a_policy/actions",
            "z_observation/features"
        ]
    );
    let port = b.input_info("a_policy", "features").unwrap();
    assert_eq!(port.shape, [3]);
    assert_eq!(port.dtype, DataType::Float32);
    assert_eq!(port.tensor_type, "tensor");
    assert_eq!(
        b.output_info("a_policy", "actions").unwrap().metadata["temporal_period_ms"],
        value("100")
    );
    assert_eq!(b.pipeline_configs()["frequency"], value("50"));
    assert_eq!(b.system_information()["leapp config version"], "1.3");
    let parameters = b.model_parameters("z_observation").unwrap();
    assert_eq!(parameters["backend"], "jit");
    assert_eq!(parameters["model_path"], "model.bin");
    assert_eq!(
        parameters["custom_options"],
        value("{device: cpu, flags: [one, two]}")
    );
}

#[test]
fn model_io_preserves_internal_port_order_and_distinguishes_absent_models() {
    let mut f = Fixture::new();
    let model = &mut f.value["models"]["a_policy"];
    for direction in ["inputs", "outputs"] {
        let ports = model[direction].as_sequence_mut().unwrap();
        ports.push(ports[0].clone());
    }
    let mut empty = model.clone();
    empty["inputs"] = value("[]");
    empty["outputs"] = value("[]");
    f.value["models"]["empty"] = empty;

    let b = f.open();
    let (inputs, outputs) = b.model_io("a_policy").unwrap();
    assert_eq!(inputs, ["features", "state", "features"]);
    assert_eq!(outputs, ["actions", "next_state", "actions"]);
    for name in inputs {
        assert!(b.input_info("a_policy", name).is_some());
    }
    for name in outputs {
        assert!(b.output_info("a_policy", name).is_some());
    }
    assert_eq!(b.model_io("empty"), Some((vec![], vec![])));
    assert!(b.model_io("absent").is_none());
}

#[test]
fn getters_preserve_unknown_fields_at_each_level() {
    let mut f = Fixture::new();
    f.value["project"] =
        value("{name: example, axes: [x, y, z], mixed: [null, true, -2, 2.5, '3']}");
    f.value["pipeline"]["description"] = value("feedback");
    f.value["models"]["a_policy"]["purpose"] = value("control");
    f.value["models"]["a_policy"]["parameters"][""] = value("custom");
    let b = f.open();
    assert_eq!(b.metadata()["project"], f.value["project"]);
    assert_eq!(b.pipeline_metadata()["description"], "feedback");
    assert_eq!(b.model_metadata("a_policy").unwrap()["purpose"], "control");
    assert_eq!(b.model_parameters("a_policy").unwrap()[""], "custom");
}

#[test]
fn absent_lookups_return_none() {
    let b = Fixture::new().open();
    assert!(b.model_parameters("absent").is_none());
    assert!(b.model_metadata("absent").is_none());
    assert!(b.input_info("absent", "features").is_none());
    assert!(b.input_info("a_policy", "absent").is_none());
    assert!(b.output_info("a_policy", "state").is_none());
}

#[test]
fn clones_share_a_snapshot_and_survive_file_removal() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Bundle>();
    let f = Fixture::new();
    let original = f.open();
    let copy = original.clone();
    assert!(std::ptr::eq(
        original.input_info("a_policy", "state").unwrap(),
        copy.input_info("a_policy", "state").unwrap()
    ));
    drop(original);
    fs::remove_file(&f.path).unwrap();
    fs::remove_file(f.directory.path().join("model.bin")).unwrap();
    std::thread::spawn(move || {
        assert_eq!(copy.input_info("a_policy", "state").unwrap().shape, [3])
    })
    .join()
    .unwrap();
}

#[test]
fn failed_open_does_not_replace_an_existing_bundle() {
    let f = Fixture::new();
    let original = f.open();
    let error = Bundle::open(f.directory.path().join("missing.yaml")).unwrap_err();
    assert_eq!(error.code(), ErrorCode::NotFound);
    assert_eq!(error.phase(), "open");
    assert_eq!(original.model_names(), ["z_observation", "a_policy"]);
    use std::error::Error;
    let mut cause = error.source().unwrap();
    while let Some(next) = cause.source() {
        cause = next;
    }
    assert!(cause.downcast_ref::<std::io::Error>().is_some());
}

#[test]
fn malformed_documents_are_rejected() {
    let f = Fixture::new();
    for text in [
        "[",
        "null",
        "[]",
        "{}",
        "models: {}\nmodels: {}",
        "models: {}\n---\nmodels: {}",
        "models: !custom {}",
    ] {
        fs::write(&f.path, text).unwrap();
        let error = Bundle::open(&f.path).unwrap_err();
        assert_eq!(error.phase(), "parse", "{error}");
    }
    assert_eq!(
        Bundle::open("").unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

#[test]
fn collections_cannot_be_null_or_scalar() {
    for section in ["inputs", "outputs", "data_flow", "feedback_flow", "configs"] {
        for text in ["null", "invalid"] {
            let mut f = Fixture::new();
            f.value["pipeline"][section] = value(text);
            f.fail("parse", section);
        }
    }
    for field in ["inputs", "outputs", "parameters"] {
        let mut f = Fixture::new();
        f.value["models"]["a_policy"][field] = Value::Null;
        f.fail("parse", field);
    }
}

#[test]
fn accepts_empty_graph_and_unspecified_version() {
    let mut f = Fixture::new();
    f.value = value("models: {}\npipeline: {inputs: {}, outputs: {}}\nsystem information: {}\n");
    let b = f.open();
    assert!(b.model_names().is_empty());
    assert!(b.pipeline_inputs().is_empty());
    assert!(b.pipeline_outputs().is_empty());
    assert!(b.pipeline_configs().is_empty());
    f.value["system information"]["leapp config version"] = value("future");
    assert_eq!(
        f.open().system_information()["leapp config version"],
        "future"
    );
}

#[test]
fn preserves_supported_backend_aliases_in_parameters() {
    for backend in [
        "onnx",
        "onnx-dynamo",
        "onnx-torchscript",
        "jit",
        "jit-script",
        "jit-trace",
        "pt2",
        "exported-program",
        "trt",
        "tensorrt",
    ] {
        let mut f = Fixture::new();
        f.value["models"]["a_policy"]["parameters"]["backend"] = value(backend);
        assert_eq!(
            f.open().model_parameters("a_policy").unwrap()["backend"],
            backend
        );
    }
    let mut f = Fixture::new();
    f.value["models"]["a_policy"]["parameters"]["backend"] = value("custom");
    assert_eq!(
        f.open().model_parameters("a_policy").unwrap()["backend"],
        "custom"
    );
    f.value["models"]["a_policy"]["parameters"]["backend"] = value(" ");
    f.write();
    assert_eq!(
        Bundle::open(&f.path).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

#[test]
fn preserves_shapes_and_all_canonical_dtypes() {
    for shape in ["[]", "[0]", "[-1]", "[2, 3]", "'[2, 3]'"] {
        for (name, dtype) in [
            ("bool", DataType::Bool),
            ("uint8", DataType::Uint8),
            ("uint16", DataType::Uint16),
            ("uint32", DataType::Uint32),
            ("uint64", DataType::Uint64),
            ("int8", DataType::Int8),
            ("int16", DataType::Int16),
            ("int32", DataType::Int32),
            ("int64", DataType::Int64),
            ("float16", DataType::Float16),
            ("bfloat16", DataType::Bfloat16),
            ("float32", DataType::Float32),
            ("float64", DataType::Float64),
        ] {
            let mut f = Fixture::new();
            for node in f.value["models"].as_mapping_mut().unwrap().values_mut() {
                for direction in ["inputs", "outputs"] {
                    for port in node[direction].as_sequence_mut().unwrap() {
                        port["shape"] = value(shape);
                        port["dtype"] = value(name);
                    }
                }
            }
            let b = f.open();
            assert_eq!(b.input_info("a_policy", "features").unwrap().dtype, dtype);
            let expected = if shape.starts_with('\'') {
                value("[2, 3]")
            } else {
                value(shape)
            };
            assert_eq!(
                b.input_info("a_policy", "features").unwrap().shape,
                serde_yaml_ng::from_value::<Vec<i64>>(expected).unwrap()
            );
        }
    }
}

#[test]
fn malformed_shapes_and_unknown_dtypes_fail() {
    for shape in [
        "[1.5]",
        "[true]",
        "[null]",
        "[batch]",
        "[9223372036854775808]",
        "3",
        "null",
        "'(3,)'",
        "'[3,]'",
    ] {
        let mut f = Fixture::new();
        f.value["models"]["a_policy"]["inputs"][0]["shape"] = value(shape);
        f.fail("parse", "shape");
    }
    let mut f = Fixture::new();
    f.value["models"]["a_policy"]["inputs"][0]["dtype"] = value("complex64");
    f.fail("parse", "dtype");
}

#[test]
fn sha256_is_optional_but_verified_when_present() {
    let mut f = Fixture::new();
    fs::write(f.directory.path().join("model.bin"), b"changed").unwrap();
    f.fail("artifact", "SHA256 mismatch");
    for node in f.value["models"].as_mapping_mut().unwrap().values_mut() {
        let params = node["parameters"].as_mapping_mut().unwrap();
        params.remove("sha256sum");
        params.remove("md5sum");
    }
    f.open();
    fs::remove_file(f.directory.path().join("model.bin")).unwrap();
    f.write();
    assert_eq!(
        Bundle::open(&f.path).unwrap_err().code(),
        ErrorCode::NotFound
    );
}

#[test]
fn preserves_md5_metadata_and_accepts_uppercase_sha256() {
    let mut f = Fixture::new();
    for node in f.value["models"].as_mapping_mut().unwrap().values_mut() {
        node["parameters"]["md5sum"] = value("ignored");
        let hash = node["parameters"]["sha256sum"]
            .as_str()
            .unwrap()
            .to_uppercase();
        node["parameters"]["sha256sum"] = Value::String(hash);
    }
    assert_eq!(
        f.open().model_parameters("a_policy").unwrap()["md5sum"],
        "ignored"
    );
}

#[test]
fn checks_references_and_edge_compatibility() {
    for target in ["missing/features", "a_policy/missing", "a_policy/actions"] {
        let mut f = Fixture::new();
        f.value["pipeline"]["data_flow"]["z_observation/features"] =
            Value::Sequence(vec![Value::String(target.into())]);
        f.fail("validate", target);
    }
    for field in ["dtype", "shape"] {
        let mut f = Fixture::new();
        f.value["models"]["a_policy"]["inputs"][0][field] =
            value(if field == "dtype" { "float64" } else { "[2]" });
        f.fail("validate", "incompatible tensors");
    }
}

#[test]
fn accepts_reversed_model_order_and_cycles() {
    let mut f = Fixture::new();
    let models = f.value["models"].as_mapping_mut().unwrap();
    let first = models.shift_remove("z_observation").unwrap();
    models.insert(value("z_observation"), first);
    assert_eq!(f.open().model_names(), ["a_policy", "z_observation"]);
    f.value["pipeline"]["inputs"] = value("{}");
    f.value["pipeline"]["data_flow"]["a_policy/actions"] = value("[z_observation/joint_pos]");
    f.open();
}

#[test]
fn permits_multiple_producers_but_requires_output_routing() {
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"]["a_policy"] = value("[features]");
    f.open();
    let mut f = Fixture::new();
    f.value["pipeline"]["outputs"]["a_policy"] = value("[next_state]");
    f.fail("validate", "output is not routed");
    f.value["pipeline"]["data_flow"]["a_policy/actions"] = value("[]");
    f.open();
    f.value["pipeline"]["inputs"] = value("{}");
    f.open();
}

#[test]
fn accepts_duplicate_ports_and_boundaries() {
    let mut f = Fixture::new();
    let port = f.value["models"]["a_policy"]["inputs"][0].clone();
    f.value["models"]["a_policy"]["inputs"]
        .as_sequence_mut()
        .unwrap()
        .push(port);
    f.open();
    let mut f = Fixture::new();
    f.value["pipeline"]["outputs"]["a_policy"] = value("[actions, actions]");
    f.open();
}

#[test]
fn initial_values_are_optional_and_can_target_any_input() {
    for initial in ["null", "''"] {
        let mut f = Fixture::new();
        f.value["pipeline"]["initial_values"] = value(initial);
        f.open();
    }
    let mut f = Fixture::new();
    f.initial("z_observation/joint_pos", Dtype::I32, vec![1, 3], &[0; 12]);
    let b = f.open();
    assert_eq!(
        b.input_info("z_observation", "joint_pos").unwrap().dtype,
        DataType::Float32
    );
    f.initial("a_policy/state", Dtype::F32, vec![3], &[0; 12]);
    f.open();
    f.initial("a_policy/missing", Dtype::F32, vec![3], &[0; 12]);
    f.fail("initial_state", "a_policy/missing");
}

#[test]
fn malformed_or_truncated_initial_values_fail() {
    let mut f = Fixture::new();
    f.initial("a_policy/state", Dtype::F32, vec![3], &[0; 12]);
    let path = f.directory.path().join("initial.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    bytes.pop();
    fs::write(&path, bytes).unwrap();
    f.fail("initial_state", "initial.safetensors");
    fs::write(&path, b"invalid").unwrap();
    f.fail("initial_state", "initial.safetensors");
}

#[cfg(unix)]
#[test]
fn yaml_symlinks_resolve_artifacts_beside_the_supplied_path() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new();
    f.write();
    let other = tempfile::tempdir().unwrap();
    let path = other.path().join("alias.yaml");
    symlink(&f.path, &path).unwrap();
    assert_eq!(Bundle::open(&path).unwrap_err().code(), ErrorCode::NotFound);
    fs::write(other.path().join("model.bin"), b"abc").unwrap();
    assert_eq!(Bundle::open(&path).unwrap().yaml_path(), path);
}

#[cfg(unix)]
#[test]
fn yaml_and_artifacts_use_the_same_normalized_path() {
    use std::os::unix::fs::symlink;
    let mut f = Fixture::new();
    f.value["origin"] = value("normalized");
    f.write();
    let mut other = Fixture::new();
    other.value["origin"] = value("filesystem");
    other.write();
    let child = other.directory.path().join("child");
    fs::create_dir(&child).unwrap();
    symlink(&child, f.directory.path().join("link")).unwrap();
    let path = f.directory.path().join("link/../pipeline.yaml");

    // The filesystem follows the symlink before processing the parent segment.
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        fs::read_to_string(&other.path).unwrap()
    );
    let bundle = Bundle::open(&path).unwrap();
    assert_eq!(bundle.yaml_path(), f.path);
    assert_eq!(bundle.metadata()["origin"], "normalized");

    fs::remove_file(f.directory.path().join("model.bin")).unwrap();
    assert_eq!(Bundle::open(&path).unwrap_err().code(), ErrorCode::NotFound);
}

#[test]
fn uses_yaml12_scalar_resolution() {
    let f = Fixture::new();
    fs::write(
        &f.path,
        format!("{YAML}\nlabels: [x, y, z, yes, on, true, 'true']\n"),
    )
    .unwrap();
    let b = Bundle::open(&f.path).unwrap();
    assert_eq!(
        b.metadata()["labels"],
        value("[x, y, z, 'yes', 'on', true, 'true']")
    );
}

#[test]
fn metadata_has_no_additional_depth_limit() {
    let f = Fixture::new();
    fs::write(
        &f.path,
        format!("{YAML}\ndeep: {}0{}\n", "[".repeat(65), "]".repeat(65)),
    )
    .unwrap();
    Bundle::open(&f.path).unwrap();
}

#[test]
fn duplicate_mapping_keys_and_yaml_merges_follow_python() {
    let f = Fixture::new();
    fs::write(&f.path, format!("{YAML}\ncustom: {{a: 1, a: 2}}\nshared: &base {{label: inherited}}\nmerged: {{<<: *base, own: true}}\n")).unwrap();
    let b = Bundle::open(&f.path).unwrap();
    assert_eq!(b.metadata()["custom"]["a"], value("2"));
    assert_eq!(b.metadata()["merged"]["label"], "inherited");
    assert_eq!(b.metadata()["merged"]["own"], value("true"));

    fs::write(
        &f.path,
        format!("{YAML}\nmodels: {{}}\npipeline: {{inputs: {{}}, outputs: {{}}}}\n"),
    )
    .unwrap();
    assert!(Bundle::open(&f.path).unwrap().model_names().is_empty());
}

#[test]
fn external_input_references_are_descriptive() {
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"] = value("{unknown: [missing, missing], unused: []}");
    let b = f.open();
    assert_eq!(b.pipeline_inputs()[0].name(), "unknown/missing");
    assert_eq!(b.pipeline_inputs().len(), 2);
}

#[test]
fn duplicate_edge_ports_match_python_lookup_order() {
    let mut f = Fixture::new();
    let mut input = f.value["models"]["a_policy"]["inputs"][0].clone();
    input["shape"] = value("[7]");
    f.value["models"]["a_policy"]["inputs"]
        .as_sequence_mut()
        .unwrap()
        .insert(0, input);
    let mut output = f.value["models"]["z_observation"]["outputs"][0].clone();
    output["shape"] = value("[7]");
    f.value["models"]["z_observation"]["outputs"]
        .as_sequence_mut()
        .unwrap()
        .push(output);
    f.open();
}

#[test]
fn names_and_tensor_labels_have_no_extra_validation() {
    let mut f = Fixture::new();
    let port = &mut f.value["models"]["z_observation"]["inputs"][0];
    port["name"] = Value::String("/\0".into());
    port["type"] = value("''");
    f.open();
    f.value["models"]["z_observation"]["inputs"][0]
        .as_mapping_mut()
        .unwrap()
        .remove("type");
    assert_eq!(
        f.open()
            .input_info("z_observation", "/\0")
            .unwrap()
            .tensor_type,
        ""
    );
    assert!(PortRef::try_from("/".to_string()).is_ok());
    assert!(PortRef::try_from("node/\0".to_string()).is_ok());
    assert!(PortRef::try_from("node/port/extra".to_string()).is_err());
}

#[test]
fn nested_metadata_keys_need_not_be_strings() {
    let mut f = Fixture::new();
    f.value["custom"] = value("{1: number, false: boolean}");
    assert_eq!(f.open().metadata()["custom"][value("1")], "number");
}

#[test]
fn artifacts_fall_back_to_the_working_directory() {
    let working_directory = std::env::current_dir().unwrap();
    let artifact_directory = tempfile::tempdir_in(&working_directory).unwrap();
    let artifact_path = artifact_directory.path().join("model.bin");
    fs::write(&artifact_path, b"abc").unwrap();
    let relative_path = artifact_path.strip_prefix(&working_directory).unwrap();
    let mut f = Fixture::new();
    for model in f.value["models"].as_mapping_mut().unwrap().values_mut() {
        model["parameters"]["model_path"] = Value::String(relative_path.to_str().unwrap().into());
    }
    f.open();

    // An existing path beside the YAML takes precedence over the fallback.
    let adjacent = f.directory.path().join(relative_path);
    fs::create_dir_all(adjacent.parent().unwrap()).unwrap();
    fs::write(adjacent, b"changed").unwrap();
    f.fail("artifact", "SHA256 mismatch");
}

#[test]
fn nested_yaml_merges_expand_before_overrides() {
    let f = Fixture::new();
    fs::write(&f.path, format!("{YAML}\nbase: &base {{frequency: 50, label: inherited}}\nmiddle: &middle {{<<: *base, frequency: 100}}\nlast: {{<<: *middle}}\ncombined: {{<<: [*middle, *base], label: explicit}}\n")).unwrap();
    let b = Bundle::open(&f.path).unwrap();
    assert_eq!(b.metadata()["last"]["frequency"], value("100"));
    assert_eq!(b.metadata()["last"]["label"], "inherited");
    assert!(b.metadata()["last"].get("<<").is_none());
    assert_eq!(b.metadata()["combined"]["frequency"], value("100"));
    assert_eq!(b.metadata()["combined"]["label"], "explicit");
}
