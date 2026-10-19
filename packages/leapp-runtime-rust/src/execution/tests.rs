// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{DataType, ErrorCode, MetadataValue as Value};
use proto::model_config::{model_instance_group::Kind, DataType as TritonDataType};
use safetensors::{tensor::TensorView, Dtype};
use std::fs;
use tempfile::TempDir;

struct Fixture {
    directory: TempDir,
    value: Value,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("model.bin"), b"abc").unwrap();
        let mut value: Value =
            serde_yaml_ng::from_str(include_str!("../../tests/fixtures/pipeline.yaml")).unwrap();
        for model in value["models"].as_mapping_mut().unwrap().values_mut() {
            model["parameters"]["backend"] = yaml("onnx");
        }
        Self { directory, value }
    }

    fn open(&self) -> Bundle {
        let path = self.directory.path().join("pipeline.yaml");
        fs::write(&path, serde_yaml_ng::to_string(&self.value).unwrap()).unwrap();
        Bundle::open(path).unwrap()
    }

    fn convert(&self) -> TritonPipeline {
        convert_bundle(&self.open(), &ConversionOptions::default()).unwrap()
    }

    fn unsupported(&self, message: &str) {
        let bundle = self.open();
        let error = convert_bundle(&bundle, &ConversionOptions::default()).unwrap_err();
        assert_eq!(error.phase(), "convert");
        assert_eq!(error.code(), ErrorCode::Unsupported);
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(
            bundle.model_names().len(),
            self.value["models"].as_mapping().unwrap().len()
        );
    }
}

fn yaml(text: &str) -> Value {
    serde_yaml_ng::from_str(text).unwrap()
}

#[test]
fn converts_native_configs_and_preserves_qualified_tensor_names() {
    let f = Fixture::new();
    let p = f.convert();
    assert_eq!(p.models.len(), 2);
    let model = &p.models[0];
    assert_eq!(model.config.name, "leapp_node_0");
    assert_eq!(model.config.backend, "onnxruntime");
    assert_eq!(model.config.default_model_filename, "model.bin");
    assert_eq!(
        model.artifact.model_path,
        f.directory.path().join("model.bin")
    );
    assert_eq!(model.version, 1);
    assert_eq!(
        model
            .config
            .version_policy
            .as_ref()
            .unwrap()
            .specific()
            .versions,
        [1]
    );
    assert_eq!(model.config.max_batch_size, 0);
    assert_eq!(
        model.config.instance_group[0].kind.enum_value().unwrap(),
        Kind::KIND_CPU
    );
    assert_eq!(model.config.input[0].name, "joint_pos");
    assert_eq!(model.config.input[0].dims, [3]);
    assert_eq!(
        model.config.input[0].data_type.enum_value().unwrap(),
        TritonDataType::TYPE_FP32
    );
    assert_eq!(p.ensemble.platform, "ensemble");
    assert_eq!(p.ensemble.max_batch_size, 0);
    assert_eq!(p.inputs[0].port.name(), "z_observation/joint_pos");
    assert_eq!(p.inputs[0].tensor_name, "z_observation/joint_pos");
    assert_eq!(
        p.outputs
            .iter()
            .map(|b| b.tensor_name.as_str())
            .collect::<Vec<_>>(),
        [
            "a_policy/next_state",
            "a_policy/actions",
            "z_observation/features"
        ]
    );
    let steps = &p.ensemble.ensemble_scheduling().step;
    assert_eq!(steps[0].output_map["features"], "z_observation/features");
    assert_eq!(steps[1].input_map["features"], "z_observation/features");
    assert!(steps.iter().all(|step| step.model_version == 1));
    for config in p
        .models
        .iter()
        .map(|model| &model.config)
        .chain([&p.ensemble])
    {
        let text = protobuf::text_format::print_to_string(config);
        let decoded: ModelConfig = protobuf::text_format::parse_from_str(&text).unwrap();
        assert_eq!(*config, decoded);
    }
}

#[test]
fn conversion_only_uses_the_bundle_snapshot() {
    let f = Fixture::new();
    let bundle = f.open();
    fs::remove_file(bundle.yaml_path()).unwrap();
    fs::remove_file(f.directory.path().join("model.bin")).unwrap();
    assert_eq!(
        convert_bundle(&bundle, &ConversionOptions::default())
            .unwrap()
            .models
            .len(),
        2
    );
}

#[test]
fn resolves_edges_without_reordering_models() {
    let mut f = Fixture::new();
    let models = f.value["models"].as_mapping_mut().unwrap();
    let first = models.shift_remove("z_observation").unwrap();
    models.insert(yaml("z_observation"), first);
    let p = f.convert();
    assert_eq!(p.models[0].config.input[0].name, "features");
    assert_eq!(
        p.ensemble.ensemble_scheduling().step[0].input_map["features"],
        "z_observation/features"
    );
}

#[test]
fn records_feedback_defaults_and_checked_initial_value_references() {
    let mut f = Fixture::new();
    let mut port = f.value["models"]["a_policy"]["inputs"][0].clone();
    port["name"] = yaml("bias");
    f.value["models"]["a_policy"]["inputs"]
        .as_sequence_mut()
        .unwrap()
        .push(port);
    let bytes = [0; 12];
    let tensors = ["a_policy/state", "a_policy/bias"].map(|name| {
        (
            name,
            TensorView::new(Dtype::I32, vec![1, 3], &bytes).unwrap(),
        )
    });
    let file = f.directory.path().join("initial.safetensors");
    safetensors::serialize_to_file(tensors, None, &file).unwrap();
    f.value["pipeline"]["initial_values"] = yaml("initial.safetensors");
    let p = f.convert();
    assert_eq!(p.feedback.len(), 1);
    let feedback = &p.feedback[0];
    assert_eq!(feedback.target.name(), "a_policy/state");
    assert_eq!(feedback.input_name, "a_policy/state");
    assert_eq!(feedback.output_name, "a_policy/next_state");
    assert_eq!(feedback.initial_value.as_ref().unwrap().file, file);
    assert_eq!(
        feedback.initial_value.as_ref().unwrap().tensor_name,
        "a_policy/state"
    );
    assert_eq!(p.default_inputs.len(), 1);
    assert_eq!(p.default_inputs[0].input.port.name(), "a_policy/bias");
    assert_eq!(
        p.default_inputs[0]
            .initial_value
            .as_ref()
            .unwrap()
            .tensor_name,
        "a_policy/bias"
    );
    assert_eq!(p.inputs.len(), 1);
    assert_eq!(p.outputs.len(), 3);
}

#[test]
fn absent_initial_values_remain_optional() {
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"] = yaml("{}");
    let p = f.convert();
    assert!(p.feedback[0].initial_value.is_none());
    assert!(p.default_inputs[0].initial_value.is_none());
}

#[test]
fn duplicate_boundaries_and_edges_share_their_bindings() {
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"]["z_observation"] = yaml("[joint_pos, joint_pos]");
    f.value["pipeline"]["outputs"]["a_policy"] = yaml("[actions, actions, next_state]");
    f.value["pipeline"]["data_flow"]["z_observation/features"] =
        yaml("[a_policy/features, a_policy/features]");
    f.value["pipeline"]["feedback_flow"]["a_policy/next_state"] =
        yaml("[a_policy/state, a_policy/state]");
    let p = f.convert();
    assert_eq!(p.inputs.len(), 2);
    assert_eq!(p.inputs[0].tensor_name, p.inputs[1].tensor_name);
    assert_eq!(p.ensemble.input.len(), 2);
    assert_eq!(p.ensemble.output.len(), 3);
    assert_eq!(p.feedback.len(), 1);
    assert_eq!(p.outputs[0].tensor_name, p.outputs[1].tensor_name);
}

#[test]
fn requests_terminal_outputs_without_publishing_them() {
    let mut f = Fixture::new();
    f.value["pipeline"]["outputs"]["a_policy"] = yaml("[next_state]");
    f.value["pipeline"]["data_flow"]["a_policy/actions"] = yaml("[]");
    let p = f.convert();
    assert!(!p
        .outputs
        .iter()
        .any(|b| b.port.name() == "a_policy/actions"));
    assert!(p
        .ensemble
        .output
        .iter()
        .any(|p| p.name == "a_policy/actions"));
}

#[test]
fn rejects_cycles_and_competing_producers_only_during_conversion() {
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"] = yaml("{}");
    f.value["pipeline"]["data_flow"]["a_policy/actions"] = yaml("[z_observation/joint_pos]");
    f.unsupported("cycle");
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"]["a_policy"] = yaml("[features]");
    f.unsupported("multiple sources");
    let mut f = Fixture::new();
    f.value["pipeline"]["data_flow"]["z_observation/features"] =
        yaml("[a_policy/features, a_policy/state]");
    f.unsupported("multiple sources");
    let mut f = Fixture::new();
    f.value["pipeline"]["data_flow"]["a_policy/actions"] = yaml("[a_policy/features]");
    f.unsupported("multiple sources");
}

#[test]
fn reports_unresolved_public_inputs() {
    let mut f = Fixture::new();
    f.value["pipeline"]["inputs"]["missing"] = yaml("[port]");
    let error = convert_bundle(&f.open(), &ConversionOptions::default()).unwrap_err();
    assert_eq!(error.phase(), "convert");
    assert!(error.to_string().contains("missing/port"));
}

#[test]
fn rejects_ambiguous_model_ports() {
    for direction in ["inputs", "outputs"] {
        let mut f = Fixture::new();
        let port = f.value["models"]["a_policy"][direction][0].clone();
        f.value["models"]["a_policy"][direction]
            .as_sequence_mut()
            .unwrap()
            .push(port);
        f.unsupported("duplicate");
    }
}

#[test]
fn keeps_generated_model_paths_separate_from_user_names() {
    let mut f = Fixture::new();
    let model = f.value["models"]["z_observation"].clone();
    f.value["models"] = yaml("{}");
    f.value["models"]["../outside"] = model;
    f.value["pipeline"] =
        yaml("{inputs: {'../outside': [joint_pos]}, outputs: {'../outside': [features]}}");
    let p = f.convert();
    assert_eq!(p.models[0].config.name, "leapp_node_0");
    assert_eq!(p.outputs[0].tensor_name, "../outside/features");
}

#[test]
fn distinguishes_input_and_output_tensors_with_the_same_name() {
    let mut f = Fixture::new();
    f.value["models"]["a_policy"]["outputs"][0]["name"] = yaml("state");
    f.value["pipeline"]["outputs"]["a_policy"] = yaml("[state, next_state]");
    let p = f.convert();
    let feedback = &p.feedback[0];
    assert_ne!(feedback.input_name, "a_policy/state");
    assert_eq!(p.outputs[0].tensor_name, "a_policy/state");
    let all_inputs: HashSet<_> = p.ensemble.input.iter().map(|p| &p.name).collect();
    assert!(p
        .ensemble
        .output
        .iter()
        .all(|p| !all_inputs.contains(&p.name)));
    assert_eq!(
        p.ensemble.ensemble_scheduling().step[1].input_map["state"],
        feedback.input_name
    );
}

#[test]
fn preserves_dynamic_shapes_and_selects_cuda_without_loading_it() {
    let mut f = Fixture::new();
    for model in f.value["models"].as_mapping_mut().unwrap().values_mut() {
        for direction in ["inputs", "outputs"] {
            for port in model[direction].as_sequence_mut().unwrap() {
                port["shape"] = yaml("[-1, 3]");
            }
        }
    }
    let options = ConversionOptions {
        device: ExecutionDevice::Cuda { device_id: 2 },
    };
    let p = convert_bundle(&f.open(), &options).unwrap();
    assert_eq!(p.models[0].config.input[0].dims, [-1, 3]);
    let group = &p.models[0].config.instance_group[0];
    assert_eq!(group.kind.enum_value().unwrap(), Kind::KIND_GPU);
    assert_eq!(group.gpus, [2]);
    assert_eq!(group.count, 1);
    let invalid = ConversionOptions {
        device: ExecutionDevice::Cuda {
            device_id: u32::MAX,
        },
    };
    assert_eq!(
        convert_bundle(&f.open(), &invalid).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

#[test]
fn reports_unsupported_shapes_and_empty_graphs() {
    for shape in ["[]", "[0]", "[-2]"] {
        let mut f = Fixture::new();
        for model in f.value["models"].as_mapping_mut().unwrap().values_mut() {
            for direction in ["inputs", "outputs"] {
                for port in model[direction].as_sequence_mut().unwrap() {
                    port["shape"] = yaml(shape);
                }
            }
        }
        f.unsupported("shape");
    }
    let mut f = Fixture::new();
    f.value = yaml("models: {}\npipeline: {inputs: {}, outputs: {}}\nsystem information: {}\n");
    f.unsupported("require a model");
}

#[test]
fn maps_every_bundle_dtype_to_its_native_triton_enum() {
    for (dtype, native) in [
        (DataType::Bool, TritonDataType::TYPE_BOOL),
        (DataType::Uint8, TritonDataType::TYPE_UINT8),
        (DataType::Uint16, TritonDataType::TYPE_UINT16),
        (DataType::Uint32, TritonDataType::TYPE_UINT32),
        (DataType::Uint64, TritonDataType::TYPE_UINT64),
        (DataType::Int8, TritonDataType::TYPE_INT8),
        (DataType::Int16, TritonDataType::TYPE_INT16),
        (DataType::Int32, TritonDataType::TYPE_INT32),
        (DataType::Int64, TritonDataType::TYPE_INT64),
        (DataType::Float16, TritonDataType::TYPE_FP16),
        (DataType::Float32, TritonDataType::TYPE_FP32),
        (DataType::Float64, TritonDataType::TYPE_FP64),
        (DataType::Bfloat16, TritonDataType::TYPE_BF16),
    ] {
        let port = PortSpec {
            name: "p".into(),
            dtype,
            shape: vec![3],
            tensor_type: "tensor".into(),
            metadata: Default::default(),
        };
        assert_eq!(
            config::input(&port, "node/p")
                .data_type
                .enum_value()
                .unwrap(),
            native
        );
        assert_eq!(
            config::output(&port, "node/p")
                .data_type
                .enum_value()
                .unwrap(),
            native
        );
    }
}

#[test]
fn forward_fanout_keeps_one_producer_tensor() {
    let mut f = Fixture::new();
    let mut input = f.value["models"]["a_policy"]["inputs"][0].clone();
    input["name"] = yaml("extra");
    f.value["models"]["a_policy"]["inputs"]
        .as_sequence_mut()
        .unwrap()
        .push(input);
    f.value["pipeline"]["data_flow"]["z_observation/features"] =
        yaml("[a_policy/features, a_policy/extra]");
    let p = f.convert();
    let step = &p.ensemble.ensemble_scheduling().step[1];
    assert_eq!(step.input_map["features"], step.input_map["extra"]);
    assert!(p.default_inputs.is_empty());
}

#[test]
fn qualified_names_cannot_alias_different_outputs() {
    let mut f = Fixture::new();
    let mut other = f.value["models"]["z_observation"].clone();
    other["outputs"][0]["name"] = yaml("port");
    f.value["models"] = yaml("{}");
    f.value["models"]["node/part"] = other.clone();
    other["outputs"][0]["name"] = yaml("part/port");
    f.value["models"]["node"] = other;
    f.value["pipeline"] = yaml("{inputs: {}, outputs: {'node/part': [port], node: ['part/port']}}");
    f.unsupported("ambiguous qualified output name");
}

#[test]
fn rejects_nul_tensor_names_and_models_without_outputs() {
    let mut f = Fixture::new();
    f.value["models"]["z_observation"]["inputs"][0]["name"] = Value::String("input\0name".into());
    f.unsupported("invalid Triton tensor name");
    let mut f = Fixture::new();
    let mut model = f.value["models"]["z_observation"].clone();
    model["outputs"] = yaml("[]");
    f.value["models"]["side_effect"] = model;
    f.unsupported("requires an output");
}

#[test]
fn qualified_names_cannot_alias_different_inputs() {
    let mut f = Fixture::new();
    let mut other = f.value["models"]["z_observation"].clone();
    other["inputs"][0]["name"] = yaml("port");
    f.value["models"] = yaml("{}");
    f.value["models"]["node/part"] = other.clone();
    other["inputs"][0]["name"] = yaml("part/port");
    f.value["models"]["node"] = other;
    f.value["pipeline"] =
        yaml("{inputs: {}, outputs: {'node/part': [features], node: [features]}}");
    f.unsupported("ambiguous qualified input name");
}

#[cfg(unix)]
#[test]
fn repository_serializes_configs_and_links_original_artifact_directories() {
    use super::repository::ModelRepository;

    let f = Fixture::new();
    let pipeline = f.convert();
    let weights = f.directory.path().join("weights");
    fs::create_dir(&weights).unwrap();
    fs::write(weights.join("parameters.bin"), b"external weights").unwrap();
    let repository = ModelRepository::create(&pipeline).unwrap();
    let root = repository.path().to_owned();
    for model in &pipeline.models {
        let directory = root.join(&model.config.name);
        let version = directory.join(model.version.to_string());
        assert!(fs::symlink_metadata(&version)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read_link(&version).unwrap(), f.directory.path());
        assert_eq!(
            fs::read(version.join(&model.config.default_model_filename)).unwrap(),
            b"abc"
        );
        assert_eq!(
            fs::read(version.join("weights/parameters.bin")).unwrap(),
            b"external weights"
        );
        let text = fs::read_to_string(directory.join("config.pbtxt")).unwrap();
        let decoded: ModelConfig = protobuf::text_format::parse_from_str(&text).unwrap();
        assert_eq!(decoded, model.config);
    }
    let ensemble = root.join(&pipeline.ensemble.name);
    let text = fs::read_to_string(ensemble.join("config.pbtxt")).unwrap();
    let decoded: ModelConfig = protobuf::text_format::parse_from_str(&text).unwrap();
    assert_eq!(decoded, pipeline.ensemble);
    assert!(ensemble.join("1").is_dir());
    assert!(!fs::symlink_metadata(ensemble.join("1"))
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(fs::read_dir(ensemble.join("1")).unwrap().count(), 0);
    assert!(!f.directory.path().join("config.pbtxt").exists());
    drop(repository);
    assert!(!root.exists());
    assert_eq!(
        fs::read(f.directory.path().join("model.bin")).unwrap(),
        b"abc"
    );
    assert_eq!(
        fs::read(weights.join("parameters.bin")).unwrap(),
        b"external weights"
    );
}

#[cfg(unix)]
#[test]
fn repositories_are_independent_and_use_the_selected_model_version() {
    use super::repository::ModelRepository;

    let f = Fixture::new();
    let mut pipeline = f.convert();
    pipeline.models[0].version = 7;
    pipeline.models[0]
        .config
        .version_policy
        .as_mut()
        .unwrap()
        .mut_specific()
        .versions = vec![7];
    pipeline.ensemble.mut_ensemble_scheduling().step[0].model_version = 7;
    let first = ModelRepository::create(&pipeline).unwrap();
    let second = ModelRepository::create(&pipeline).unwrap();
    assert_ne!(first.path(), second.path());
    let model = first.path().join(&pipeline.models[0].config.name);
    assert!(model.join("7").is_dir());
    assert!(!model.join("1").exists());
    drop(first);
    assert!(second
        .path()
        .join(&pipeline.ensemble.name)
        .join("config.pbtxt")
        .is_file());
}

#[cfg(unix)]
#[test]
fn repository_errors_identify_the_phase_and_reject_escaping_names() {
    use super::repository::ModelRepository;

    let f = Fixture::new();
    for name in [
        "",
        ".",
        "..",
        "../outside",
        "nested/model",
        "model/",
        "/absolute",
    ] {
        let mut pipeline = f.convert();
        pipeline.models[0].config.name = name.into();
        let error = ModelRepository::create(&pipeline).unwrap_err();
        assert_eq!(error.phase(), "repository");
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert!(error
            .to_string()
            .contains("invalid Triton model directory name"));
    }
    let mut pipeline = f.convert();
    pipeline.models[1].config.name = pipeline.models[0].config.name.clone();
    let error = ModelRepository::create(&pipeline).unwrap_err();
    assert_eq!(error.phase(), "repository");
    assert_eq!(error.code(), ErrorCode::Io);
    assert!(error.to_string().contains("create model directory"));
    for version in [0, -1] {
        pipeline.models[0].version = version;
        let error = ModelRepository::create(&pipeline).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert!(error.to_string().contains("invalid Triton model version"));
    }
}

#[test]
fn converts_pt2_and_alias_with_onnx_in_one_ensemble() {
    for backend in ["pt2", "exported-program"] {
        let mut f = Fixture::new();
        f.value["models"]["a_policy"]["parameters"]["backend"] = yaml(backend);
        let p = f.convert();
        let backends: Vec<_> = p.models.iter().map(|m| m.config.backend.as_str()).collect();
        assert!(backends.contains(&"leapp_pt2"));
        assert!(backends.contains(&"onnxruntime"));
        assert_eq!(p.ensemble.ensemble_scheduling().step.len(), 2);
    }
}

#[test]
fn converts_tensorrt_aliases_with_gpu_placement() {
    for backend in ["trt", "tensorrt"] {
        let mut f = Fixture::new();
        let parameters = &mut f.value["models"]["a_policy"]["parameters"];
        parameters["backend"] = yaml(backend);
        parameters["model_path"] = yaml("policy.engine");
        fs::copy(
            f.directory.path().join("model.bin"),
            f.directory.path().join("policy.engine"),
        )
        .unwrap();
        f.unsupported("TensorRT requires CUDA execution");
        let options = ConversionOptions {
            device: ExecutionDevice::Cuda { device_id: 2 },
        };
        let p = convert_bundle(&f.open(), &options).unwrap();
        let config = &p
            .models
            .iter()
            .find(|m| m.config.backend == "tensorrt")
            .unwrap()
            .config;
        assert_eq!(config.platform, "tensorrt_plan");
        assert_eq!(config.default_model_filename, "policy.engine");
        assert_eq!(config.max_batch_size, 0);
        assert_eq!(
            config.instance_group[0].kind.enum_value().unwrap(),
            Kind::KIND_GPU
        );
        assert_eq!(config.instance_group[0].gpus, [2]);
        assert!(config.instance_group[0].profile.is_empty());
        assert!(p.models.iter().any(|m| m.config.backend == "onnxruntime"));
        assert_eq!(p.ensemble.ensemble_scheduling().step.len(), 2);
    }
}

#[test]
fn torchscript_ports_use_ordinals_without_changing_public_or_feedback_names() {
    for backend in ["jit", "jit-script", "jit-trace"] {
        for device in [ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 2 }] {
            let mut f = Fixture::new();
            f.value["models"]["a_policy"]["parameters"]["backend"] = yaml(backend);
            let p = convert_bundle(&f.open(), &ConversionOptions { device }).unwrap();
            let config = &p.models[1].config;
            assert_eq!(config.backend, "leapp_torchscript");
            assert!(config.platform.is_empty());
            assert_eq!(config.default_model_filename, "model.bin");
            assert_eq!(config.max_batch_size, 0);
            assert_eq!(
                config
                    .input
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>(),
                ["INPUT__0", "INPUT__1"]
            );
            assert_eq!(
                config
                    .output
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>(),
                ["OUTPUT__0", "OUTPUT__1"]
            );
            assert!(config.parameters.is_empty());
            assert!(config.model_operations.is_none());
            let step = &p.ensemble.ensemble_scheduling().step[1];
            assert_eq!(step.input_map["INPUT__0"], "z_observation/features");
            assert_eq!(step.input_map["INPUT__1"], p.feedback[0].input_name);
            assert_eq!(step.output_map["OUTPUT__0"], "a_policy/actions");
            assert_eq!(step.output_map["OUTPUT__1"], "a_policy/next_state");
            assert_eq!(p.outputs[0].port.name(), "a_policy/next_state");
            assert_eq!(
                p.ensemble.ensemble_scheduling().step[0].output_map["features"],
                "z_observation/features"
            );
            match device {
                ExecutionDevice::Cpu => assert_eq!(
                    config.instance_group[0].kind.enum_value().unwrap(),
                    Kind::KIND_CPU
                ),
                ExecutionDevice::Cuda { .. } => assert_eq!(config.instance_group[0].gpus, [2]),
            }
        }
    }
}
