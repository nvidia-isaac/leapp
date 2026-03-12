import torch
from typing import Tuple
from leapp.backends.export_backend import ExportBackend, prepare_tensors_for_export, SimplifiedONNXProgram
from leapp.utils.logging import _get_logger
import os
import onnx
from onnx import numpy_helper
import tempfile

from torch.onnx import _constants

class ONNXExportBackend(ExportBackend):
    def get_backend_metadata(self):
        metadata = {}
        metadata['opset_version'] = getattr(self, 'opset_version', _constants.ONNX_DEFAULT_OPSET)
        return metadata

    def get_backend_model_type(self):
        return "onnx"
    
    def _get_onnx_model(self, onnx_program):
        """Get the ONNX model proto from an ONNX program."""
        if hasattr(onnx_program, 'model_proto'):
            # dynamo export returns ONNXProgram with model_proto
            return onnx_program.model_proto
        else:
            # SimplifiedONNXProgram - load from file
            model_path = os.path.join(onnx_program._source_dir, onnx_program._source_filename)
            return onnx.load(model_path)

    def _get_actual_onnx_inputs(self, onnx_model):
        """Get actual input names from ONNX model, excluding initializers (constants)."""
        # Initializers are constants baked into the model - exclude them
        initializer_names = {init.name for init in onnx_model.graph.initializer}
        
        # Return only true dynamic inputs
        return [inp.name for inp in onnx_model.graph.input if inp.name not in initializer_names]

    def _handle_duplicate_io_names(self):
        """Rename overlapping input/output names for ONNX's flat I/O namespace."""
        input_names = [td.name for td in self.node_context.inputs]
        output_names = [td.name for td in self.node_context.outputs]
        overlaps = sorted(set(input_names) & set(output_names))
        if not overlaps:
            return

        _get_logger().warning(
            f"[{self.node_context.name}] Renaming overlapping ONNX input/output names: {overlaps}"
        )
        used_names = set(input_names) | set(output_names)
        for name in overlaps:
            new_input_name = f"{name}_in"
            new_output_name = f"{name}_out"
            if new_input_name in used_names or new_output_name in used_names:
                raise ValueError(
                    f"[{self.node_context.name}] Cannot resolve overlapping ONNX I/O name '{name}' "
                    f"because '{new_input_name}' or '{new_output_name}' is already in use."
                )
            self.node_context.change_input_name(name, new_input_name)
            self.node_context.change_output_name(name, new_output_name)
            used_names.remove(name)
            used_names.add(new_input_name)
            used_names.add(new_output_name)

    def _sync_inputs_with_onnx(self, onnx_program):
        """Sync inputs with what ONNX actually exported.
        
        ONNX export can remove unused inputs (constant folding, dead code elimination).
        This is allowed and we update node_context.inputs accordingly.
        
        However, if ONNX renames inputs (which happens with identity/passthrough functions
        when using onnx-dynamo), we raise an error and suggest using onnx-torchscript instead.
        """
        onnx_model = self._get_onnx_model(onnx_program)
        
        expected_input_names = set(td.name for td in self.node_context.inputs)
        actual_input_names = self._get_actual_onnx_inputs(onnx_model)
        actual_input_set = set(actual_input_names)
        
        # Check for removed inputs (this is OK - ONNX optimized them away)
        removed_inputs = [td.name for td in self.node_context.inputs if td.name not in actual_input_set]
        
        # Check for renamed/new inputs (this is NOT OK - we can't track them)
        renamed_inputs = [name for name in actual_input_names if name not in expected_input_names]
        
        if renamed_inputs:
            raise ValueError(
                f"[{self.node_context.name}] ONNX export renamed inputs: {renamed_inputs}. "
                "This typically happens with identity/passthrough functions with onnx-dynamo. "
                "This usecase is not supported by LEAPP, please: \n"
                "1. use the onnx-torchscript export backend instead\n"
                "2. use a different backend, such as torch-script or torch-trace \n"
                "3. remove the inputs that are not used in the computation or directly returned as output"
            )
        
        if removed_inputs:
            _get_logger().warning(
                f"[{self.node_context.name}] ONNX export optimized away {len(removed_inputs)} inputs: {removed_inputs}. "
                f"These inputs were unused or became constants. Updating node to match."
            )
            # Track trimmed inputs
            self.node_context.trimmed_inputs.update(removed_inputs)
            
            # Filter inputs to only keep actual ONNX inputs (preserving order)
            self.node_context.inputs = [
                td for td in self.node_context.inputs if td.name in actual_input_set
            ]


    def load(self, model_path: str, sha256sum: str, device: str):
        self._load_onnx(model_path, sha256sum, device)

    @staticmethod
    def _fix_scalar_slice_inputs(model: onnx.ModelProto) -> int:
        """Fix scalar initializers used as Slice starts/ends/axes/steps.

        The ONNX spec requires Slice inputs (starts, ends, axes, steps) to be
        1-D tensors.  However, the dynamo-based ONNX exporter sometimes reuses
        scalar (0-D) initializers that are also shared with Gather nodes (which
        *do* need scalars).  This creates an invalid Slice node.

        The fix creates **new** 1-D copies of the scalar initializer for each
        affected Slice input, leaving the original scalar untouched for other
        consumers (Gather, etc.).

        Modifies the model proto in-place.
        """
        init_map = {init.name: init for init in model.graph.initializer}
        num_fixed = 0

        for node in model.graph.node:
            if node.op_type != "Slice":
                continue
            for i in range(1, len(node.input)):  # skip input[0] (data)
                inp_name = node.input[i]
                if inp_name in init_map:
                    arr = numpy_helper.to_array(init_map[inp_name])
                    if arr.ndim == 0:
                        role = ["data", "starts", "ends", "axes", "steps"][i] if i < 5 else f"input_{i}"
                        new_name = f"{inp_name}_1d_{node.name}_{role}"
                        new_tensor = numpy_helper.from_array(
                            arr.reshape(1), name=new_name
                        )
                        model.graph.initializer.append(new_tensor)
                        node.input[i] = new_name
                        num_fixed += 1

        if num_fixed > 0:
            _get_logger().debug(
                f"Fixed {num_fixed} scalar Slice initializer(s) "
                f"(ONNX exporter bug: shared scalar initializers between Gather and Slice nodes)")

    def save(self, save_path: str) -> Tuple[str, str, str]:
        onnx_path = os.path.join(save_path, f"{self.node_context.name}.onnx")

        # Use ONNXProgram's save method
        self.compiled_model.save(
            onnx_path,
            include_initializers=True,
            keep_initializers_as_inputs=False,
        )

        # Validate model (skip for large models as it can fail with protobuf limits)
        skip_validation = self.backend_params.get('skip_validation', False)
        if not skip_validation:
            try:
                # Try to check model, but don't fail the export if validation fails
                onnx.checker.check_model(onnx_path)
            except Exception as e:
                _get_logger().error(
                    f"ONNX model validation warning (model still saved): {e}")

        md5sum, sha256sum = self._verify_model_location_and_get_hash(onnx_path)

        return onnx_path, md5sum, sha256sum

    def compile(self,  m: torch.nn.Module = None) -> SimplifiedONNXProgram:
        raise NotImplementedError(
            "TorchExportBackend does not support compilation, please use torch-script or torch-trace instead")
        return None

class ONNXTorchScriptExportBackend(ONNXExportBackend):

    def compile(self, m: torch.nn.Module = None):
        if m is None:
            m = self.module_builder()
        m = m.eval()
        self._handle_duplicate_io_names()
        
        # Optionally pre-script the module before ONNX export
        # This is useful when using traced models as environment constants
        if self.backend_params.get('prescript', False):
            m = torch.jit.script(m)
        
        # Get flat tensor values directly from inputs (not input_formats which preserves nested structure)
        input_values = tuple(
            [tensor_desc.value for tensor_desc in self.node_context.inputs])
        # Clone tensors to escape inference mode (inference tensors can't participate in autograd)
        input_values = prepare_tensors_for_export(input_values)

        # Create temp directory manually (not context manager) so we can control its lifetime
        tmpdir = tempfile.mkdtemp()
        save_path = os.path.join(tmpdir, "model.onnx")
        self.opset_version = self.backend_params.get('opset_version', _constants.ONNX_DEFAULT_OPSET)

        torch.onnx.export(
            m,
            input_values,
            save_path,
            dynamo=False,
            input_names=[
                tensor_desc.name for tensor_desc in self.node_context.inputs],
            output_names=[
                tensor_desc.name for tensor_desc in self.node_context.outputs],

            verbose=_get_logger().is_verbose(),
            report=self.backend_params.get('report', False),
            opset_version=self.opset_version,
        )

        # Create program from file path
        # Temp dir will be cleaned up when program is deleted
        onnx_program = SimplifiedONNXProgram(save_path, temp_dir=tmpdir)

        # Sync inputs with what ONNX actually exported
        # (ONNX can remove unused inputs during optimization)
        self._sync_inputs_with_onnx(onnx_program)

        self.compiled_model = onnx_program
        self.compiled_module = m

class ONNXDynamoExportBackend(ONNXExportBackend):
    def compile(self, m: torch.nn.Module = None):
        if m is None:
            m = self.module_builder()
        m = m.eval()
        self._handle_duplicate_io_names()
        # Get flat tensor values directly from inputs (not input_formats which preserves nested structure)
        input_values = tuple(
            [tensor_desc.value for tensor_desc in self.node_context.inputs])
        # Clone tensors to escape inference mode (inference tensors can't participate in autograd)
        input_values = prepare_tensors_for_export(input_values)
        self.opset_version = self.backend_params.get('opset_version', _constants.ONNX_DEFAULT_OPSET)
        onnx_program = torch.onnx.export(
            m,
            input_values,
            None,  # no need to save the model
            dynamo=True,
            input_names=[
                tensor_desc.name for tensor_desc in self.node_context.inputs],
            output_names=[
                tensor_desc.name for tensor_desc in self.node_context.outputs],

            verify=self.backend_params.get('verify', False),
            optimize=self.backend_params.get('optimize', True),

            verbose=_get_logger().is_verbose(),
            report=self.backend_params.get('report', False),
            fallback=self.backend_params.get('fallback', None),
            opset_version=self.opset_version,
        )
        
        # Sync inputs with what ONNX actually exported
        # (ONNX can remove unused inputs during optimization)
        self._sync_inputs_with_onnx(onnx_program)

        # Capture proto once — model_proto may be a property that rebuilds on
        # each access, so we must fix and save from the same object.
        model_proto = onnx_program.model_proto

        # Fix scalar Slice initializers on the in-memory proto before saving
        # (ONNX exporter bug: shared scalar initializers between Gather and Slice nodes).
        self._fix_scalar_slice_inputs(model_proto)

        # Wrap in SimplifiedONNXProgram so validation uses controlled
        # ORT session options (ORT_ENABLE_BASIC avoids graph-opt corruption).
        tmpdir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmpdir, f"{self.node_context.name}.onnx")

        PROTOBUF_LIMIT = 1.5 * 1024 * 1024 * 1024  # 1.5 GB — headroom below 2 GB protobuf limit
        estimated_size = sum(
            numpy_helper.to_array(init).nbytes
            for init in model_proto.graph.initializer
        )
        if estimated_size > PROTOBUF_LIMIT:
            data_filename = f"{self.node_context.name}.onnx.data"
            onnx.save(
                model_proto,
                tmp_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_filename,
                size_threshold=1024,
                convert_attribute=True,
            )
        else:
            onnx.save(model_proto, tmp_path)

        self.compiled_model = SimplifiedONNXProgram(tmp_path, temp_dir=tmpdir)
        self.compiled_module = m