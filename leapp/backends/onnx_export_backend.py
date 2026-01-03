import torch
from typing import Tuple
from leapp.backends.export_backend import ExportBackend, SimplifiedONNXProgram, prepare_tensors_for_export
from leapp._logging import _get_logger
import os
import onnx
import tempfile


class ONNXExportBackend(ExportBackend):
    def get_backed_model_type(self):
        return "onnx"

    def compile_dynamo(self, m):
        # Get flat tensor values directly from inputs (not input_formats which preserves nested structure)
        input_values = tuple([tensor_desc.value for tensor_desc in self.node_context.inputs])
        # Clone tensors to escape inference mode (inference tensors can't participate in autograd)
        input_values = prepare_tensors_for_export(input_values)
        onnx_program = torch.onnx.export(
            m,
            input_values,
            None,  # no need to save the model
            dynamo=True,
            input_names=[
                tensor_desc.name for tensor_desc in self.node_context.inputs],
            output_names=[
                tensor_desc.name for tensor_desc in self.node_context.outputs],

            verify=self.backend_params.get('verify', True),
            optimize=self.backend_params.get('optimize', True),

            verbose=_get_logger().is_verbose(),
            report=self.backend_params.get('report', False),
            fallback=self.backend_params.get('fallback', None),
            opset_version=self.backend_params.get('opset_version', None),
        )

        return onnx_program

    def compile_torchscript(self, m):
        # Get flat tensor values directly from inputs (not input_formats which preserves nested structure)
        input_values = tuple([tensor_desc.value for tensor_desc in self.node_context.inputs])
        # Clone tensors to escape inference mode (inference tensors can't participate in autograd)
        input_values = prepare_tensors_for_export(input_values)

        # Create temp directory manually (not context manager) so we can control its lifetime
        tmpdir = tempfile.mkdtemp()
        save_path = os.path.join(tmpdir, "model.onnx")
        
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
            opset_version=self.backend_params.get('opset_version', None),
        )

        # Create program from file path
        # Temp dir will be cleaned up when program is deleted
        onnx_program = SimplifiedONNXProgram(save_path, temp_dir=tmpdir)

        return onnx_program

    def compile(self):
        m = self.module_builder().eval()
        if self.backend_params.get('prescript', False): #TODO: make this conditional upon use_trace
            m = torch.jit.script(m)

        if self.backend_params.get('dynamo', True):
            onnx_model = self.compile_dynamo(m)
        else:
            onnx_model = self.compile_torchscript(m)
        return onnx_model

    def save(self, save_path: str, compiled_model) -> Tuple[str, str, str]:
        onnx_path = os.path.join(save_path, f"{self.node_context.name}.onnx")

        # Use ONNXProgram's save method
        compiled_model.save(
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
                _get_logger().error(f"ONNX model validation warning (model still saved): {e}")
        
        md5sum, sha256sum = self._verify_model_location_and_get_hash(onnx_path)

        return onnx_path, md5sum, sha256sum
