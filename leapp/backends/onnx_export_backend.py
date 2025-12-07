import torch
from torch.onnx import ONNXProgram
from typing import Tuple
from leapp.backends.export_backend import ExportBackend
from leapp.backends.module_builder import ModuleBuilder
from leapp.utils import resolve_tensor_descriptions_to_values
from leapp._logging import _get_logger
import os
import onnx
import tempfile


class SimplifiedONNXProgram:
    """Wrapper for ONNX models exported via legacy torch.onnx.export"""

    def __init__(self, onnx_model):
        self.model_proto = onnx_model

    def save(self, destination, include_initializers=True, keep_initializers_as_inputs=False):
        """Save the ONNX model to disk"""
        onnx.save(self.model_proto, destination)

    def __call__(self, *args):
        raise NotImplementedError(
            "SimplifiedONNXProgram does not support __call__")


class ONNXExportBackend(ExportBackend):
    def get_backed_model_type(self):
        return "onnx"

    def compile_dynamo(self, m):
        input_values = tuple([resolve_tensor_descriptions_to_values(param_format)
                              for param_format in self.node_context.input_formats])
        onnx_program = torch.onnx.export(
            m,
            input_values,
            None,  # no need to save the model
            dynamo=True,
            input_names=[
                param_format.name for param_format in self.node_context.input_formats],
            output_names=[
                param_format.name for param_format in self.node_context.output_formats],

            verify=self.backend_params.get('verify', True),
            optimize=self.backend_params.get('optimize', True),

            verbose=_get_logger().is_verbose(),
            report=self.backend_params.get('report', False),
            fallback=self.backend_params.get('fallback', None),
            opset_version=self.backend_params.get('opset_version', None),
        )

        return onnx_program

    def compile_torchscript(self, m):
        input_values = tuple([resolve_tensor_descriptions_to_values(param_format)
                              for param_format in self.node_context.input_formats])

        m = torch.jit.script(m)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.onnx")
            torch.onnx.export(
                m,
                input_values,
                save_path,
                dynamo=False,
                input_names=[
                    param_format.name for param_format in self.node_context.input_formats],
                output_names=[
                    param_format.name for param_format in self.node_context.output_formats],

                verbose=_get_logger().is_verbose(),
                report=self.backend_params.get('report', False),
                opset_version=self.backend_params.get('opset_version', None),
            )
            onnx_model = onnx.load(save_path)

        # we should convert to a ONNXProgram object for unity and convenience when testing with this model
        onnx_program = SimplifiedONNXProgram(onnx_model)

        return onnx_program

    def compile(self):
        builder = ModuleBuilder(self.node_context)
        m = builder.module_instance.eval()
        if self.backend_params.get('prescript', False):
            m = torch.jit.script(m)

        if self.backend_params.get('dynamo', True):
            onnx_model = self.compile_dynamo(m)
        else:
            onnx_model = self.compile_torchscript(m)
        return onnx_model

    def save(self, save_path: str, compiled_model) -> Tuple[str, str, str]:
        onnx_path = os.path.join(save_path, f"{self.node_context.name}.onnx")

        try:
            onnx.checker.check_model(compiled_model.model_proto)
        except onnx.shape_inference.InferenceError as e:
            _get_logger().error(f"Error checking ONNX model: {e}")
            return None, None, None

        # Use ONNXProgram's save method
        compiled_model.save(
            onnx_path,
            include_initializers=True,
            keep_initializers_as_inputs=False,
        )
        md5sum, sha256sum = self._verify_model_location_and_get_hash(onnx_path)

        return onnx_path, md5sum, sha256sum
