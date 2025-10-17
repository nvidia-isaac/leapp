import abc
from typing import Tuple
import os
import hashlib
import shutil


class ExportBackend(abc.ABC):
    def __init__(self, node_context, logger, backend_params=None):
        self.node_context = node_context
        self.logger = logger
        if backend_params is None:
            self.backend_params = {}
        else:
            self.backend_params = backend_params

    def _verify_model_location_and_get_md5sum(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, 'rb') as f:
            md5sum = hashlib.md5(f.read()).hexdigest()
        return md5sum

    def _copy_model_to_path(self, model_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(model_path):
            return None

        # Check if save_path is the same as the directory containing model_path
        model_dir = os.path.dirname(os.path.abspath(model_path))
        save_dir = os.path.abspath(save_path)
        if model_dir == save_dir:
            # No need to copy if already in the same directory
            return model_path

        # Get the filename from model_path
        filename = os.path.basename(model_path)
        # Create the full destination path
        dest_path = os.path.join(save_path, filename)
        # Copy the file
        shutil.copy2(model_path, dest_path)

        return dest_path

    @abc.abstractmethod
    def get_backed_model_type(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, save_path: str, func: callable, **kwargs) -> Tuple[str, str]:
        raise NotImplementedError
        return None, None


class NoneExportBackend(ExportBackend):
    def __call__(self, save_path: str, **kwargs) -> Tuple[str, str]:
        if "model_path" not in self.backend_params:
            self.logger.warning(
                f"No model path provided for {self.node_context.name}")
            self.logger.warning("if this is intentional, please provide a path to the correct model "
                                "in the generated yaml file. Otherwise, please manually fill in the backend parameters.")
            return None, None

        md5sum = self._verify_model_location_and_get_md5sum(
            self.backend_params['model_path'])
        model_path = self.backend_params['model_path']
        if "copy_original_model" in self.backend_params and self.backend_params['copy_original_model'] is False:
            model_path = self._copy_model_to_path(model_path, save_path)

        return model_path, md5sum

    def get_backed_model_type(self):
        if "model_path" not in self.backend_params:
            return None

        path = self.backend_params['model_path']
        suffix = path.split('.')[-1]
        if suffix == 'pt' or suffix == 'pt2':
            return "torch"
        elif suffix == 'onnx':
            return "onnx"
        elif suffix == 'cpp' or suffix == "cc":
            return "cpp"
        elif suffix == 'py':
            return "py"
        elif suffix == 'engine':
            return 'trt'
        else:
            raise Exception(
                f"Unsupported model file suffix: {suffix}")
