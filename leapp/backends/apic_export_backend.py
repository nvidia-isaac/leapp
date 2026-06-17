from leapp.backends.export_backend import ExportBackend
from typing import Any
try:
    import warp as wp
    import warp.Graph
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise
    wp = None
    APICExportBackend = None
else:
    class APICExportBackend(ExportBackend):
        _default_sm_version = 76
        def compile(self, m: Any):
            # compile doesn't do anything yet because warp.context.Graph is already compiled
            pass
        def save(self, m: "warp.context.Graph"):
            sm_version = getattr(self, 'sm_version', self._default_sm_version)
            capture_save = wp.capture_save(m, )
