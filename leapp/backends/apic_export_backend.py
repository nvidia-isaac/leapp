from leapp.backends.export_backend import ExportBackend
from typing import Any

from leapp.leapp_graph.datatypes.warp import wp

if wp is None:
    APICExportBackend = None
else:
    import warp.Graph  # noqa: F401

    class APICExportBackend(ExportBackend):
        _default_sm_version = 76
        def compile(self, m: Any):
            # compile doesn't do anything yet because warp.context.Graph is already compiled
            pass
        def save(self, m: "warp.context.Graph"):
            sm_version = getattr(self, 'sm_version', self._default_sm_version)
            capture_save = wp.capture_save(m, )
