"""Run in a fresh process with PYTHONPATH selecting the baseline or adapter tree."""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import leapp
from leapp.leapp_graph.datatypes import TracedNpArray
from leapp.leapp_graph.traced_node import TracedTensorNode

A = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]], dtype=np.float32)
CASES = {
    "control": lambda x: np.sin(x) + np.mean(x, axis=-1, keepdims=True),
    "inv": np.linalg.inv,
    "norm": lambda x: np.linalg.norm(x, axis=-1, keepdims=True),
    "det": np.linalg.det,
    "slogdet": lambda x: tuple(np.linalg.slogdet(x)),
    "cholesky": np.linalg.cholesky,
    "solve": lambda x: np.linalg.solve(x, np.ones(3, dtype=np.float32)),
    "matrix_power": lambda x: np.linalg.matrix_power(x, 3),
    "svd_values": lambda x: np.linalg.svd(x, compute_uv=False),
    "eigvalsh": np.linalg.eigvalsh,
    "pinv": np.linalg.pinv,
    "cross": lambda x: np.cross(x, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    "isclose": lambda x: np.isclose(x, 3.0),
    "isclose_where": lambda x: np.where(np.isclose(x, 3.0), x, -x),
    "sinc": np.sinc,
    "cbrt": np.cbrt,
    "degrees": np.degrees,
    "radians": np.radians,
    "logaddexp": lambda x: np.logaddexp(x, x + 1),
    "heaviside": lambda x: np.heaviside(x - 4, 0.5),
    "copysign": lambda x: np.copysign(x, -1.0),
    "diff": lambda x: np.diff(x, axis=-1),
    "gradient": lambda x: tuple(np.gradient(x)),
    "repeat": lambda x: np.repeat(x, 2, axis=1),
    "tile": lambda x: np.tile(x, (2, 1)),
    "pad": lambda x: np.pad(x, ((1, 1), (0, 1)), mode="constant"),
    "take": lambda x: np.take(x, [0, 2], axis=1),
    "take_along_axis": lambda x: np.take_along_axis(
        x, np.array([[2, 0], [0, 1], [1, 2]]), axis=1
    ),
    "broadcast_to": lambda x: np.broadcast_to(x, (2, 3, 3)),
    "median": lambda x: np.median(x, axis=1),
    "quantile": lambda x: np.quantile(x, 0.25, axis=1),
    "ptp": lambda x: np.ptp(x, axis=1),
    "nanmean": lambda x: np.nanmean(x, axis=1),
    "unique": np.unique,
    "fft": np.fft.fft,
    "scalar_dependency": lambda x: x + x[0, 0],
    "reverse": lambda x: x[::-1],
    "integer_mean": lambda x: np.mean(x, axis=-1, keepdims=True),
}


def leaves(x):
    if isinstance(x, (tuple, list)):
        return [v for item in x for v in leaves(item)]
    return [x]


def compare(actual, expected):
    actual, expected = leaves(actual), leaves(expected)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(
            a, torch.from_numpy(np.array(e, copy=True)), equal_nan=True
        )


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "leapp": leapp.__file__,
        },
        "cases": {},
    }
    for name, fn in CASES.items():
        x = A.astype(np.int32) if name == "integer_mean" else A.copy()
        corner = x.copy()
        corner[0, 0] += 4
        samples = (x, x + 2, corner)
        record = results["cases"][name] = {}
        try:
            ctx = TracedTensorNode(name="probe", node_index=0)
            y = fn(ctx.create_input(x.copy(), name="x"))
            if not all(isinstance(v, TracedNpArray) for v in leaves(y)):
                raise RuntimeError("untraced output")
            ctx.compile_trace({f"y{i}": v for i, v in enumerate(leaves(y))})
            (out / f"{name}.fx.py").write_text(ctx.m.code)
        except Exception as exc:
            record["capture"] = f"{type(exc).__name__}: {exc}"
            continue
        for backend in ("fx", "jit_script", "jit_trace", "pt2"):
            try:
                if backend == "fx":
                    m = ctx.m
                elif backend == "jit_script":
                    m = torch.jit.script(ctx.m)
                elif backend == "jit_trace":
                    m = torch.jit.trace(ctx.m, (torch.from_numpy(x.copy()),))
                else:
                    ep = torch.export.export(
                        ctx.m, (torch.from_numpy(x.copy()),), strict=False
                    )
                    m = ep.module()
                    (out / f"{name}.pt2.py").write_text(ep.graph_module.code)
                for sample in samples:
                    compare(m(torch.from_numpy(sample.copy())), fn(sample.copy()))
                record[backend] = "pass"
            except Exception as exc:
                record[backend] = f"{type(exc).__name__}: {exc}"
        print(name, {k: v[:160] for k, v in record.items()}, flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
