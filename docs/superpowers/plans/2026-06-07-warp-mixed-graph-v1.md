# Unified Non-Invasive torch↔warp Mixed Graph (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user mark only input/output tensors with the existing `annotate.input_tensors`/`output_tensors`, write ordinary mixed PyTorch + Warp code in between, and have LEAPP automatically split the region into native single-kind nodes (torch→ONNX/TorchScript, warp→`.wrp`) wired by auto-discovered edges — proven end-to-end on the Isaac Lab root-quat→projected-gravity warp kernel via the existing `InferenceManager`.

**Architecture:** Edges in LEAPP form by `leapp_tag` identity carried on tensors. We monkeypatch `wp.from_torch`/`wp.to_torch` (the *only* v1 bridge) during `leapp.start()`. A torch→warp crossing finalizes the open torch segment (`TracedTensorNode.compile_trace`) with the bridged tensor as its output, then opens a `WarpRegionNode`; a warp→torch crossing finalizes the warp node (replaying recorded `wp.launch` calls into one `wp.ScopedCapture(apic=True)`) and returns a fresh TracedTensor that re-enters torch tracing. `annotate.input_tensors`/`output_tensors` keep their signatures but resolve a region base-name to its currently-open segment node. Warp nodes deploy as native `.wrp` ([ADR-0001](../../adr/0001-warp-native-wrp-not-onnx.md)); only linear segment chains are supported in v1 ([ADR-0002](../../adr/0002-linear-segment-chain-v1.md)).

**Tech Stack:** Python 3.12, PyTorch 2.7 (`torch.fx`, `__torch_function__`), NVIDIA Warp 1.14 (APIC `wp.ScopedCapture(apic=True)`, `wp.capture_save/load`), pytest. Tests requiring a GPU/warp are guarded and skip cleanly where CUDA/warp is absent.

**How to run the test suite (this machine):**
`PYTHONPATH=$PWD python3.12 -m pytest <path> -q` (default python3.12 has torch+warp 1.14; live-Triton tier not needed for this plan).

---

## Scope

In scope (one working vertical slice): structured warp dtypes; a graph-registered `WarpRegionNode`; bridge interception + auto-split state machine; `start()/stop()` wiring; end-to-end mixed graph validated through the Python `InferenceManager`.

Out of scope (separate downstream plans): the Triton repo generator / live Triton run for auto-split graphs (the standalone warp path already covers Triton; the generator change is the handoff's separate item); native warp-resident state; intra-region DAGs; DLPack/`ptr` bridges; batching; Jetson.

## File Structure

- **Create `leapp/backends/warp_dtypes.py`** — single source of truth mapping warp dtypes (scalar *and* structured: `vec3f`, `vec4f`, `quatf`, `transformf`, `mat33f`, …) ↔ a string name, plus each dtype's scalar base, scalar count, and the torch-view trailing shape. Pure data + helpers, no warp graph logic. Unit-testable without a GPU.
- **Modify `leapp/backends/warp_export_backend.py`** — consume `warp_dtypes`; carry `warp_dtype` (struct name) alongside the torch-view `dtype`/`shape` in the sidecar meta and port descriptions; fix `_WarpGraphCallable` to reshape per dtype and allow the *declared* struct reinterpret.
- **Create `leapp/backends/warp_capture_core.py`** — extract the reusable "replay recorded launches into one APIC capture" + device-resolution helpers so both the standalone `warp_node` and the new `WarpRegionNode` share one implementation.
- **Modify `leapp/backends/warp_capture.py`** — re-implement the standalone `warp_node` context manager on top of `warp_capture_core` (behavior unchanged; the ptr-order I/O heuristic stays as the standalone fallback).
- **Create `leapp/leapp_graph/warp_region_node.py`** — `WarpRegionNode(LeappNode)`: holds recorded launches, device, and bridged I/O (`wp.array` + the torch tensor whose `leapp_tag` it inherits); `compile_model()` builds the `.wrp` and wires a `WarpExportBackend`; `get_description()`/validation reuse the base class.
- **Create `leapp/warp_bridge.py`** — the bridge interceptor + `RegionSegmenter` state machine. `install()/uninstall()` patch the `wp.from_torch`/`wp.to_torch` symbols; the segmenter finalizes/opens segment nodes and enforces the linear-chain + fail-loud rules.
- **Modify `leapp/export_manager.py`** — add a region→open-segment resolution map so `input_tensors`/`output_tensors` route to the current segment; expose the small hooks the segmenter calls (`_finalize_torch_segment`, `_register_warp_node`, `_open_continuation_torch_segment`).
- **Modify `leapp/leapp.py`** — `start()` installs the bridge (only if warp importable); `stop()` uninstalls it.
- **Tests** under `tests/functional_tests/`: `test_warp_dtypes.py`, `test_warp_region_node.py`, `test_warp_bridge.py`, `test_mixed_autosplit.py`, `test_projected_gravity_mixed.py`.

---

## Phase A — Structured warp dtypes (self-contained, no auto-split)

### Task A1: Warp dtype registry

**Files:**
- Create: `leapp/backends/warp_dtypes.py`
- Test: `tests/functional_tests/test_warp_dtypes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/functional_tests/test_warp_dtypes.py
import pytest
wp = pytest.importorskip("warp")
from leapp.backends import warp_dtypes as wd


def test_scalar_roundtrip():
    assert wd.warp_dtype_to_str(wp.float32) == "float32"
    assert wd.str_to_warp_dtype("float32") is wp.float32


def test_struct_dtype_metadata():
    # vec3f: float32 base, 3 scalars, torch view trailing shape (3,)
    assert wd.warp_dtype_to_str(wp.vec3f) == "vec3f"
    assert wd.str_to_warp_dtype("vec3f") is wp.vec3f
    assert wd.scalar_base_str("vec3f") == "float32"
    assert wd.scalar_count("vec3f") == 3
    assert wd.trailing_shape("vec3f") == (3,)


def test_transform_and_matrix():
    assert wd.scalar_count("transformf") == 7
    assert wd.trailing_shape("transformf") == (7,)
    assert wd.scalar_count("mat33f") == 9
    assert wd.trailing_shape("mat33f") == (3, 3)


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        wd.str_to_warp_dtype("not_a_dtype")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_dtypes.py -q`
Expected: FAIL (`ModuleNotFoundError: leapp.backends.warp_dtypes`)

- [ ] **Step 3: Write minimal implementation**

```python
# leapp/backends/warp_dtypes.py
"""Single source of truth for warp dtype <-> string, including structured dtypes.

Each entry records: the warp dtype object, its scalar base (e.g. float32), the number of
scalars, and the trailing torch-view shape. Scalars have count 1 and trailing shape ().
The torch *view* of a structured-dtype port is always its scalar base with the trailing
shape appended to the per-element (batch) dims.
"""
import warp as wp

# name -> (warp dtype, scalar base name, scalar count, trailing torch-view shape)
_REGISTRY = {
    "float16": (wp.float16, "float16", 1, ()),
    "float32": (wp.float32, "float32", 1, ()),
    "float64": (wp.float64, "float64", 1, ()),
    "int8":  (wp.int8,  "int8",  1, ()),
    "int16": (wp.int16, "int16", 1, ()),
    "int32": (wp.int32, "int32", 1, ()),
    "int64": (wp.int64, "int64", 1, ()),
    "uint8": (wp.uint8, "uint8", 1, ()),
    "bool":  (wp.bool,  "bool",  1, ()),
    "vec2f": (wp.vec2f, "float32", 2, (2,)),
    "vec3f": (wp.vec3f, "float32", 3, (3,)),
    "vec4f": (wp.vec4f, "float32", 4, (4,)),
    "quatf": (wp.quatf, "float32", 4, (4,)),
    "transformf": (wp.transformf, "float32", 7, (7,)),
    "mat33f": (wp.mat33f, "float32", 9, (3, 3)),
    "mat44f": (wp.mat44f, "float32", 16, (4, 4)),
}
_WARP_TO_NAME = {entry[0]: name for name, entry in _REGISTRY.items()}


def warp_dtype_to_str(dtype) -> str:
    name = _WARP_TO_NAME.get(dtype)
    if name is None:
        raise KeyError(f"unsupported warp dtype {dtype!r}")
    return name


def str_to_warp_dtype(name: str):
    return _REGISTRY[name][0]


def scalar_base_str(name: str) -> str:
    return _REGISTRY[name][1]


def scalar_count(name: str) -> int:
    return _REGISTRY[name][2]


def trailing_shape(name: str) -> tuple:
    return _REGISTRY[name][3]


def is_structured(name: str) -> bool:
    return _REGISTRY[name][2] > 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_dtypes.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add leapp/backends/warp_dtypes.py tests/functional_tests/test_warp_dtypes.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): structured warp dtype registry"
```

### Task A2: `_WarpGraphCallable` honors structured dtypes (reshape + declared reinterpret)

**Files:**
- Modify: `leapp/backends/warp_export_backend.py:42-127` (dtype maps, `_WarpGraphCallable`) and `:208-265` (`save_warp_node` meta)
- Test: `tests/functional_tests/test_warp_region_node.py` (first test; file shared with Phase B)

- [ ] **Step 1: Write the failing test** — a one-launch warp graph whose *input* is `wp.vec3f` (torch view `float32 [N,3]`) round-trips through save→load→call.

```python
# tests/functional_tests/test_warp_region_node.py
import math
import pytest
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
if not torch.cuda.is_available():
    pytest.skip("warp .wrp requires CUDA", allow_module_level=True)

from leapp.backends.warp_export_backend import save_warp_node, WarpExportBackend

DEV = "cuda:0"


@wp.kernel
def _normalize3(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = wp.normalize(x[i])


def test_vec3f_input_roundtrips(tmp_path):
    wp.init()
    n = 5
    x = wp.zeros(n, dtype=wp.vec3f, device=DEV)
    out = wp.zeros(n, dtype=wp.vec3f, device=DEV)
    wp.load_module(device=DEV)
    with wp.ScopedCapture(device=DEV, force_module_load=True, apic=True) as cap:
        wp.launch(_normalize3, dim=n, inputs=[x], outputs=[out], device=DEV)

    node = save_warp_node(cap.graph, str(tmp_path), "normy",
                          inputs={"v": x}, outputs={"o": out})
    # the port records BOTH the torch view and the warp struct dtype
    assert node["inputs"][0]["dtype"] == "float32"
    assert node["inputs"][0]["shape"] == [n, 3]
    assert node["inputs"][0]["warp_dtype"] == "vec3f"

    backend = WarpExportBackend(node_stub := _NodeStub(), node["parameters"])
    backend.load(str(tmp_path / "normy.wrp"), node["parameters"]["sha256sum"])

    t = torch.randn(n, 3, dtype=torch.float32, device=DEV)
    got = backend.compiled_model(t)
    ref = torch.nn.functional.normalize(t, dim=1)
    assert got.shape == (n, 3)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


class _NodeStub:
    name = "normy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_region_node.py::test_vec3f_input_roundtrips -q`
Expected: FAIL — `save_warp_node` raises "unsupported warp array dtype" (vec3f not in old scalar map) or the port has no `warp_dtype` key.

- [ ] **Step 3: Write minimal implementation**

In `leapp/backends/warp_export_backend.py`, replace the scalar-only maps (lines 42-52) with imports from the registry and add struct-aware describe + runtime reshape:

```python
from leapp.backends import warp_dtypes as wd

# torch dtype <-> string for the SCALAR base view (validation stays strict on torch side)
_STR_TO_TORCH = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "uint8": torch.uint8, "bool": torch.bool,
}
```

In `_WarpGraphCallable.__init__`, also read `self.input_warp_dtypes = meta["input_warp_dtypes"]`, `self.output_warp_dtypes = meta["output_warp_dtypes"]`.

Replace the input-binding loop body (old lines 104-113) with:

```python
for name, base_dstr, wdstr, t in zip(
        self.input_names, self.input_dtypes, self.input_warp_dtypes, torch_inputs):
    expected = _STR_TO_TORCH.get(base_dstr)
    if expected is None:
        raise ValueError(f"warp node '{name}': unsupported declared dtype '{base_dstr}'")
    if t.dtype != expected:
        raise TypeError(
            f"warp node input '{name}' expected dtype {base_dstr}, got {t.dtype}.")
    count = wd.scalar_count(wdstr)
    # warp's from_torch infers the array length from a [-1, count] view for vector/quat/
    # transform dtypes, or [-1, r, c] for matrices; scalars stay flat.
    trailing = wd.trailing_shape(wdstr)
    tt = t.to(self.device).contiguous()
    tt = tt.reshape(-1) if count == 1 else tt.reshape((-1,) + trailing)
    self.graph.set_param(name, wp.from_torch(tt, dtype=wd.str_to_warp_dtype(wdstr)))
```

Replace the output read-back loop (old lines 117-121) with:

```python
for name, base_dstr, wdstr, shape in zip(
        self.output_names, self.output_dtypes, self.output_warp_dtypes, self.output_shapes):
    numel = int(math.prod(shape)) if shape else 1
    buf = wp.empty(numel // max(wd.scalar_count(wdstr), 1),
                   dtype=wd.str_to_warp_dtype(wdstr), device=self.device)
    self.graph.get_param(name, buf)
    outs.append(wp.to_torch(buf).reshape(tuple(shape)))
```

In `save_warp_node`, replace `_dstr`/`_desc` (lines 221-228) so each port records the torch *view* (scalar base + full shape incl. trailing) and the warp struct dtype:

```python
def _wstr(arr):
    return wd.warp_dtype_to_str(arr.dtype)

def _torch_view_shape(arr):
    return list(arr.shape) + list(wd.trailing_shape(wd.warp_dtype_to_str(arr.dtype)))

def _desc(name, arr):
    wdstr = _wstr(arr)
    return {"name": name, "dtype": wd.scalar_base_str(wdstr),
            "shape": _torch_view_shape(arr), "warp_dtype": wdstr, "type": "tensor"}
```

And extend the `meta` dict (lines 239-248) with the warp-dtype lists and torch-view output shapes:

```python
meta = {
    "inputs": list(inputs.keys()),
    "outputs": list(outputs.keys()),
    "input_dtypes": [wd.scalar_base_str(_wstr(a)) for a in inputs.values()],
    "output_dtypes": [wd.scalar_base_str(_wstr(a)) for a in outputs.values()],
    "input_warp_dtypes": [_wstr(a) for a in inputs.values()],
    "output_warp_dtypes": [_wstr(a) for a in outputs.values()],
    "output_shapes": [_torch_view_shape(a) for a in outputs.values()],
    "device_type": "cuda",
    "modules_dir": os.path.basename(modules_dir),
    "modules_sha256": modules_sha256,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_region_node.py::test_vec3f_input_roundtrips -q`
Expected: PASS

- [ ] **Step 5: Run the existing warp tests to confirm no regression**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_node.py tests/functional_tests/test_warp_autocapture.py -q`
Expected: same pass count as before (scalar `.wrp` paths still work — they now carry `warp_dtype == "float32"`).

- [ ] **Step 6: Commit**

```bash
git add leapp/backends/warp_export_backend.py tests/functional_tests/test_warp_region_node.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): structured warp dtypes across the .wrp bridge"
```

---

## Phase B — `WarpRegionNode` as a graph-registered node

A `WarpRegionNode` is a `LeappNode` (backend `warp`) created by the segmenter. It stores the recorded launches + device + bridged I/O and builds its `.wrp` at `compile_model()`. Because its `compiled_model` is the `_WarpGraphCallable` (`torch->torch`), the base `validate_compiled_model` validates it unchanged.

### Task B1: Shared capture core

**Files:**
- Create: `leapp/backends/warp_capture_core.py`
- Modify: `leapp/backends/warp_capture.py` (delegate to the core)
- Test: extend `tests/functional_tests/test_warp_autocapture.py` is already covering standalone; add `tests/functional_tests/test_warp_region_node.py::test_capture_core_replays`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/functional_tests/test_warp_region_node.py
from leapp.backends.warp_capture_core import replay_into_apic_capture, resolve_device


@wp.kernel
def _scale(x: wp.array(dtype=wp.float32), s: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * s


def test_capture_core_replays(tmp_path):
    wp.init()
    n = 4
    x = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=DEV)
    out = wp.zeros(n, dtype=wp.float32, device=DEV)
    records = [{"args": (_scale,), "kwargs": {"dim": n, "inputs": [x, 2.0],
                                              "outputs": [out], "device": DEV}}]
    assert resolve_device(wp, records, None) == DEV
    graph = replay_into_apic_capture(wp, records, device=DEV)
    node = save_warp_node(graph, str(tmp_path), "scaler", inputs={"x": x}, outputs={"o": out})
    assert node["parameters"]["backend"] == "warp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_region_node.py::test_capture_core_replays -q`
Expected: FAIL (`ModuleNotFoundError: leapp.backends.warp_capture_core`)

- [ ] **Step 3: Write minimal implementation**

```python
# leapp/backends/warp_capture_core.py
"""Reusable Warp capture helpers shared by the standalone warp_node() and WarpRegionNode.

A "record" is a dict {"args": tuple, "kwargs": dict} captured from a patched wp.launch call.
Replay runs ONLY the recorded launches inside one wp.ScopedCapture(apic=True); the arrays were
already allocated during the eager pass, so nothing allocates during capture (allocating inside
a CUDA-graph capture segfaults).
"""


def resolve_device(wp, records, explicit):
    if explicit:
        return str(explicit)
    for r in records:
        dev = r["kwargs"].get("device")
        if dev is not None:
            return str(dev)
    for r in records:
        for a in list(r["kwargs"].get("inputs") or []) + list(r["kwargs"].get("outputs") or []):
            if isinstance(a, wp.array):
                return str(a.device)
    return "cuda:0"


def replay_into_apic_capture(wp, records, device, orig_launch=None):
    launch = orig_launch or wp.launch
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as cap:
        for r in records:
            launch(*r["args"], **r["kwargs"])
    return cap.graph
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_region_node.py::test_capture_core_replays -q`
Expected: PASS

- [ ] **Step 5: Refactor `warp_capture.py` to use the core (no behavior change)**

In `leapp/backends/warp_capture.py`, replace the body of `WarpNodeCapture.__exit__` capture block (current lines 88-94) with:

```python
from leapp.backends.warp_capture_core import replay_into_apic_capture, resolve_device
device = resolve_device(wp, self._records, self.device)
graph = replay_into_apic_capture(wp, self._records, device, orig_launch=self._orig_launch)
```

and pass `graph` to `save_warp_node`. Keep `_detect_io` as the standalone fallback. Update `_records` construction in `patched_launch` to also store `"args"`/`"kwargs"` (already does).

- [ ] **Step 6: Run standalone warp tests to confirm no regression**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_autocapture.py -q`
Expected: same pass count as before.

- [ ] **Step 7: Commit**

```bash
git add leapp/backends/warp_capture_core.py leapp/backends/warp_capture.py tests/functional_tests/test_warp_region_node.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "refactor(warp): shared capture core for standalone + region node"
```

### Task B2: `WarpRegionNode`

**Files:**
- Create: `leapp/leapp_graph/warp_region_node.py`
- Test: `tests/functional_tests/test_warp_region_node.py::test_region_node_compiles_and_validates`

- [ ] **Step 1: Write the failing test** — build a node directly (no segmenter yet), compile, validate.

```python
# add to tests/functional_tests/test_warp_region_node.py
from leapp.leapp_graph.warp_region_node import WarpRegionNode


def test_region_node_compiles_and_validates(tmp_path):
    wp.init()
    n = 4
    x = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=DEV)
    out = wp.zeros(n, dtype=wp.float32, device=DEV)
    records = [{"args": (_scale,), "kwargs": {"dim": n, "inputs": [x, 2.0],
                                              "outputs": [out], "device": DEV}}]
    node = WarpRegionNode("seg.02_warp", device=DEV)
    # source torch tensors carry the leapp_tag that forms the incoming edge
    src = torch.ones(n, dtype=torch.float32, device=DEV)
    src.leapp_tag = "seg.01_torch/h/"
    node.set_io(records,
                inputs={"in0": (x, src)},
                outputs={"out0": out})
    node.node_index = 1
    node.compile_model()
    node.save_model(str(tmp_path))
    # incoming edge tag preserved on the input description
    assert node.inputs[0].tag == "seg.01_torch/h/"
    # compiled_model is the torch->torch callable; base validation works
    node._max_cached_io = 0
    got = node.compiled_model(torch.tensor([1., 2., 3., 4.], device=DEV))
    assert torch.allclose(got, torch.tensor([2., 4., 6., 8.], device=DEV))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_region_node.py::test_region_node_compiles_and_validates -q`
Expected: FAIL (`ModuleNotFoundError: leapp.leapp_graph.warp_region_node`)

- [ ] **Step 3: Write minimal implementation**

```python
# leapp/leapp_graph/warp_region_node.py
"""A graph-registered warp node: a contiguous warp segment captured as one native .wrp.

Built by the bridge segmenter (leapp.warp_bridge). Inputs/outputs are declared by the bridge
crossings (wp.from_torch / wp.to_torch), not by the ptr-order heuristic. Each input carries the
leapp_tag of the torch tensor that produced it, so the cross-kind edge forms by tag-matching
exactly like a torch->torch edge.
"""
import os
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.backends.warp_capture_core import replay_into_apic_capture, resolve_device
from leapp.backends.warp_export_backend import save_warp_node


class WarpRegionNode(LeappNode):
    def __init__(self, name, device=None, dry_run=False):
        super().__init__(name, dry_run=dry_run)
        self.device = str(device) if device else None
        self._records = []
        self._wp_inputs = {}     # port name -> wp.array
        self._wp_outputs = {}    # port name -> wp.array
        self._wrp_path = None

    def set_io(self, records, inputs, outputs):
        """inputs: name -> (wp.array, source_torch_tensor_carrying_leapp_tag).
        outputs: name -> wp.array."""
        import warp as wp
        self._records = records
        if self.device is None:
            self.device = resolve_device(wp, records, None)
        self._wp_inputs = {n: a for n, (a, _src) in inputs.items()}
        self._wp_outputs = dict(outputs)
        for n, (arr, src) in inputs.items():
            # add_input pulls the tag off `src` (it has .leapp_tag); use src for shape/dtype view
            self.add_input(n, n, src)
        # outputs are tagged with this node's name so downstream torch picks up the edge
        for n, arr in outputs.items():
            from leapp.backends import warp_dtypes as wd
            import torch
            wdstr = wd.warp_dtype_to_str(arr.dtype)
            view_shape = tuple(arr.shape) + wd.trailing_shape(wdstr)
            placeholder = torch.zeros(view_shape,
                                      dtype=getattr(torch, wd.scalar_base_str(wdstr)),
                                      device=self.device)
            self.tag_data(placeholder, n)         # -> placeholder.leapp_tag = "<name>/<n>/"
            self.add_output(n, n, placeholder)

    def compile_model(self):
        import warp as wp
        graph = replay_into_apic_capture(wp, self._records, self.device)
        node_dict = save_warp_node(graph, self._save_dir, self.name,
                                   inputs=self._wp_inputs, outputs=self._wp_outputs)
        params = node_dict["parameters"]
        # the .wrp now lives at <save_dir>/<name>.wrp; wire a warp backend pointed at it
        self.setup_backend("warp", {**params,
                                    "model_path": os.path.join(self._save_dir, self.name + ".wrp")})
        self.export_backend.load(self.export_backend.backend_params["model_path"],
                                 params["sha256sum"])
        self._wrp_path = self.export_backend.backend_params["model_path"]
        self._model_captured = True

    def save_model(self, save_path: str):
        # compile_model already wrote the .wrp into self._save_dir == save_path; relocate via backend
        return self.export_backend.save(save_path)

    def set_save_dir(self, save_dir):
        self._save_dir = save_dir
```

Note for the executor: `compile_model()` needs `self._save_dir`. The segmenter calls `node.set_save_dir(manager.get_save_path())` before compile. In this unit test, set it directly: add `node.set_save_dir(str(tmp_path))` before `node.compile_model()`.

- [ ] **Step 4: Update the test to set the save dir, then run**

Add `node.set_save_dir(str(tmp_path))` immediately before `node.compile_model()`.
Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_region_node.py::test_region_node_compiles_and_validates -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add leapp/leapp_graph/warp_region_node.py tests/functional_tests/test_warp_region_node.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): WarpRegionNode (graph-registered warp segment)"
```

---

## Phase C — Bridge interception + segmenter state machine

### Task C1: `RegionSegmenter` — torch→warp finalize/open

**Files:**
- Create: `leapp/warp_bridge.py`
- Modify: `leapp/export_manager.py` (add segment-resolution map + hooks)
- Test: `tests/functional_tests/test_warp_bridge.py::test_from_torch_splits_segment`

- [ ] **Step 1: Write the failing test** — drive the segmenter directly with a fake manager to assert the split bookkeeping (names, edge tag), no real warp needed.

```python
# tests/functional_tests/test_warp_bridge.py
import pytest
torch = pytest.importorskip("torch")
from leapp.warp_bridge import RegionSegmenter


class FakeTorchNode:
    def __init__(self, name):
        self.name = name
        self.compiled = None
    def compile_trace(self, tensors, backend=None, **kw):
        self.compiled = (dict(tensors), backend)
    # tag the finalized output like the real create_output path does
        for n, t in tensors.items():
            t.leapp_tag = f"{self.name}/{n}/"


class FakeManager:
    def __init__(self):
        self.nodes = {}
        self.renamed = []
        self.warp_nodes = []
        self.indices = []
    def _rename_node(self, old, new):
        self.nodes[new] = self.nodes.pop(old)
        self.nodes[new].name = new
        self.renamed.append((old, new))
    def _assign_index(self, node):
        self.indices.append(node.name)
    def _default_torch_backend(self):
        return "onnx-torchscript"


def test_from_torch_splits_segment():
    mgr = FakeManager()
    seg0 = FakeTorchNode("policy")
    mgr.nodes["policy"] = seg0
    seg = RegionSegmenter(mgr, region="policy", first_node=seg0)

    h = torch.ones(3)              # the tensor crossing into warp
    seg.on_from_torch_input(h, out_name="out0", warp_dtype="vec3f")

    # first torch segment renamed and finalized with h as its output
    assert ("policy", "policy.01_torch") in mgr.renamed
    assert seg0.compiled[0]["out0"] is h
    assert seg0.compiled[1] == "onnx-torchscript"
    # edge tag now lives on h and will be inherited by the warp input
    assert h.leapp_tag == "policy.01_torch/out0/"
    # a warp segment is now open
    assert seg.open_kind == "warp"
    assert seg.open_node.name == "policy.02_warp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_bridge.py::test_from_torch_splits_segment -q`
Expected: FAIL (`ModuleNotFoundError: leapp.warp_bridge`)

- [ ] **Step 3: Write minimal implementation**

```python
# leapp/warp_bridge.py
"""Bridge interception + the per-region linear segmenter.

v1 bridge = wp.from_torch / wp.to_torch ONLY. install() patches those symbols on the warp
module; the segmenter turns each crossing into a node boundary. Linear chains only
(ADR-0002): a forked tensor across a bridge fails loudly.
"""
from leapp.leapp_graph.warp_region_node import WarpRegionNode


def _seg_name(region, idx, kind):
    return f"{region}.{idx:02d}_{kind}"


class RegionSegmenter:
    """One per active marked region. Tracks the currently-open segment node and splits at
    bridges. The first segment starts as the bare `region` node and is renamed to
    `<region>.01_torch` the first time a split actually happens."""

    def __init__(self, manager, region, first_node):
        self.mgr = manager
        self.region = region
        self.open_node = first_node
        self.open_kind = "torch"
        self._seg_idx = 1
        self._split_happened = False
        self._bridge_counter = 0

    def _ensure_first_renamed(self):
        if not self._split_happened:
            new = _seg_name(self.region, 1, "torch")
            self.mgr._rename_node(self.open_node.name, new)
            self._split_happened = True
            self._seg_idx = 1

    def on_from_torch_input(self, torch_tensor, out_name, warp_dtype):
        if self.open_kind != "torch":
            raise RuntimeError(
                f"warp_bridge: wp.from_torch reached while a warp segment is open in region "
                f"'{self.region}'. v1 supports only linear torch<->warp chains (ADR-0002); "
                "express this as explicit manual nodes.")
        self._ensure_first_renamed()
        torch_node = self.open_node
        # finalize the torch segment with this tensor as a named output (tags it)
        torch_node.compile_trace({out_name: torch_tensor},
                                 backend=self.mgr._default_torch_backend())
        self.mgr._assign_index(torch_node)
        # open a warp segment
        self._seg_idx += 1
        warp_node = WarpRegionNode(_seg_name(self.region, self._seg_idx, "warp"))
        self.mgr.nodes[warp_node.name] = warp_node
        self.open_node = warp_node
        self.open_kind = "warp"
        # remember the producing tensor so its leapp_tag forms the cross-kind edge
        self._pending_warp_inputs = getattr(self, "_pending_warp_inputs", {})
        self._pending_warp_inputs[out_name] = torch_tensor
        self._pending_warp_dtypes = getattr(self, "_pending_warp_dtypes", {})
        self._pending_warp_dtypes[out_name] = warp_dtype
        return torch_tensor
```

Add the three hooks to `ExportManager` (`leapp/export_manager.py`), near the other node helpers:

```python
def _rename_node(self, old, new):
    if new in self.nodes:
        raise Exception(f"cannot rename node '{old}' to existing '{new}'")
    node = self.nodes.pop(old)
    node.name = new
    self.nodes[new] = node

def _assign_index(self, node):
    self._assign_completion_index(node)

def _default_torch_backend(self):
    return "onnx-torchscript"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_bridge.py::test_from_torch_splits_segment -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add leapp/warp_bridge.py leapp/export_manager.py tests/functional_tests/test_warp_bridge.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): RegionSegmenter torch->warp split"
```

### Task C2: `RegionSegmenter` — warp→torch finalize/open + continuation

**Files:**
- Modify: `leapp/warp_bridge.py`
- Test: `tests/functional_tests/test_warp_bridge.py::test_to_torch_opens_continuation`

- [ ] **Step 1: Write the failing test** — after a warp segment, `wp.to_torch` records warp outputs, opens a continuation torch node, and tags the returned tensor with the warp output tag.

```python
# add to tests/functional_tests/test_warp_bridge.py
class FakeWarpArray:
    def __init__(self, ptr): self.ptr = ptr

def test_to_torch_opens_continuation(monkeypatch):
    mgr = FakeManager()
    seg0 = FakeTorchNode("policy")
    mgr.nodes["policy"] = seg0
    seg = RegionSegmenter(mgr, region="policy", first_node=seg0)
    h = torch.ones(3)
    seg.on_from_torch_input(h, out_name="out0", warp_dtype="vec3f")

    # the continuation torch node must be a fresh node the segmenter creates; stub the factory
    created = {}
    def fake_open_torch(region, idx):
        node = FakeTorchNode(_n := f"{region}.{idx:02d}_torch")
        created["node"] = node
        return node
    monkeypatch.setattr(seg, "_make_torch_node", fake_open_torch)

    out_arr = FakeWarpArray(ptr=123)
    d = torch.zeros(3)            # the to_torch result
    seg.on_to_torch_output(out_arr, result_tensor=d)

    # warp node finalized with one output, indexed
    assert seg._finalized_warp is not None
    assert "policy.02_warp" in mgr.indices
    # the returned tensor carries the warp node's output tag (forms warp->torch edge)
    assert d.leapp_tag == "policy.02_warp/out0/"
    # a continuation torch segment is open
    assert seg.open_kind == "torch"
    assert seg.open_node.name == "policy.03_torch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_bridge.py::test_to_torch_opens_continuation -q`
Expected: FAIL (`AttributeError: 'RegionSegmenter' object has no attribute 'on_to_torch_output'`)

- [ ] **Step 3: Write minimal implementation** — add to `RegionSegmenter`:

```python
    def _make_torch_node(self, region, idx):
        # real impl: create a TracedTensorNode via the manager so input_tensors/output_tensors
        # route correctly. Overridden in unit tests.
        from leapp.leapp_graph.traced_node import TracedTensorNode
        node = TracedTensorNode(_seg_name(region, idx, "torch"))
        node._max_cached_io = 0
        self.mgr.nodes[node.name] = node
        return node

    def on_to_torch_output(self, warp_array, result_tensor):
        if self.open_kind != "warp":
            raise RuntimeError(
                f"warp_bridge: wp.to_torch reached with no open warp segment in region "
                f"'{self.region}'.")
        warp_node = self.open_node
        out_name = f"out{len(getattr(warp_node, '_pending_outputs', {}))}"
        warp_node._pending_outputs = getattr(warp_node, "_pending_outputs", {})
        warp_node._pending_outputs[out_name] = warp_array
        # tag the torch result so the downstream torch node inherits the edge
        result_tensor.leapp_tag = f"{warp_node.name}/{out_name}/"
        # finalize the warp node (records assembled by the patched wp.launch live on it)
        self._finalize_warp_node(warp_node)
        self.mgr._assign_index(warp_node)
        self._finalized_warp = warp_node
        # open the continuation torch segment
        self._seg_idx += 1
        cont = self._make_torch_node(self.region, self._seg_idx)
        self.open_node = cont
        self.open_kind = "torch"
        return result_tensor

    def _finalize_warp_node(self, warp_node):
        # assemble bridged I/O onto the WarpRegionNode and let it build its .wrp lazily at
        # compile_model(); in unit tests this is monkeypatched away.
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_bridge.py::test_to_torch_opens_continuation -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add leapp/warp_bridge.py tests/functional_tests/test_warp_bridge.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): RegionSegmenter warp->torch continuation"
```

### Task C3: `install()/uninstall()` patch the warp bridge symbols + record launches

**Files:**
- Modify: `leapp/warp_bridge.py`
- Test: `tests/functional_tests/test_warp_bridge.py::test_install_patches_and_records`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/functional_tests/test_warp_bridge.py
def test_install_patches_and_records(monkeypatch):
    import types
    calls = {"from_torch": 0, "launch": 0}
    fake_wp = types.SimpleNamespace(
        from_torch=lambda t, dtype=None: ("warp_arr", t, dtype),
        to_torch=lambda a: ("torch", a),
        launch=lambda *a, **k: calls.__setitem__("launch", calls["launch"] + 1),
        array=object,
    )
    from leapp import warp_bridge
    monkeypatch.setattr(warp_bridge, "_import_warp", lambda: fake_wp)

    state = warp_bridge.install()
    assert fake_wp.from_torch is not (lambda t, dtype=None: None)  # patched
    # original callables restored on uninstall
    warp_bridge.uninstall(state)
    assert fake_wp.launch.__name__ != "patched_launch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_bridge.py::test_install_patches_and_records -q`
Expected: FAIL (`AttributeError: module 'leapp.warp_bridge' has no attribute 'install'`)

- [ ] **Step 3: Write minimal implementation** — add module-level install/uninstall and a global active-segmenter registry the patches consult:

```python
# add to leapp/warp_bridge.py
_ACTIVE = {"segmenter": None, "orig_launch": None, "records_sink": None}


def _import_warp():
    import warp as wp
    return wp


def set_active_segmenter(segmenter):
    _ACTIVE["segmenter"] = segmenter


def install():
    wp = _import_warp()
    orig = {"from_torch": wp.from_torch, "to_torch": wp.to_torch, "launch": wp.launch}
    _ACTIVE["orig_launch"] = orig["launch"]

    def patched_from_torch(t, dtype=None, *a, **k):
        seg = _ACTIVE["segmenter"]
        arr = orig["from_torch"](t, dtype=dtype, *a, **k)
        if seg is not None and getattr(t, "leapp_tag", None) is not None:
            from leapp.backends import warp_dtypes as wd
            wdstr = wd.warp_dtype_to_str(dtype) if dtype is not None else "float32"
            name = f"out{seg._bridge_counter}"
            seg._bridge_counter += 1
            seg.on_from_torch_input(t, out_name=name, warp_dtype=wdstr)
            seg.bind_warp_input(name, arr, t)
        return arr

    def patched_to_torch(a, *args, **k):
        seg = _ACTIVE["segmenter"]
        out = orig["to_torch"](a, *args, **k)
        if seg is not None and seg.open_kind == "warp":
            seg.on_to_torch_output(a, result_tensor=out)
        return out

    def patched_launch(*a, **k):
        seg = _ACTIVE["segmenter"]
        if seg is not None and seg.open_kind == "warp":
            seg.record_launch(a, k)
        return orig["launch"](*a, **k)
    patched_launch.__name__ = "patched_launch"

    wp.from_torch, wp.to_torch, wp.launch = patched_from_torch, patched_to_torch, patched_launch
    return {"wp": wp, "orig": orig}


def uninstall(state):
    wp, orig = state["wp"], state["orig"]
    wp.from_torch, wp.to_torch, wp.launch = orig["from_torch"], orig["to_torch"], orig["launch"]
    _ACTIVE["segmenter"] = None
    _ACTIVE["orig_launch"] = None
```

Add the small helpers to `RegionSegmenter`:

```python
    def record_launch(self, args, kwargs):
        self.open_node._records.append({"args": args, "kwargs": kwargs})

    def bind_warp_input(self, name, warp_array, src_tensor):
        self.open_node._wp_inputs[name] = warp_array
        self._pending_warp_inputs[name] = src_tensor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_bridge.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add leapp/warp_bridge.py tests/functional_tests/test_warp_bridge.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): install/uninstall bridge patches + launch recording"
```

---

## Phase D — Wire the segmenter into the export lifecycle

### Task D1: `start()/stop()` install/uninstall the bridge; `input_tensors` opens a region

**Files:**
- Modify: `leapp/leapp.py:36-105` (`start`/`stop`)
- Modify: `leapp/export_manager.py` (region map + segment resolution in `input_tensors`/`output_tensors`)
- Test: `tests/functional_tests/test_mixed_autosplit.py::test_start_installs_bridge_when_warp_available`

- [ ] **Step 1: Write the failing test**

```python
# tests/functional_tests/test_mixed_autosplit.py
import pytest
torch = pytest.importorskip("torch")
import leapp


def test_start_installs_bridge_when_warp_available(tmp_path):
    pytest.importorskip("warp")
    leapp.start("g", save_path=str(tmp_path))
    import warp as wp
    assert wp.launch.__name__ == "patched_launch"
    leapp.stop()
    assert wp.launch.__name__ != "patched_launch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py::test_start_installs_bridge_when_warp_available -q`
Expected: FAIL (`wp.launch.__name__` is not `patched_launch`)

- [ ] **Step 3: Write minimal implementation** — in `leapp/leapp.py`, at the end of `start()` (after `ExportManager.set_interpret_graph(True)`):

```python
    # Install the warp bridge if warp is importable (optional dependency).
    manager._warp_bridge_state = None
    try:
        import warp  # noqa: F401
        from leapp import warp_bridge
        manager._warp_bridge_state = warp_bridge.install()
    except ImportError:
        pass
```

and in `stop()`, before/after disabling interpret_graph:

```python
    state = getattr(manager, "_warp_bridge_state", None)
    if state is not None:
        from leapp import warp_bridge
        warp_bridge.uninstall(state)
        manager._warp_bridge_state = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py::test_start_installs_bridge_when_warp_available -q`
Expected: PASS

- [ ] **Step 5: Run the full non-warp suite to confirm torch-only graphs are unaffected**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests -q -k "not live"`
Expected: previously-passing tests still pass (the bridge is installed but dormant — `set_active_segmenter` is never called for torch-only graphs, so patches no-op).

- [ ] **Step 6: Commit**

```bash
git add leapp/leapp.py tests/functional_tests/test_mixed_autosplit.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): start/stop install the bridge when warp is present"
```

### Task D2: `input_tensors` registers a segmenter; `output_tensors` finalizes the open segment

**Files:**
- Modify: `leapp/export_manager.py:243-365` (`input_tensors`, `output_tensors`)
- Test: `tests/functional_tests/test_mixed_autosplit.py::test_region_resolves_to_open_segment`

- [ ] **Step 1: Write the failing test** — torch-only region must behave exactly as today (single node keeps bare name, no segmenter side effects); then a region whose name is reused resolves to the open segment.

```python
# add to tests/functional_tests/test_mixed_autosplit.py
from leapp import annotate


def test_torch_only_region_keeps_bare_name(tmp_path):
    leapp.start("g", save_path=str(tmp_path))
    x = torch.ones(4)
    xt = annotate.input_tensors("policy", {"x": x})
    y = xt * 2.0
    annotate.output_tensors("policy", {"y": y}, export_with="onnx-torchscript")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)
    # single-kind region => bare name, no .NN_ suffix
    from leapp.export_manager import ExportManager
    assert "policy" in ExportManager().get_nodes()
    assert not any(n.startswith("policy.") for n in ExportManager().get_nodes())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py::test_torch_only_region_keeps_bare_name -q`
Expected: it should PASS already if `input_tensors`/`output_tensors` are untouched — confirm it does, which guards against regressions before we add resolution. If it FAILS, fix the regression introduced by D1 first.

- [ ] **Step 3: Add region/segment resolution (does not change torch-only behavior)**

In `ExportManager.__init__`, add `self._region_segmenters = {}` and `self._region_open_node = {}`.

In `input_tensors`, immediately after the node is created/fetched (after the `if node_name in self.nodes: ... else: ... _setup_new_node(...)` block, ~line 261), register a segmenter the first time a region is opened and only when the bridge is active:

```python
        # Register a segmenter for this region so a warp bridge can auto-split it.
        if node_name not in self._region_segmenters and getattr(self, "_warp_bridge_state", None):
            from leapp.warp_bridge import RegionSegmenter, set_active_segmenter
            seg = RegionSegmenter(self, region=node_name, first_node=self.nodes[node_name])
            self._region_segmenters[node_name] = seg
            set_active_segmenter(seg)
```

In both `input_tensors` and `output_tensors`, resolve the *region* base name to the currently-open segment node name before touching `self.nodes`:

```python
    def _resolve_open_node_name(self, node_name):
        seg = self._region_segmenters.get(node_name)
        return seg.open_node.name if seg is not None else node_name
```

Use it in `output_tensors`: replace `if node_name in self.nodes:` lookup target with `resolved = self._resolve_open_node_name(node_name)` and operate on `self.nodes[resolved]`. (For torch-only regions `resolved == node_name`, so behavior is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add leapp/export_manager.py tests/functional_tests/test_mixed_autosplit.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): region->open-segment resolution in input/output_tensors"
```

### Task D3: empty-continuation pruning + clear `set_active_segmenter` on output_tensors

**Files:**
- Modify: `leapp/export_manager.py` (`output_tensors`), `leapp/warp_bridge.py`
- Test: `tests/functional_tests/test_mixed_autosplit.py::test_empty_continuation_pruned`

- [ ] **Step 1: Write the failing test** — a region that ends right after `wp.to_torch` (no torch ops before the marked output) must NOT produce a trailing empty torch node.

```python
# add to tests/functional_tests/test_mixed_autosplit.py
def test_empty_continuation_pruned():
    # Pure-bookkeeping test against the segmenter (no GPU): simulate finalize with an
    # untouched continuation node, then ask output_tensors to finalize it.
    from leapp.warp_bridge import RegionSegmenter
    class FakeNode:
        def __init__(self, name): self.name = name; self.touched = False; self._records=[]; self._wp_inputs={}
    class Mgr:
        def __init__(self): self.nodes={}; 
        def _rename_node(self,o,n): self.nodes[n]=self.nodes.pop(o); self.nodes[n].name=n
        def _assign_index(self,node): pass
        def _default_torch_backend(self): return "onnx-torchscript"
    mgr = Mgr(); first = FakeNode("r"); mgr.nodes["r"]=first
    seg = RegionSegmenter(mgr, "r", first)
    # mark continuation empty and ask: should it be pruned?
    cont = FakeNode("r.03_torch"); cont.touched = False
    seg.open_node = cont; seg.open_kind = "torch"
    assert seg.is_open_segment_empty() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py::test_empty_continuation_pruned -q`
Expected: FAIL (`AttributeError: ... 'is_open_segment_empty'`)

- [ ] **Step 3: Write minimal implementation** — in `RegionSegmenter`:

```python
    def is_open_segment_empty(self):
        """True if the open torch continuation has traced no ops (its graph has only the
        placeholder input). For a real TracedTensorNode, an empty graph has no call nodes."""
        node = self.open_node
        graph = getattr(node, "graph", None)
        if graph is None:
            return not getattr(node, "touched", True)
        return not any(n.op == "call_function" or n.op == "call_method" for n in graph.nodes)

    def drop_open_segment(self):
        self.mgr.nodes.pop(self.open_node.name, None)
```

In `ExportManager.output_tensors`, before finalizing, if the region's open segment is an empty continuation drop it and finalize the *previous* (warp) node's outputs as the region outputs. Concretely, at the start of `output_tensors`, after resolving:

```python
        seg = self._region_segmenters.get(node_name)
        if seg is not None and seg.open_kind == "torch" and seg.is_open_segment_empty():
            seg.drop_open_segment()
            # the region's outputs are already the finalized warp node's outputs; clear segmenter
            from leapp.warp_bridge import set_active_segmenter
            set_active_segmenter(None)
            return self._passthrough_dict_values(
                self._normalize_named_tensor_payload("output_tensors", node_name, tensors)[0])
```

Always clear the active segmenter at the end of `output_tensors` for a region (so the next region starts clean):

```python
        if node_name in self._region_segmenters:
            from leapp.warp_bridge import set_active_segmenter
            set_active_segmenter(None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add leapp/export_manager.py leapp/warp_bridge.py tests/functional_tests/test_mixed_autosplit.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): prune empty continuation, reset segmenter per region"
```

---

## Phase E — End-to-end: projected-gravity mixed graph

### Task E1: end-to-end torch→warp→torch round-trip via InferenceManager

**Files:**
- Create: `tests/functional_tests/test_projected_gravity_mixed.py`
- Modify: `leapp/warp_bridge.py` `_finalize_warp_node` to populate the `WarpRegionNode` from pending I/O + set its save dir at compile (wire to `manager.get_save_path()`)

- [ ] **Step 1: Write the failing test** — mirrors `examples/warp_mixed_graph_prototype.py` but with **one** `leapp.start()/stop()` trace and **no manual YAML**; warp input is `wp.vec3f`.

```python
# tests/functional_tests/test_projected_gravity_mixed.py
import pytest
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)
import leapp
from leapp import annotate, InferenceManager

DEV = "cuda:0"
N = 6


@wp.kernel
def _norm_vec3(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = wp.normalize(x[i])


def test_mixed_autosplit_roundtrips(tmp_path):
    wp.init()
    leapp.start("pg", save_path=str(tmp_path))
    g = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    gt = annotate.input_tensors("obs", {"g": g})
    scaled = gt * 2.0                          # torch segment
    a = wp.from_torch(scaled.contiguous().reshape(-1, 3), dtype=wp.vec3f)  # bridge
    out = wp.zeros(N, dtype=wp.vec3f, device=DEV)
    wp.launch(_norm_vec3, dim=N, inputs=[a], outputs=[out], device=DEV)    # warp segment
    d = wp.to_torch(out).reshape(N, 3)         # bridge back
    annotate.output_tensors("obs", {"pg": d})  # region output IS the warp output
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    yaml_path = str(tmp_path / "pg" / "pg.yaml")
    im = InferenceManager(yaml_path)
    g_in = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    res = im({"obs.01_torch/g": g_in})
    out_key = [k for k in res if k.endswith("/pg")][0]
    ref = torch.nn.functional.normalize(g_in * 2.0, dim=1)
    assert torch.allclose(res[out_key], ref, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_projected_gravity_mixed.py -q`
Expected: FAIL — `_finalize_warp_node` is a stub, so the warp node has no `.wrp` / `compile_model` errors.

- [ ] **Step 3: Implement `_finalize_warp_node` and save-dir wiring**

In `RegionSegmenter._finalize_warp_node`, assemble bridged I/O onto the node and let `compile_model` build the `.wrp`:

```python
    def _finalize_warp_node(self, warp_node):
        inputs = {n: (warp_node._wp_inputs[n], self._pending_warp_inputs[n])
                  for n in warp_node._wp_inputs}
        outputs = dict(warp_node._pending_outputs)
        warp_node.set_save_dir(self.mgr.get_save_path())
        warp_node.set_io(warp_node._records, inputs=inputs, outputs=outputs)
```

`compile_models()` (base manager) will later call `warp_node.compile_model()`, which writes the `.wrp` into the save dir and wires the backend. `save_models()` then relocates via `WarpExportBackend.save` (idempotent when already in place).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_projected_gravity_mixed.py -q`
Expected: PASS (`max_abs_err` within tolerance)

- [ ] **Step 5: Inspect the emitted YAML once (manual sanity)**

Run: `sed -n '1,80p' "$(ls -d /tmp/pytest*/**/pg/pg.yaml 2>/dev/null | head -1)"` — or print from the test. Confirm: two nodes (`obs.01_torch` backend onnx/jit, `obs.02_warp` backend warp), a `data_flow` edge `obs.01_torch/<out> -> obs.02_warp/<in>`, region output `obs.02_warp/pg`, and the warp input port shows `dtype: float32, shape: [N, 3], warp_dtype: vec3f`.

- [ ] **Step 6: Commit**

```bash
git add leapp/warp_bridge.py tests/functional_tests/test_projected_gravity_mixed.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "feat(warp): end-to-end auto-split mixed graph (projected-gravity)"
```

### Task E2: fail-loud guards (linear-chain + non-traced constant)

**Files:**
- Modify: `leapp/warp_bridge.py`
- Test: `tests/functional_tests/test_mixed_autosplit.py::test_fork_fails_loud`, `::test_nontraced_from_torch_is_constant`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/functional_tests/test_mixed_autosplit.py
def test_fork_fails_loud():
    from leapp.warp_bridge import RegionSegmenter
    class FakeNode:
        def __init__(s,n): s.name=n
        def compile_trace(s,t,backend=None,**k):
            for nm,tt in t.items(): tt.leapp_tag=f"{s.name}/{nm}/"
    class Mgr:
        def __init__(s): s.nodes={}
        def _rename_node(s,o,n): s.nodes[n]=s.nodes.pop(o); s.nodes[n].name=n
        def _assign_index(s,node): pass
        def _default_torch_backend(s): return "onnx-torchscript"
    mgr=Mgr(); f=FakeNode("r"); mgr.nodes["r"]=f
    seg=RegionSegmenter(mgr,"r",f)
    import torch
    h=torch.ones(3); seg.on_from_torch_input(h,"out0","vec3f")
    # a second from_torch while a warp segment is open => linear-chain violation
    with pytest.raises(RuntimeError, match="linear"):
        seg.on_from_torch_input(torch.ones(3), "out1", "vec3f")


def test_nontraced_from_torch_is_constant(tmp_path):
    pytest.importorskip("warp")
    leapp.start("c", save_path=str(tmp_path))
    import warp as wp
    const = torch.ones(3)              # NOT a marked/traced tensor -> no leapp_tag
    arr = wp.from_torch(const.reshape(-1, 3), dtype=wp.vec3f)
    # no segmenter is active (no region opened), so this is a plain warp array, no split
    from leapp.export_manager import ExportManager
    assert ExportManager().get_nodes() == {}
    leapp.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py -k "fork or nontraced" -q`
Expected: `test_fork_fails_loud` already passes (guard added in C1); `test_nontraced_from_torch_is_constant` passes if the patched `from_torch` only splits when `t.leapp_tag` exists AND a segmenter is active. If either fails, fix per Step 3.

- [ ] **Step 3: Confirm/strengthen guards**

The C1 guard already raises on a second `from_torch` while warp is open (linear-chain). The C3 patched `from_torch` already checks `getattr(t, "leapp_tag", None) is not None` and `seg is not None`. Add an explicit comment in `warp_bridge.install().patched_from_torch` documenting that an untagged tensor is treated as a baked constant (no split), per the design notes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_mixed_autosplit.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add leapp/warp_bridge.py tests/functional_tests/test_mixed_autosplit.py
git -c commit.gpgsign=false commit --no-gpg-sign -m "test(warp): linear-chain + non-traced-constant guards"
```

### Task E3: full suite + docs

**Files:**
- Modify: `docs/design/warp-mixed-graph.md` (mark v1 status), `examples/` (add `examples/warp_autosplit_mixed.py` mirroring the E1 test as a runnable example)

- [ ] **Step 1: Run the whole warp + core suite**

Run: `PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_warp_dtypes.py tests/functional_tests/test_warp_region_node.py tests/functional_tests/test_warp_bridge.py tests/functional_tests/test_mixed_autosplit.py tests/functional_tests/test_projected_gravity_mixed.py tests/functional_tests/test_warp_node.py tests/functional_tests/test_warp_autocapture.py -q`
Expected: all pass (live-Triton tier excluded).

- [ ] **Step 2: Add the runnable example** (copy the E1 test body into a `main()` that prints `max_abs_err` and a PASS/FAIL line, matching the style of `examples/warp_mixed_graph_prototype.py`).

- [ ] **Step 3: Update `docs/design/warp-mixed-graph.md`** — add a "## Status" line: "v1 implemented: structured dtypes, auto-split linear chains, projected-gravity end-to-end via InferenceManager. Triton-generator support for auto-split graphs tracked separately."

- [ ] **Step 4: Commit**

```bash
git add examples/warp_autosplit_mixed.py docs/design/warp-mixed-graph.md
git -c commit.gpgsign=false commit --no-gpg-sign -m "docs(warp): runnable auto-split example + v1 status"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** structured dtypes (A), graph-registered warp node (B), bridge + segmenter (C), lifecycle wiring (D), end-to-end + guards + docs (E). Stateless-only and linear-chain are enforced by guards (C1, E2) and documented (ADR-0002, design notes).
- **Known integration risks to watch (TDD will surface these):**
  1. Calling `TracedTensorNode.compile_trace` from inside the patched `wp.from_torch` runs while `ExportManager._interpret_graph` is True — confirm `validate_status`/`TracingLock` don't reject it (they guard against *nested* traced functions, not annotation-time finalization). If they do, finalize via the same path `output_tensors` uses.
  2. The continuation `wp.to_torch` must return a tensor the user can keep using in torch ops *and* that re-enters tracing. In E1 the region ends at the warp output (empty continuation, pruned), so tracing-through-continuation is exercised separately — add a follow-up test where real torch ops follow `to_torch` before lifting beyond the projected-gravity case.
  3. `wp.from_torch` arg shape for struct dtypes: verify warp expects `[-1, count]` (vectors) / `[-1, r, c]` (matrices); adjust `_WarpGraphCallable` reshape if warp 1.14 wants a flat buffer + explicit length.
- **Type consistency:** segment names use `_seg_name(region, idx, kind)` everywhere; warp ports carry both `dtype` (scalar base) and `warp_dtype` (struct) consistently across `save_warp_node`, `_WarpGraphCallable`, and `WarpRegionNode.set_io`.
