# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Variable-axis contracts, inference, and runtime validation."""

from dataclasses import dataclass
import copy
import json
import itertools
import operator
import re
from typing import Literal, TypedDict

import sympy
import torch


class _DimensionBounds(TypedDict):
    min: int
    max: int


class DimensionRange(_DimensionBounds, total=False):
    multiple_of: int


class AffineExpression(TypedDict):
    terms: dict[str, int]
    offset: int


class FormulaOperation(TypedDict):
    op: Literal["add", "mul", "floordiv", "mod"]
    args: list["ShapeFormula"]


ShapeFormula = int | str | FormulaOperation


class FormulaExpression(TypedDict):
    formula: ShapeFormula


DimensionExpression = AffineExpression | FormulaExpression


class _TensorShape(TypedDict):
    shape: list[int | str | None]


class ShapeContract(_TensorShape, total=False):
    dimension_ranges: dict[str, DimensionRange]
    dimension_expressions: dict[str, DimensionExpression]


@dataclass(frozen=True)
class Variable:
    """A named dimension shared by explicitly reusing its name and bounds.

    Unspecified bounds use the domain [1, 2**31-1].
    The compiler must honor this domain; it may not silently specialize it.
    """

    name: str
    min: int = 1
    max: int = 2**31 - 1
    multiple_of: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.name
        ):
            raise ValueError("Variable.name must be an identifier")
        if self.name.startswith("__leapp_"):
            raise ValueError("Variable names starting __leapp_ are reserved")
        if (
            type(self.min) is not int
            or type(self.max) is not int
            or not 0 <= self.min < self.max <= 2**31 - 1
        ):
            raise ValueError("Variable bounds must satisfy 0 <= min < max <= 2**31-1")

        if type(self.multiple_of) is not int or self.multiple_of < 1:
            raise ValueError("Variable.multiple_of must be a positive integer")
        if (
            self.min + self.multiple_of - 1
        ) // self.multiple_of >= self.max // self.multiple_of:
            raise ValueError("Variable domain must contain at least two admitted sizes")

    def to_dict(self):
        result = {"symbol": self.name, "min": self.min, "max": self.max}
        if self.multiple_of != 1:
            result["multiple_of"] = self.multiple_of
        return result


def has_variable_axis_declaration(desc):
    axes = getattr(desc.semantics, "axes", None) or []
    return any(
        a is not None and a.variable is not None and a.variable is not False
        for a in axes
    )


def has_variable_inputs(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return False
    seen.add(id(node))
    return any(
        has_variable_axis_declaration(d)
        or (d.has_source and has_variable_inputs(d.source_node, seen))
        for d in node.inputs
    )


def get_shape_contract(desc) -> ShapeContract:
    if hasattr(desc, "shape_contract"):
        return desc.shape_contract
    return {"shape": list(desc.shape)}


def serialize_shape_axes(desc, axes):
    specification = get_shape_contract(desc)
    axes = copy.deepcopy(axes)
    for index, symbol in enumerate(specification["shape"]):
        if not isinstance(symbol, str):
            continue
        axes.extend([None] * (index + 1 - len(axes)))
        axis = axes[index] or {}
        variable = {"symbol": symbol, **specification["dimension_ranges"][symbol]}
        expression = specification.get("dimension_expressions", {}).get(symbol)
        if expression is not None:
            variable["expression"] = copy.deepcopy(expression)
            roots = (
                _collect_formula_symbols(expression["formula"])
                if "formula" in expression
                else expression["terms"].keys()
            )
            variable["roots"] = {
                root: copy.deepcopy(specification["dimension_ranges"][root])
                for root in sorted(roots)
            }
        axis["variable"] = variable
        axes[index] = axis
    return axes


def parse_shape_contract(description) -> ShapeContract:
    shape = list(get_capture_shape(description))
    ranges: dict[str, DimensionRange] = {}
    expressions: dict[str, DimensionExpression] = {}
    for index, axis in enumerate(description.get("axes") or []):
        variable = (axis or {}).get("variable")
        if not isinstance(variable, dict):
            continue
        symbol = variable["symbol"]
        shape[index] = symbol
        bounds = {
            key: variable[key]
            for key in ("min", "max", "multiple_of")
            if key in variable
        }
        for root, domain in {**variable.get("roots", {}), symbol: bounds}.items():
            if root in ranges and ranges[root] != domain:
                raise ValueError(f"Conflicting bounds for {root}")
            ranges[root] = domain
        if "expression" in variable:
            expressions[symbol] = variable["expression"]
    result: ShapeContract = {"shape": shape}
    if ranges:
        result["dimension_ranges"] = ranges
    if expressions:
        result["dimension_expressions"] = expressions
    return result


def iter_root_dimension_ranges(specification: ShapeContract):
    derived = specification.get("dimension_expressions", {})
    return (
        (symbol, bounds)
        for symbol, bounds in specification.get("dimension_ranges", {}).items()
        if symbol not in derived
    )


def get_shape_contract_version(specifications: list[ShapeContract]) -> int:
    return (
        2
        if any(
            any("multiple_of" in b for b in s.get("dimension_ranges", {}).values())
            or any("formula" in e for e in s.get("dimension_expressions", {}).values())
            for s in specifications
        )
        else 1
    )


def build_declared_shape_contract(desc, prefix: str) -> ShapeContract:
    shape = list(desc.shape)
    ranges: dict[str, DimensionRange] = {}
    for i, axis in enumerate(getattr(desc.semantics, "axes", None) or []):
        if axis is None or axis.variable is None or axis.variable is False:
            continue
        variable = axis.variable
        if variable is True:
            symbol = (
                f"__leapp_axis_{prefix.encode().hex()}_{desc.name.encode().hex()}_{i}"
            )
            bounds = {"min": 1, "max": 2**31 - 1}
        else:
            symbol = variable.name
            bounds = {k: v for k, v in variable.to_dict().items() if k != "symbol"}
        if symbol in ranges and ranges[symbol] != bounds:
            raise ValueError(f"Conflicting bounds for {symbol}")
        ranges[symbol] = bounds
        shape[i] = symbol
    if not ranges:
        return {"shape": shape}
    result: ShapeContract = {
        "shape": shape,
        "dimension_ranges": ranges,
    }
    validate_shape_contract(result, desc.shape, name=desc.name)
    return result


def get_dimension_expression(
    symbol: str, specification: ShapeContract
) -> DimensionExpression:
    return specification.get("dimension_expressions", {}).get(
        symbol, {"terms": {symbol: 1}, "offset": 0}
    )


_FORMULA_OPERATORS = {
    "add": operator.add,
    "mul": operator.mul,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
}


def _encode_shape_formula(expr) -> ShapeFormula:
    if isinstance(expr, sympy.Symbol):
        return str(expr)
    if expr.is_Integer:
        return int(expr)
    numerator, denominator = sympy.fraction(expr)
    if denominator.is_Integer and denominator != 1:
        return {
            "op": "floordiv",
            "args": [_encode_shape_formula(numerator), int(denominator)],
        }
    name = {sympy.Add: "add", sympy.Mul: "mul"}.get(expr.func)
    if name is not None:
        values = [_encode_shape_formula(a) for a in expr.args]
        result = values[0]
        for value in values[1:]:
            result = {"op": name, "args": [result, value]}
        return result
    if expr.func.__name__ in {"FloorDiv", "Mod", "PythonMod"}:
        return {
            "op": "floordiv" if expr.func.__name__ == "FloorDiv" else "mod",
            "args": [_encode_shape_formula(a) for a in expr.args],
        }
    raise ValueError(f"Unsupported runtime output shape formula: {expr}")


def _collect_formula_symbols(formula: ShapeFormula) -> set[str]:
    if isinstance(formula, str):
        return {formula}
    if isinstance(formula, int):
        return set()
    return set().union(*(_collect_formula_symbols(a) for a in formula["args"]))


def _evaluate_formula(formula: ShapeFormula, bindings: dict[str, int]) -> int:
    if isinstance(formula, str):
        return bindings[formula]
    if isinstance(formula, int):
        return formula
    return _FORMULA_OPERATORS[formula["op"]](
        *(_evaluate_formula(a, bindings) for a in formula["args"])
    )


def _compute_formula_bounds(
    formula: ShapeFormula, ranges: dict[str, DimensionRange]
) -> tuple[int, int]:
    if isinstance(formula, str):
        return ranges[formula]["min"], ranges[formula]["max"]
    if isinstance(formula, int):
        return formula, formula
    left, right = [_compute_formula_bounds(a, ranges) for a in formula["args"]]
    op = formula["op"]
    if op in {"floordiv", "mod"} and (right[0] != right[1] or right[0] <= 0):
        raise ValueError(
            "Output shape division/modulo requires a positive constant divisor"
        )
    if op == "mod":
        return 0, right[0] - 1
    values = [_FORMULA_OPERATORS[op](a, b) for a, b in itertools.product(left, right)]
    return min(values), max(values)


def evaluate_dimension_expression(
    expr: DimensionExpression, bindings: dict[str, int]
) -> int:
    if "formula" in expr:
        return _evaluate_formula(expr["formula"], bindings)
    return expr["offset"] + sum(c * bindings[s] for s, c in expr["terms"].items())


def _resolve_expression_roots(
    expression: DimensionExpression,
    specification: ShapeContract,
    expected,
    size,
    bindings,
    name,
):
    if "formula" in expression:
        formula = expression["formula"]
        missing = _collect_formula_symbols(formula) - bindings.keys()
        if missing:
            raise ValueError(
                f"{name}: cannot resolve shape formula roots {sorted(missing)}"
            )
    else:
        expr = expression
        missing = [s for s in expr["terms"] if s not in bindings]
        if len(missing) == 1:
            symbol = missing[0]
            rest = expr["offset"] + sum(
                c * bindings[s] for s, c in expr["terms"].items() if s in bindings
            )
            coefficient = expr["terms"][symbol]
            if (size - rest) % coefficient:
                raise ValueError(
                    f"{name}: size {size} violates derived dimension {expected}"
                )
            bindings[symbol] = (size - rest) // coefficient
            root = specification["dimension_ranges"][symbol]
            if not root["min"] <= bindings[symbol] <= root["max"] or bindings[
                symbol
            ] % root.get("multiple_of", 1):
                raise ValueError(f"{name}: root dimension {symbol} outside bounds")
        elif missing:
            raise ValueError(f"{name}: cannot resolve derived dimensions {missing}")


def validate_shape_contract(
    specification: ShapeContract, actual, bindings=None, *, name="tensor"
):
    bindings = {} if bindings is None else bindings
    if len(actual) != len(specification["shape"]):
        raise ValueError(
            f"{name}: expected rank {len(specification['shape'])}, got {len(actual)}"
        )
    expressions = specification.get("dimension_expressions", {})
    for expected, size in zip(specification["shape"], actual):
        if expected is None:
            continue
        if isinstance(expected, int):
            if expected != size:
                raise ValueError(
                    f"{name}: expected fixed dimension {expected}, got {size}"
                )
            continue
        bounds = specification["dimension_ranges"][expected]
        if not bounds["min"] <= size <= bounds["max"]:
            raise ValueError(
                f"{name}: {expected} must be in [{bounds['min']}, {bounds['max']}], got {size}"
            )
        if size % bounds.get("multiple_of", 1):
            raise ValueError(
                f"{name}: {expected} must be a multiple of {bounds['multiple_of']}"
            )
        if expected in expressions:
            _resolve_expression_roots(
                expressions[expected], specification, expected, size, bindings, name
            )
            predicted = evaluate_dimension_expression(expressions[expected], bindings)
            if predicted != size:
                raise ValueError(
                    f"{name}: derived dimension {expected} must be {predicted}, got {size}"
                )
        if expected in bindings and bindings[expected] != size:
            raise ValueError(
                f"{name}: shared dimension {expected} is {bindings[expected]}, got {size}"
            )
        bindings[expected] = size
    return bindings


def get_capture_shape(specification):
    shape = specification["shape"]
    return json.loads(shape) if isinstance(shape, str) else shape


def inherit_shape_contract(source, target):
    inherited = copy.deepcopy(get_shape_contract(source))
    requested = get_shape_contract(target)
    if len(inherited["shape"]) != len(requested["shape"]):
        raise ValueError("Connected ranks conflict")
    axes = getattr(target.semantics, "axes", None) or []
    for i, axis in enumerate(axes):
        if (
            axis is not None
            and axis.names is not None
            and isinstance(inherited["shape"][i], str)
        ):
            raise ValueError(
                "Inherited variable axis cannot have fixed exhaustive names"
            )
        if axis is None or axis.variable is None:
            continue
        left, right = inherited["shape"][i], requested["shape"][i]
        if isinstance(left, int):
            if isinstance(right, str):
                raise ValueError(
                    f"{target.name}: requested variable dimension is static at producer"
                )
        elif axis.variable is False:
            raise ValueError(
                f"{target.name}: explicit static axis conflicts with producer"
            )
        elif axis.variable is not True:
            offered = inherited["dimension_ranges"][left]
            wanted = requested["dimension_ranges"][right]
            if offered != wanted or left != right:
                raise ValueError(
                    f"{target.name}: explicit consumer dimension conflicts with producer contract"
                )
    target.shape_contract = inherited


def build_torch_dynamic_shapes(node):
    specifications = [get_shape_contract(desc) for desc in node.inputs]
    symbols = {}
    for spec in specifications:
        for symbol, bounds in iter_root_dimension_ranges(spec):
            if symbol in symbols:
                if symbols[symbol][0] != bounds:
                    raise ValueError(
                        f"Conflicting bounds for shared dimension {symbol}"
                    )
                continue
            stride = bounds.get("multiple_of", 1)
            root = torch.export.Dim(
                symbol if stride == 1 else "__leapp_factor_" + symbol,
                min=(bounds["min"] + stride - 1) // stride,
                max=bounds["max"] // stride,
            )
            symbols[symbol] = (bounds, root if stride == 1 else stride * root)
    result = []
    for spec in specifications:
        dimensions = {}
        for i, dimension in enumerate(spec["shape"]):
            if isinstance(dimension, str):
                expr = get_dimension_expression(dimension, spec)
                if "formula" in expr or len(expr["terms"]) != 1:
                    raise ValueError(
                        "Consumer export currently requires a single-root affine dimension"
                    )
                root, coefficient = next(iter(expr["terms"].items()))
                dimensions[i] = symbols[root][1] * coefficient + expr["offset"]
        result.append(dimensions)
    return tuple(result)


def build_onnx_dynamic_axes(node):
    axes = {}
    for desc in node.inputs + node.outputs:
        dimensions = {
            i: size
            for i, size in enumerate(get_shape_contract(desc)["shape"])
            if isinstance(size, str)
        }
        if dimensions:
            axes[desc.name] = dimensions
    return axes or None


def _export_shape_program(node) -> torch.export.ExportedProgram:
    if any(not d.dynamic_array_supported for d in node.inputs + node.outputs):
        raise TypeError(
            "Dynamic Warp ports are unsupported: the APIC runner uses fixed buffers and launch dimensions"
        )
    from leapp.backends.export_backend import prepare_tensors_for_export

    values = tuple(prepare_tensors_for_export([d.value for d in node.inputs]))
    # This is also the constraint proof for legacy/JIT paths; they do not infer
    # their own executable dynamic shape domain from axis names.
    try:
        kwargs = {}
        if (
            node.backend == "exported-program"
            and "strict" in node.export_backend.backend_params
        ):
            kwargs["strict"] = node.export_backend.backend_params["strict"]
        program = torch.export.export(
            node.m.eval(),
            values,
            dynamic_shapes=build_torch_dynamic_shapes(node),
            **kwargs,
        )
    except Exception as exc:
        raise ValueError(
            f"{node.name}: cannot honor requested variable-axis domain; use a nondegenerate example and inspect compiler constraints: {exc}"
        ) from exc
    return program


def _map_input_dimensions(node, program, by_name):
    user_specs = [
        s for s in program.graph_signature.input_specs if s.kind.name == "USER_INPUT"
    ]
    substitutions = {}
    ranges: dict[str, DimensionRange] = {}
    for desc, input_spec in zip(node.inputs, user_specs):
        spec = get_shape_contract(desc)
        for symbol, bounds in spec.get("dimension_ranges", {}).items():
            if symbol in ranges and ranges[symbol] != bounds:
                raise ValueError(f"Conflicting bounds for {symbol}")
            ranges[symbol] = bounds
        actual = by_name[input_spec.arg.name].meta["val"].shape
        for wanted, produced in zip(spec["shape"], actual):
            if isinstance(wanted, str):
                if not isinstance(produced, torch.SymInt):
                    raise ValueError(
                        f"{desc.name}: requested variable axis became static"
                    )
                e = get_dimension_expression(wanted, spec)
                desired = (
                    sum(sympy.Symbol(s) * c for s, c in e["terms"].items())
                    + e["offset"]
                )
                # Input expressions are affine in one compiler root.
                roots = list(produced.node.expr.free_symbols)
                if len(roots) != 1:
                    raise ValueError("Unsupported compiler input dimension expression")
                root = roots[0]
                solved = sympy.solve(produced.node.expr - desired, root)
                if len(solved) != 1 or (
                    root in substitutions and substitutions[root] != solved[0]
                ):
                    raise ValueError("Compiler tied independently declared dimensions")
                substitutions[root] = solved[0]
    return substitutions, ranges


def _infer_output_contract(
    node_name: str, desc, value, substitutions, ranges: dict[str, DimensionRange]
) -> ShapeContract:
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            "Unknown output rank/type is unsupported in the current variable-shape subset"
        )
    shape: list[int | str | None] = []
    derived: dict[str, DimensionExpression] = {}
    output_ranges: dict[str, DimensionRange] = {}
    for axis, size in enumerate(value.shape):
        if not isinstance(size, torch.SymInt):
            shape.append(int(size))
            continue
        expr = sympy.expand(size.node.expr.subs(substitutions))
        if expr.is_Integer:
            shape.append(int(expr))
            continue
        if any(str(s) not in ranges for s in expr.free_symbols):
            raise ValueError(
                "Data-dependent/unknown output dimensions require an explicit supported extension"
            )
        output_ranges.update(
            (str(s), copy.deepcopy(ranges[str(s)])) for s in expr.free_symbols
        )
        if isinstance(expr, sympy.Symbol):
            shape.append(str(expr))
            continue
        coefficients = {str(s): expr.coeff(s) for s in expr.free_symbols}
        offset = expr.subs({s: 0 for s in expr.free_symbols})
        is_affine = (
            offset.is_Integer
            and all(c.is_Integer and c > 0 for c in coefficients.values())
            and sympy.expand(
                expr
                - offset
                - sum(sympy.Symbol(s) * c for s, c in coefficients.items())
            )
            == 0
        )
        symbol = f"__leapp_derived_{node_name.encode().hex()}_{desc.name.encode().hex()}_{axis}"
        while symbol in output_ranges:
            symbol += "_"
        if is_affine:
            terms = {s: int(c) for s, c in coefficients.items()}
            derived[symbol] = {"terms": terms, "offset": int(offset)}
            output_ranges[symbol] = {
                k: int(offset) + sum(c * ranges[s][k] for s, c in terms.items())
                for k in ("min", "max")
            }
        else:
            formula = _encode_shape_formula(expr)
            lower, upper = _compute_formula_bounds(formula, ranges)
            if lower < 0:
                raise ValueError(f"Cannot prove nonnegative output shape: {expr}")
            derived[symbol] = {"formula": formula}
            output_ranges[symbol] = {"min": lower, "max": upper}
        shape.append(symbol)
    inferred: ShapeContract = {
        "shape": shape,
        "dimension_ranges": output_ranges,
    }
    if derived:
        inferred["dimension_expressions"] = derived
    if not any(isinstance(s, str) for s in shape):
        inferred = {"shape": shape}
    return inferred


def _validate_output_contract(node_name: str, desc, inferred: ShapeContract):
    requested = build_declared_shape_contract(desc, node_name)
    for i, axis in enumerate(getattr(desc.semantics, "axes", None) or []):
        if axis is None:
            continue
        actual, wanted = inferred["shape"][i], requested["shape"][i]
        if isinstance(actual, str) and axis.names is not None:
            raise ValueError(
                "Inferred variable output axis cannot have fixed exhaustive names"
            )
        if axis.variable is False and isinstance(actual, str):
            raise ValueError(
                "Explicit static output axis conflicts with inferred variability"
            )
        if axis.variable is True and not isinstance(actual, str):
            raise ValueError("Requested variable output became static")
        if isinstance(axis.variable, Variable) and (
            actual != wanted
            or inferred.get("dimension_ranges", {}).get(actual)
            != requested["dimension_ranges"][wanted]
        ):
            raise ValueError(
                f"{desc.name}: explicit output axis conflicts with inferred contract"
            )


def _validate_cached_shapes(node):
    for _, inputs, outputs in node._build_validation_examples():
        bindings = {}
        for desc, tensor in zip(
            node.inputs + node.outputs, list(inputs) + list(outputs)
        ):
            validate_shape_contract(
                get_shape_contract(desc),
                tensor.shape,
                bindings,
                name=f"{node.name}/{desc.name}",
            )


def export_and_infer_shape_contracts(node):
    program = _export_shape_program(node)
    node.shape_program = program
    by_name = {n.name: n for n in program.graph.nodes}
    substitutions, ranges = _map_input_dimensions(node, program, by_name)
    output_specs = [
        s for s in program.graph_signature.output_specs if s.kind.name == "USER_OUTPUT"
    ]
    if len(output_specs) != len(node.outputs):
        raise ValueError("Exported user-output count does not match node ports")
    for desc, out in zip(node.outputs, output_specs):
        value = by_name[out.arg.name].meta.get("val")
        inferred = _infer_output_contract(node.name, desc, value, substitutions, ranges)
        _validate_output_contract(node.name, desc, inferred)
        desc.shape_contract = inferred
    _validate_cached_shapes(node)


def prepare_shape_contracts(node):
    output_requested = any(has_variable_axis_declaration(d) for d in node.outputs)
    if not has_variable_inputs(node) and not output_requested:
        return
    for desc in node.inputs:
        desc.shape_contract = build_declared_shape_contract(desc, node.name)
        if desc.has_source and desc.source_node.node_index < node.node_index:
            source = next(
                (d for d in desc.source_node.outputs if d.port == desc.port), None
            )
            if source is None:
                raise ValueError(f"Missing producer port for {node.name}/{desc.name}")
            inherit_shape_contract(source, desc)
    if output_requested or any(
        isinstance(size, str)
        for d in node.inputs
        for size in get_shape_contract(d)["shape"]
    ):
        if not node.exports_model:
            raise ValueError("Variable axes require an executable export backend")
        for desc in node.inputs:
            if desc.has_source and desc.source_node.node_index >= node.node_index:
                raise ValueError(
                    "Variable feedback-state shapes are outside the current variable-shape subset"
                )
        export_and_infer_shape_contracts(node)


def validate_reentry_variable_axes(metadata, descriptions):
    """Changing accepted shapes requires a new capture, not metadata enrichment."""
    for desc in descriptions:
        current = metadata.get(desc.name)
        if current is None:
            continue
        old = [
            getattr(a, "variable", None)
            for a in getattr(desc.semantics, "axes", None) or []
        ]
        new = [getattr(a, "variable", None) for a in current.axes or []]
        if old != new and any(v is not None and v is not False for v in old + new):
            raise ValueError(
                f"{desc.name}: variable declarations changed on re-entry; recapture the node"
            )


def validate_variable_declarations(nodes):
    declarations = {}
    for node in nodes:
        for desc in node.inputs + node.outputs:
            for axis in getattr(desc.semantics, "axes", None) or []:
                variable = getattr(axis, "variable", None)
                if isinstance(variable, Variable):
                    if (
                        variable.name in declarations
                        and declarations[variable.name] != variable
                    ):
                        raise ValueError(
                            f"Conflicting bounds for shared variable {variable.name}"
                        )
                    declarations[variable.name] = variable
