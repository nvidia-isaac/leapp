// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capture dimensions remain public; execution contracts are compiled once at load.

use super::{
    schema::{BundleDescription, NodeSpec},
    MetadataValue, PortSpec,
};
use anyhow::{ensure, Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;

/// Bundle-wide dimension symbols. Ports refer to them by index.
pub(crate) type Symbols = IndexMap<String, Domain>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Domain {
    min: i64,
    max: i64,
    multiple_of: i64,
}

impl Domain {
    fn accepts(&self, size: i64) -> bool {
        self.min <= size && size <= self.max && size % self.multiple_of == 0
    }
}

/// A port's execution shape: capture sizes, with each variable axis replaced by
/// -1 and checked against its bounds and shared symbol instead.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExecutionShape {
    pub dims: Vec<i64>,
    bounds: Vec<AxisBounds>,
}

#[derive(Clone, Debug, PartialEq)]
struct AxisBounds {
    axis: usize,
    symbol: usize,
    domain: Domain,
    expression: Option<Expression>,
}

impl From<Vec<i64>> for ExecutionShape {
    fn from(dims: Vec<i64>) -> Self {
        Self {
            dims,
            bounds: Vec::new(),
        }
    }
}

impl ExecutionShape {
    pub fn is_bounded(&self) -> bool {
        !self.bounds.is_empty()
    }

    /// Rank, fixed sizes and per-axis bounds. Shared symbols are checked by `bind`.
    pub fn matches(&self, actual: &[i64]) -> bool {
        actual.iter().all(|&size| size >= 0)
            && self.compatible(actual)
            && self
                .bounds
                .iter()
                .all(|bound| bound.domain.accepts(actual[bound.axis]))
    }

    /// Same rank, with equal sizes wherever neither side is dynamic (negative).
    pub fn compatible(&self, other: &[i64]) -> bool {
        self.dims.len() == other.len()
            && self
                .dims
                .iter()
                .zip(other)
                .all(|(&a, &b)| a < 0 || b < 0 || a == b)
    }

    /// Binds this port's symbols for one invocation. `actual` has passed `matches`.
    pub fn bind(
        &self,
        actual: &[i64],
        symbols: &Symbols,
        values: &mut [Option<i64>],
    ) -> Result<()> {
        for bound in &self.bounds {
            let size = actual[bound.axis];
            if let Some(expression) = &bound.expression {
                // A connected model may receive only an affine-derived dimension.
                // Match PR #58: infer its one missing root, never invert // or %.
                if let Expression::Affine(terms, offset) = expression {
                    if let (sum, Some((id, scale))) = affine_sum(terms, *offset, values)? {
                        let delta = checked(size.checked_sub(sum))?;
                        ensure!(
                            delta.checked_rem(scale) == Some(0),
                            "size violates derived dimension"
                        );
                        bind_symbol(symbols, values, id, checked(delta.checked_div(scale))?)?;
                    }
                }
                ensure!(
                    expression.evaluate(values)? == size,
                    "derived dimension does not match actual size {size}"
                );
            }
            bind_symbol(symbols, values, bound.symbol, size)?;
        }
        Ok(())
    }
}

fn bind_symbol(symbols: &Symbols, values: &mut [Option<i64>], id: usize, size: i64) -> Result<()> {
    let (name, domain) = symbols.get_index(id).unwrap();
    ensure!(domain.accepts(size), "symbol {name} outside bounds: {size}");
    ensure!(
        values[id].is_none_or(|previous| previous == size),
        "shared dimension {name} differs: {:?} versus {size}",
        values[id]
    );
    values[id] = Some(size);
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
enum Expression {
    Constant(i64),
    Symbol(usize),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    FloorDiv(Box<Self>, Box<Self>),
    Mod(Box<Self>, Box<Self>),
    /// `offset + sum(scale * symbol)`: the only form whose missing root is inferred.
    Affine(Vec<(usize, i64)>, i64),
}

/// One `axes[].variable` declaration.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Variable {
    symbol: String,
    min: i64,
    max: i64,
    multiple_of: Option<i64>,
    expression: Option<MetadataValue>,
    roots: Option<IndexMap<String, Root>>,
}

#[derive(Deserialize)]
struct Root {
    min: i64,
    max: i64,
    multiple_of: Option<i64>,
}

#[derive(Deserialize)]
struct Affine {
    terms: IndexMap<String, i64>,
    offset: i64,
}

fn checked(value: Option<i64>) -> Result<i64> {
    value.context("shape arithmetic overflow")
}

/// Sum of the resolved affine terms, and the single unresolved term if any.
fn affine_sum(
    terms: &[(usize, i64)],
    offset: i64,
    values: &[Option<i64>],
) -> Result<(i64, Option<(usize, i64)>)> {
    let mut sum = offset;
    let mut missing = None;
    for &(id, scale) in terms {
        match values[id] {
            Some(value) => {
                sum = checked(value.checked_mul(scale).and_then(|v| sum.checked_add(v)))?
            }
            None => {
                ensure!(
                    missing.is_none(),
                    "cannot resolve multiple affine shape roots"
                );
                missing = Some((id, scale));
            }
        }
    }
    Ok((sum, missing))
}

impl Expression {
    fn evaluate(&self, values: &[Option<i64>]) -> Result<i64> {
        let operands = |a: &Self, b: &Self| -> Result<(i64, i64)> {
            Ok((a.evaluate(values)?, b.evaluate(values)?))
        };
        let divisor = |a: &Self, b: &Self| -> Result<(i64, i64)> {
            let (a, b) = operands(a, b)?;
            ensure!(b > 0, "shape divisor must be positive");
            Ok((a, b))
        };
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Symbol(id) => values[*id].context("unresolved shape formula root"),
            Self::Add(a, b) => operands(a, b).and_then(|(a, b)| checked(a.checked_add(b))),
            Self::Mul(a, b) => operands(a, b).and_then(|(a, b)| checked(a.checked_mul(b))),
            Self::FloorDiv(a, b) => divisor(a, b).map(|(a, b)| a.div_euclid(b)),
            Self::Mod(a, b) => divisor(a, b).map(|(a, b)| a.rem_euclid(b)),
            Self::Affine(terms, offset) => match affine_sum(terms, *offset, values)? {
                (sum, None) => Ok(sum),
                (_, Some(_)) => anyhow::bail!("unresolved affine shape root"),
            },
        }
    }

    /// An integer `formula` tree, or affine `terms` and `offset`.
    fn from_metadata(value: &MetadataValue, symbols: &Symbols) -> Result<Self> {
        if let Some(formula) = value.get("formula") {
            return Self::parse(formula, symbols);
        }
        let Affine { terms, offset } = Affine::deserialize(value)?;
        let terms = terms
            .into_iter()
            .map(|(name, scale)| {
                ensure!(scale != 0, "affine coefficient must be nonzero");
                let id = symbols
                    .get_index_of(&name)
                    .context("undeclared affine root")?;
                Ok((id, scale))
            })
            .collect::<Result<_>>()?;
        Ok(Self::Affine(terms, offset))
    }

    fn parse(value: &MetadataValue, symbols: &Symbols) -> Result<Self> {
        if let Some(value) = value.as_i64() {
            return Ok(Self::Constant(value));
        }
        if let Some(name) = value.as_str() {
            return Ok(Self::Symbol(
                symbols
                    .get_index_of(name)
                    .context("undeclared shape root")?,
            ));
        }
        let args = value["args"]
            .as_sequence()
            .context("shape formula requires args")?;
        ensure!(args.len() == 2, "shape formula requires two operands");
        let a = Box::new(Self::parse(&args[0], symbols)?);
        let b = Box::new(Self::parse(&args[1], symbols)?);
        Ok(match value["op"].as_str() {
            Some("add") => Self::Add(a, b),
            Some("mul") => Self::Mul(a, b),
            Some("floordiv") => Self::FloorDiv(a, b),
            Some("mod") => Self::Mod(a, b),
            _ => anyhow::bail!("unsupported shape formula operator"),
        })
    }
}

/// Compiles every port once, storing shapes on each `NodeSpec` and the symbols
/// on the description. Later stages never reparse `axes` metadata.
pub(crate) fn compile(description: &mut BundleDescription) -> Result<()> {
    let version = description.system_information.get("shape contract version");
    if let Some(version) = version {
        ensure!(
            matches!(version.as_i64(), Some(1 | 2)),
            "unsupported shape contract version {version:?}"
        );
    }
    let version = version.and_then(MetadataValue::as_i64);
    let mut compiler = Compiler::default();
    for (name, model) in &mut description.models {
        compiler.node(name, model)?;
    }
    if !compiler.symbols.is_empty() {
        ensure!(
            version.is_some(),
            "variable axes require shape contract version 1 or 2"
        );
        ensure!(
            !compiler.needs_v2 || version == Some(2),
            "divisibility and shape formulas require shape contract version 2"
        );
    }
    description.symbols = compiler.symbols;
    Ok(())
}

/// Compiles a node built outside a bundle, with node-local symbols.
#[cfg(test)]
pub(crate) fn compile_node(model: &mut NodeSpec) -> Result<()> {
    Compiler::default().node("test", model)
}

#[derive(Default)]
struct Compiler {
    symbols: Symbols,
    needs_v2: bool,
}

impl Compiler {
    fn node(&mut self, name: &str, model: &mut NodeSpec) -> Result<()> {
        let mut ports = |ports: &[PortSpec]| {
            ports
                .iter()
                .map(|port| {
                    self.port(port)
                        .with_context(|| format!("port {name}/{}", port.name))
                })
                .collect::<Result<Vec<_>>>()
        };
        model.input_shapes = ports(&model.inputs)?;
        model.output_shapes = ports(&model.outputs)?;
        Ok(())
    }

    fn port(&mut self, port: &PortSpec) -> Result<ExecutionShape> {
        let mut shape = ExecutionShape::from(port.shape.clone());
        let Some(axes) = port.metadata.get("axes").filter(|v| !v.is_null()) else {
            return Ok(shape);
        };
        let axes = axes.as_sequence().context("axes must be a list")?;
        ensure!(axes.len() <= shape.dims.len(), "axes exceed tensor rank");
        for (axis, metadata) in axes.iter().enumerate() {
            let Some(variable) = metadata
                .get("variable")
                .filter(|v| !v.is_null() && v.as_bool() != Some(false))
            else {
                continue;
            };
            let bound = self.axis(axis, Variable::deserialize(variable)?)?;
            ensure!(
                bound.domain.accepts(shape.dims[axis]),
                "port {} capture shape violates axis {axis} bounds",
                port.name
            );
            shape.dims[axis] = -1;
            shape.bounds.push(bound);
        }
        Ok(shape)
    }

    fn axis(&mut self, axis: usize, variable: Variable) -> Result<AxisBounds> {
        let domain = self.domain(variable.min, variable.max, variable.multiple_of)?;
        let symbol = self.symbol(&variable.symbol, domain)?;
        if let Some(roots) = variable.roots {
            ensure!(variable.expression.is_some(), "roots require an expression");
            for (name, root) in roots {
                let domain = self.domain(root.min, root.max, root.multiple_of)?;
                self.symbol(&name, domain)?;
            }
        }
        let expression = match variable.expression {
            Some(expression) => {
                self.needs_v2 |= expression.get("formula").is_some();
                Some(Expression::from_metadata(&expression, &self.symbols)?)
            }
            None => None,
        };
        Ok(AxisBounds {
            axis,
            symbol,
            domain,
            expression,
        })
    }

    fn domain(&mut self, min: i64, max: i64, multiple_of: Option<i64>) -> Result<Domain> {
        self.needs_v2 |= multiple_of.is_some();
        let multiple_of = multiple_of.unwrap_or(1);
        ensure!(
            0 <= min && min <= max && max <= i64::from(i32::MAX) && multiple_of > 0,
            "invalid finite variable-axis bounds"
        );
        Ok(Domain {
            min,
            max,
            multiple_of,
        })
    }

    fn symbol(&mut self, name: &str, domain: Domain) -> Result<usize> {
        ensure!(!name.is_empty(), "variable symbol must be nonempty");
        let (id, previous) = self.symbols.insert_full(name.to_owned(), domain);
        ensure!(
            previous.is_none_or(|previous| previous == domain),
            "conflicting bounds for symbol {name}"
        );
        Ok(id)
    }
}

#[cfg(test)]
#[path = "../../tests/unit_tests/bundle/shapes.rs"]
mod tests;
