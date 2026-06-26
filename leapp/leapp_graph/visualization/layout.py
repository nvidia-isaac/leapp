from __future__ import annotations

from dataclasses import dataclass

from fast_sugiyama import from_edges

from .model import VisualGraph


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class LayoutResult:
    centers: dict[str, Point]
    forward_edge_ids: tuple[str, ...]


_LAYOUT_EDGE_KINDS = frozenset({"forward", "graph_input", "graph_output"})
_GRID_X_STEP = 260.0
_GRID_Y_STEP = 140.0


def compute_layered_layout(graph: VisualGraph) -> LayoutResult:
    visual_ids = tuple(sorted(graph.visual_ids()))
    if not visual_ids:
        return LayoutResult(centers={}, forward_edge_ids=())

    layout_edges = tuple(edge for edge in graph.edges if edge.kind in _LAYOUT_EDGE_KINDS)
    if not layout_edges:
        return LayoutResult(
            centers=_deterministic_grid(visual_ids),
            forward_edge_ids=tuple(),
        )

    id_to_index = {visual_id: index for index, visual_id in enumerate(visual_ids)}
    edge_pairs = sorted(
        (id_to_index[edge.source_id], id_to_index[edge.target_id]) for edge in layout_edges
    )
    raw_positions = _compute_sugiyama_positions(edge_pairs)
    centers = _normalize_points(_decode_positions(visual_ids, raw_positions))
    missing_ids = tuple(visual_id for visual_id in visual_ids if visual_id not in centers)
    if missing_ids:
        centers.update(_place_missing_ids(centers, missing_ids))

    return LayoutResult(
        centers=centers,
        forward_edge_ids=tuple(edge.id for edge in layout_edges),
    )


def _compute_sugiyama_positions(edge_pairs: list[tuple[int, int]]) -> dict[int, tuple[float, float]]:
    layouts = from_edges(
        edge_pairs,
        vertex_spacing=96,
        dummy_vertices=False,
        crossing_minimization="median",
        check_layout=True,
    ).dot_layout(spacing=96)
    return layouts.to_dict()


def _decode_positions(
    visual_ids: tuple[str, ...],
    raw_positions: dict[int, tuple[float, float]],
) -> dict[str, Point]:
    raw_max_y = max(raw_y for _, raw_y in raw_positions.values())
    return {
        visual_ids[index]: Point(
            x=(raw_max_y - raw_y) * 2.6,
            y=raw_x * 1.8,
        )
        for index, (raw_x, raw_y) in raw_positions.items()
    }


def _normalize_points(points: dict[str, Point]) -> dict[str, Point]:
    min_x = min(point.x for point in points.values())
    min_y = min(point.y for point in points.values())
    return {
        visual_id: Point(x=point.x - min_x, y=point.y - min_y)
        for visual_id, point in points.items()
    }


def _place_missing_ids(points: dict[str, Point], missing_ids: tuple[str, ...]) -> dict[str, Point]:
    next_x = max(point.x for point in points.values()) + _GRID_X_STEP
    return {
        visual_id: Point(x=next_x, y=row * _GRID_Y_STEP)
        for row, visual_id in enumerate(sorted(missing_ids))
    }


def _deterministic_grid(visual_ids: tuple[str, ...]) -> dict[str, Point]:
    return {
        visual_id: Point(x=index * _GRID_X_STEP, y=0.0)
        for index, visual_id in enumerate(visual_ids)
    }
