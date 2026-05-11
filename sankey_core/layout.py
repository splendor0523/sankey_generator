
"""
layout.py
----------
All geometry/layout computations:
- Column config resolution (node widths, gaps, x positions)
- Group-gap support
- Node center stacking
- Band assignment within nodes for edges
- Label density detection (decide which labels to show)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .graph import SankeyConfig, GraphData
from .colors import normalize_hex_or_default


def px_to_frac_y(px: float, fig_h_in: float, dpi: int) -> float:
    return px / (fig_h_in * dpi)


def px_to_frac_x(px: float, fig_w_in: float, dpi: int) -> float:
    return px / (fig_w_in * dpi)


# --- 分组间隙计算逻辑 ---
def calculate_total_height_with_groups(
    h_list: List[float],
    node_names: List[str],
    col_idx: int,
    in_edges_by_col: List[Dict[str, List[int]]],
    edge_src_nodes: List[str],
    edge_w_px: List[float],
    gap: float,
    group_gap: float,
    enable_group: bool,
) -> float:
    """
    Column total height including optional extra gaps between "groups".
    Grouping heuristic: in a given column, nodes with different "major source"
    (largest incoming contribution) are separated by an additional `group_gap`.

    Supports "skip columns" links by computing major sources from ALL incoming edges.
    """
    if not h_list:
        return 0.0

    total = sum(h_list) + gap * max(0, (len(h_list) - 1))
    if (not enable_group) or col_idx == 0:
        return total

    extra_gap = 0.0
    prev_major_source = None

    for i, node_name in enumerate(node_names):
        incoming = in_edges_by_col[col_idx].get(node_name, []) if 0 <= col_idx < len(in_edges_by_col) else []
        if not incoming:
            major_source = f"___NO_SOURCE___{col_idx}_{i}"
        else:
            source_weights = defaultdict(float)
            for eid in incoming:
                if 0 <= eid < len(edge_w_px):
                    w = float(edge_w_px[eid])
                    if w > 0:
                        src = edge_src_nodes[eid] if 0 <= eid < len(edge_src_nodes) else "___NO_SOURCE___"
                        source_weights[src] += w
            if not source_weights:
                major_source = f"___NO_SOURCE___{col_idx}_{i}"
            else:
                major_source = max(source_weights, key=source_weights.get)

        if i > 0 and major_source != prev_major_source:
            extra_gap += group_gap
        prev_major_source = major_source

    return total + extra_gap


def stack_centers_topdown_with_grouping(
    h_list: List[float],
    gap: float,
    top_y: float,
    node_names: List[str],
    col_idx: int,
    in_edges_by_col: List[Dict[str, List[int]]],
    edge_src_nodes: List[str],
    edge_w_px: List[float],
    group_gap: float,
    enable_group: bool,
) -> List[float]:
    """
    Stack node centers from top to bottom, inserting an extra gap between groups.
    Grouping heuristic is the same as `calculate_total_height_with_groups`.
    """
    centers: List[float] = []
    y = float(top_y)

    if (not enable_group) or col_idx == 0:
        for h in h_list:
            centers.append(y - h / 2)
            y -= (h + gap)
        return centers

    prev_major_source = None

    for i, (h, node_name) in enumerate(zip(h_list, node_names)):
        incoming = in_edges_by_col[col_idx].get(node_name, []) if 0 <= col_idx < len(in_edges_by_col) else []
        if not incoming:
            major_source = f"___NO_SOURCE___{col_idx}_{i}"
        else:
            source_weights = defaultdict(float)
            for eid in incoming:
                if 0 <= eid < len(edge_w_px):
                    w = float(edge_w_px[eid])
                    if w > 0:
                        src = edge_src_nodes[eid] if 0 <= eid < len(edge_src_nodes) else "___NO_SOURCE___"
                        source_weights[src] += w
            if not source_weights:
                major_source = f"___NO_SOURCE___{col_idx}_{i}"
            else:
                major_source = max(source_weights, key=source_weights.get)

        if i > 0 and major_source != prev_major_source:
            y -= group_gap
        prev_major_source = major_source

        centers.append(y - h / 2)
        y -= (h + gap)

    return centers


def assign_bands_centered(node_center: float, node_h: float, row_ids: List[int], link_h_list: List[float]) -> Dict[int, float]:
    band_y: Dict[int, float] = {}
    if not row_ids:
        return band_y
    top = node_center + node_h / 2
    total = sum(link_h_list[i] for i in row_ids)
    cur = top - (node_h - total) / 2 if total < node_h else top
    for ridx in row_ids:
        h = link_h_list[ridx]
        band_y[ridx] = cur - h / 2
        cur -= h
    return band_y


def _select_labels_by_density(candidates: List[Dict[str, Any]], min_sep_px: float) -> set:
    cand_sorted = sorted(candidates, key=lambda c: (-c["priority"], c["orig_idx"]))
    accepted = []
    for c in cand_sorted:
        y = c["y_px"]
        ok = True
        for a in accepted:
            if abs(y - a["y_px"]) < min_sep_px:
                ok = False
                break
        if ok:
            accepted.append(c)
    return {c["name"] for c in accepted}


# --------------------- Column layout config ---------------------
def build_col_cfg_dict(cfg: SankeyConfig) -> Dict[Any, Dict[str, Any]]:
    d = {
        "default": {
            "node_width_px": cfg.default_node_width_px,
            "gap_px": cfg.default_gap_px,
            "x": None,
            "align": "center",
            "group_gap_on": True,
            "group_gap_px": 20.0
        }
    }
    rows = cfg.col_cfg_rows or []
    for r in rows:
        j = int(r.get("col_index"))
        x = r.get("x")
        if x is None or (isinstance(x, float) and math.isnan(x)):
            x = None
        else:
            x = float(x)

        d[j] = {
            "x": x,
            "node_width_px": float(r.get("node_width_px", cfg.default_node_width_px)),
            "gap_px": float(r.get("gap_px", cfg.default_gap_px)),
            "align": str(r.get("align", "center")).strip().lower(),
            "group_gap_on": bool(r.get("group_gap_on", True)),
            "group_gap_px": float(r.get("group_gap_px", 20.0)),
        }
    return d


def _get_col_param(col_cfg: Dict[Any, Dict[str, Any]], j: int, key: str):
    base = dict(col_cfg.get("default", {}))
    base.update(col_cfg.get(j, {}))
    return base.get(key, None)


def resolve_col_lists(col_cfg: Dict[Any, Dict[str, Any]], n: int) -> Tuple[List[float], List[float]]:
    node_w_px: List[float] = []
    gap_px: List[float] = []
    for j in range(n):
        w = _get_col_param(col_cfg, j, "node_width_px")
        g = _get_col_param(col_cfg, j, "gap_px")
        node_w_px.append(float(w) if w is not None else 100.0)
        gap_px.append(float(g) if g is not None else 50.0)
    return node_w_px, gap_px


def resolve_x_positions(col_cfg: Dict[Any, Dict[str, Any]], n: int, x_min: float, x_max: float) -> List[float]:
    xs = [_get_col_param(col_cfg, j, "x") for j in range(n)]

    if all(v is None for v in xs):
        if n == 1:
            return [(x_min + x_max) / 2]
        return [x_min + (x_max - x_min) * j / (n - 1) for j in range(n)]

    known = [(i, xs[i]) for i in range(n) if xs[i] is not None]
    known.sort(key=lambda t: t[0])

    first_i, first_x = known[0]
    if first_i > 0:
        for i in range(0, first_i):
            xs[i] = x_min + (first_x - x_min) * (i / first_i)

    for (i0, x0), (i1, x1) in zip(known, known[1:]):
        gap = i1 - i0
        if gap <= 1:
            continue
        for k in range(1, gap):
            t = k / gap
            xs[i0 + k] = x0 + (x1 - x0) * t

    last_i, last_x = known[-1]
    if last_i < n - 1:
        for i in range(last_i + 1, n):
            t = (i - last_i) / ((n - 1) - last_i)
            xs[i] = last_x + (x_max - last_x) * t

    for i in range(n):
        if xs[i] is None:
            xs[i] = x_min + (x_max - x_min) * (i / (n - 1) if n > 1 else 0)

    return xs


# --------------------- Column label grouped config ---------------------
def build_col_label_cfg(cfg: SankeyConfig) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in (cfg.col_label_cfg_rows or []):
        j = int(r.get("col_index"))
        out[j] = {
            "show": bool(r.get("show", True)),
            "pos": str(r.get("pos", "auto")).strip() or "auto",
            "text_color": normalize_hex_or_default(r.get("text_color"), cfg.label_text_color_default),
            "dx_px": float(r.get("dx_px", 0.0) or 0.0),
            "dy_px": float(r.get("dy_px", 0.0) or 0.0),
            "font_size": None if r.get("font_size") in (None, "", "nan") else float(r.get("font_size")),
            "use_node_color": bool(r.get("use_node_color", False)),
            "bold": bool(r.get("bold", False)),
            "italic": bool(r.get("italic", False)),
            "underline": bool(r.get("underline", False)),
        }
    return out


@dataclass
class LayoutData:
    # column config
    col_cfg_dict: Dict[Any, Dict[str, Any]]
    node_w_px_col: List[float]
    gap_px_col: List[float]
    node_w_frac_col: List[float]
    col_label_cfg: Dict[int, Dict[str, Any]]

    # normalized heights
    col_h_frac: List[List[float]]
    edge_h_frac: List[float]
    # thickness at each end of the link (when gradient enabled)
    edge_h_frac_in: List[float]
    edge_h_frac_out: List[float]

    gap_frac: List[float]
    frame_top: List[float]
    col_centers: List[List[float]]

    # bands
    in_band_y: List[Dict[int, float]]
    out_band_y: List[Dict[int, float]]

    # x
    x_pos: List[float]
    gapx0: float

    # labels
    accepted_labels_per_col: List[set]
    label_offset_frac: float
    dx_frac_per_col: List[float]
    dy_frac_per_col: List[float]

    # scale
    scale_applied: float


def compute_layout(g: GraphData, cfg: SankeyConfig) -> LayoutData:
    """
    Compute all y/x positions and band allocations, and decide label density.
    Mirrors the original layout logic.
    """
    # -------- Column config / labels config --------
    col_cfg_dict = build_col_cfg_dict(cfg)
    node_w_px_col, gap_px_col = resolve_col_lists(col_cfg_dict, cfg.n_cols)
    node_w_frac_col = [w / (cfg.fig_width_in * cfg.dpi) for w in node_w_px_col]
    col_label_cfg = build_col_label_cfg(cfg)

    # heights in fraction units
    col_h_frac: List[List[float]] = []
    for j in range(cfg.n_cols):
        col_h_frac.append([px_to_frac_y(g.col_h_px[j][n], cfg.fig_height_in, cfg.dpi) for n in g.col_nodes[j]])

    edge_h_frac: List[float] = [px_to_frac_y(px, cfg.fig_height_in, cfg.dpi) for px in g.edge_w_px]
    gap_frac0 = [px_to_frac_y(gap_px_col[j], cfg.fig_height_in, cfg.dpi) for j in range(cfg.n_cols)]
    available = (cfg.y_max - cfg.y_min)

    # -------- Column totals (for vertical alignment & scaling to fit) --------
    col_h_totals: List[float] = []
    for j in range(cfg.n_cols):
        col_group_on = _get_col_param(col_cfg_dict, j, "group_gap_on")
        col_group_gap_px = _get_col_param(col_cfg_dict, j, "group_gap_px")

        final_group_enable = cfg.enable_group_gap and bool(col_group_on)
        final_group_gap_frac = px_to_frac_y(float(col_group_gap_px), cfg.fig_height_in, cfg.dpi)

        th = calculate_total_height_with_groups(
            h_list=col_h_frac[j],
            node_names=g.col_nodes[j],
            col_idx=j,
            in_edges_by_col=[d for d in g.in_edges_by_col],
            edge_src_nodes=getattr(g, 'edge_group_src_node', g.edge_src_node),
            edge_w_px=g.edge_w_px,
            gap=gap_frac0[j],
            group_gap=final_group_gap_frac,
            enable_group=final_group_enable
        )
        col_h_totals.append(th)

    H_manual_base = max(col_h_totals) if col_h_totals else 0.0

    scale_applied = 1.0
    if H_manual_base > available and H_manual_base > 0:
        s = available / H_manual_base
        scale_applied = s
        for j in range(cfg.n_cols):
            col_h_frac[j] = [h * s for h in col_h_frac[j]]
            gap_frac0[j] *= s
        edge_h_frac = [h * s for h in edge_h_frac]
        col_h_totals = [h * s for h in col_h_totals]

    gap_frac = gap_frac0[:]
    frame_top = [cfg.y_max] * cfg.n_cols

    if cfg.layout_ref_col_index is not None and 0 <= cfg.layout_ref_col_index < cfg.n_cols:
        ref_h = col_h_totals[cfg.layout_ref_col_index]
    else:
        ref_h = max(col_h_totals) if col_h_totals else 0.0

    y_center_global = (cfg.y_min + cfg.y_max) / 2
    ref_top_y = y_center_global + ref_h / 2
    ref_bottom_y = y_center_global - ref_h / 2

    if cfg.force_align_top_bottom:
        # Strong top-bottom align for participating columns; exempt columns keep per-column align/gap.
        raw_exempt = getattr(cfg, "force_align_exempt_cols", tuple()) or tuple()
        exempt_set = set()
        for v in raw_exempt:
            try:
                j = int(v)
                if 0 <= j < cfg.n_cols:
                    exempt_set.add(j)
            except Exception:
                continue
        participating_cols = [j for j in range(cfg.n_cols) if j not in exempt_set]

        if participating_cols:
            H_base = max(col_h_totals[j] for j in participating_cols)
        else:
            H_base = ref_h

        global_top = cfg.y_max
        global_bottom = cfg.y_min
        if H_base < available:
            offset = (available - H_base) / 2.0
            global_top = cfg.y_max - offset
            global_bottom = cfg.y_min + offset

        for j in range(cfg.n_cols):
            curr = col_h_totals[j] if j < len(col_h_totals) else 0.0
            n_nodes = len(col_h_frac[j]) if j < len(col_h_frac) else 0

            if j in exempt_set:
                align = _get_col_param(col_cfg_dict, j, "align") or "center"
                if align == "top":
                    frame_top[j] = ref_top_y
                elif align == "bottom":
                    frame_top[j] = ref_bottom_y + curr
                else:
                    frame_top[j] = y_center_global + curr / 2
                continue

            if n_nodes <= 1:
                frame_top[j] = global_top - (H_base - curr) / 2.0
            else:
                extra = H_base - curr
                if extra > 1e-12:
                    n_gaps = max(1, n_nodes - 1)
                    add_per_gap = extra / float(n_gaps)
                    gap_frac[j] = gap_frac[j] + add_per_gap
                frame_top[j] = global_top
    else:
        # Base column + per-column align
        for j in range(cfg.n_cols):
            align = _get_col_param(col_cfg_dict, j, "align") or "center"
            this_h = col_h_totals[j]

            if align == "top":
                frame_top[j] = ref_top_y
            elif align == "bottom":
                frame_top[j] = ref_bottom_y + this_h
            else:
                frame_top[j] = y_center_global + this_h / 2

    # -------- Compute node centers per column --------
    col_centers: List[List[float]] = []
    for j in range(cfg.n_cols):
        col_group_on = _get_col_param(col_cfg_dict, j, "group_gap_on")
        col_group_gap_px = _get_col_param(col_cfg_dict, j, "group_gap_px")

        final_group_enable = cfg.enable_group_gap and bool(col_group_on)
        raw_gap_frac = px_to_frac_y(float(col_group_gap_px), cfg.fig_height_in, cfg.dpi)
        final_group_gap_frac = raw_gap_frac * scale_applied

        centers = stack_centers_topdown_with_grouping(
            h_list=col_h_frac[j],
            gap=gap_frac[j],
            top_y=frame_top[j],
            node_names=g.col_nodes[j],
            col_idx=j,
            in_edges_by_col=[d for d in g.in_edges_by_col],
            edge_src_nodes=getattr(g, 'edge_group_src_node', g.edge_src_node),
            edge_w_px=g.edge_w_px,
            group_gap=final_group_gap_frac,
            enable_group=final_group_enable
        )
        col_centers.append(centers)

    # -------- Optional: edge thickness taper between node heights --------
    # Default: keep both ends identical.
    edge_h_frac_in = edge_h_frac[:]
    edge_h_frac_out = edge_h_frac[:]

    # New semantics: auto_balance_flow controls visual taper only.
    if bool(getattr(cfg, 'auto_balance_flow', False)) and edge_h_frac:
        n_edges = len(edge_h_frac)
        for j in range(cfg.n_cols):
            for i_node, node in enumerate(g.col_nodes[j]):
                h_node = col_h_frac[j][i_node] if i_node < len(col_h_frac[j]) else 0.0
                if h_node <= 0:
                    continue

                in_ids = g.in_edges_by_col[j].get(node, [])
                if in_ids:
                    s = 0.0
                    for eid in in_ids:
                        if 0 <= eid < n_edges:
                            s += float(edge_h_frac[eid])
                    if s > 1e-12:
                        f = h_node / s
                        for eid in in_ids:
                            if 0 <= eid < n_edges:
                                edge_h_frac_in[eid] = float(edge_h_frac[eid]) * f

                out_ids = g.out_edges_by_col[j].get(node, [])
                if out_ids:
                    s = 0.0
                    for eid in out_ids:
                        if 0 <= eid < n_edges:
                            s += float(edge_h_frac[eid])
                    if s > 1e-12:
                        f = h_node / s
                        for eid in out_ids:
                            if 0 <= eid < n_edges:
                                edge_h_frac_out[eid] = float(edge_h_frac[eid]) * f

    # -------- Assign bands within nodes for each edge --------
    in_band_y: List[Dict[int, float]] = [{} for _ in range(cfg.n_cols)]
    out_band_y: List[Dict[int, float]] = [{} for _ in range(cfg.n_cols)]

    n_edges = len(g.edge_w_px)
    for j in range(cfg.n_cols):
        for node in g.col_nodes[j]:
            i_node = g.col_idx[j][node]
            cy = col_centers[j][i_node]
            h = col_h_frac[j][i_node]

            in_ids = g.in_edges_by_col[j].get(node, [])
            if in_ids:
                in_band_y[j].update(assign_bands_centered(cy, h, in_ids, edge_h_frac_in))

            out_ids = g.out_edges_by_col[j].get(node, [])
            if out_ids:
                out_band_y[j].update(assign_bands_centered(cy, h, out_ids, edge_h_frac_out))

    # -------- X positions / default link gap --------
    x_pos = resolve_x_positions(col_cfg_dict, cfg.n_cols, cfg.x_min, cfg.x_max)
    gapx0 = px_to_frac_x(cfg.link_node_gap_px, cfg.fig_width_in, cfg.dpi)

    # -------- Label density detection --------
    label_offset_frac = px_to_frac_y(cfg.label_offset_px, cfg.fig_height_in, cfg.dpi)
    fig_h_px = cfg.fig_height_in * cfg.dpi
    accepted_labels_per_col = [set() for _ in range(cfg.n_cols)]

    dx_frac_per_col: List[float] = []
    dy_frac_per_col: List[float] = []
    raw_label_density_cols = getattr(cfg, "label_density_cols", tuple()) or tuple()
    label_density_cols = set()
    for _v in raw_label_density_cols:
        try:
            _j = int(_v)
            if 0 <= _j < cfg.n_cols:
                label_density_cols.add(_j)
        except Exception:
            continue

    for j in range(cfg.n_cols):
        cdef = col_label_cfg.get(j, {})
        dx_px = float(cdef.get("dx_px", 0.0) or 0.0)
        dy_px = float(cdef.get("dy_px", 0.0) or 0.0)
        dx_frac_per_col.append(px_to_frac_x(dx_px, cfg.fig_width_in, cfg.dpi))
        dy_frac_per_col.append(px_to_frac_y(dy_px, cfg.fig_height_in, cfg.dpi))

        align = "center"
        if j == 0:
            align = "left"
        elif j == cfg.n_cols - 1:
            align = "right"

        auto_label_pos = "inside"
        if (align == "center") and cfg.label_below_middle:
            auto_label_pos = "below"

        col_show = bool(cdef.get("show", True))
        col_pos = str(cdef.get("pos", "auto"))
        col_fs = cdef.get("font_size", None)
        if isinstance(col_fs, str) and col_fs.lower() == "nan":
            col_fs = None

        col_show_label_global = cfg.show_labels and col_show and (col_pos.lower() != "none")

        candidates = []
        font_pt = float(col_fs) if col_fs is not None else float(cfg.text_font_size)
        font_px = font_pt * (cfg.dpi / 72.0)

        if cfg.label_min_vsep_px is not None and cfg.label_min_vsep_px > 0:
            min_sep_px = float(cfg.label_min_vsep_px)
        else:
            min_sep_px = font_px * 0.9

        for orig_idx, (node, h_frac, cy) in enumerate(zip(g.col_nodes[j], col_h_frac[j], col_centers[j])):
            if node in getattr(g, 'placeholder_nodes_set', set()):
                continue
            node_h_px = g.col_h_px[j].get(node, 0.0)

            pos = col_pos.lower().strip()
            if pos == "auto":
                if align in ("left", "right"):
                    label_y_frac = cy
                else:
                    if auto_label_pos == "below":
                        label_y_frac = cy - h_frac / 2 - label_offset_frac
                    else:
                        label_y_frac = cy
            elif pos == "inside":
                label_y_frac = cy
            elif pos == "below":
                label_y_frac = cy - h_frac / 2 - label_offset_frac
            else:
                label_y_frac = cy

            label_y_px = label_y_frac * fig_h_px
            priority = node_h_px if str(cfg.label_density_priority).lower() in ("height", "weight") else abs(label_y_px - (fig_h_px / 2))
            candidates.append({"name": node, "orig_idx": orig_idx, "y_px": label_y_px, "priority": priority})

        density_on_this_col = (not label_density_cols) or (j in label_density_cols)
        if cfg.label_enable_density_detection and col_show_label_global and density_on_this_col:
            accepted = _select_labels_by_density(candidates, min_sep_px)
        else:
            accepted = set([c["name"] for c in candidates])

        accepted_labels_per_col[j] = accepted

    return LayoutData(
        col_cfg_dict=col_cfg_dict,
        node_w_px_col=node_w_px_col,
        gap_px_col=gap_px_col,
        node_w_frac_col=node_w_frac_col,
        col_label_cfg=col_label_cfg,
        col_h_frac=col_h_frac,
        edge_h_frac=edge_h_frac,
        edge_h_frac_in=edge_h_frac_in,
        edge_h_frac_out=edge_h_frac_out,
        gap_frac=gap_frac,
        frame_top=frame_top,
        col_centers=col_centers,
        in_band_y=in_band_y,
        out_band_y=out_band_y,
        x_pos=x_pos,
        gapx0=gapx0,
        accepted_labels_per_col=accepted_labels_per_col,
        label_offset_frac=label_offset_frac,
        dx_frac_per_col=dx_frac_per_col,
        dy_frac_per_col=dy_frac_per_col,
        scale_applied=scale_applied,
    )
