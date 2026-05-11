
"""
graph.py
---------
Build the Sankey "graph" (nodes, edges, per-node heights) from the input DataFrame
using the existing rules:
- Each column has (node_name, node_color) pairs.
- Weights are per-stage (n_cols-1).
- Supports "skip blank" links without inserting dummy nodes.
- Optional: auto_balance_flow.
- Optional: no_merge_cols (do not merge same-name nodes in certain columns).
- Optional: last column node height override.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from bisect import bisect_left
import pandas as pd

from .colors import auto_fill_missing_node_colors_per_column


# --------------------- Config ---------------------
@dataclass
class SankeyConfig:
    # required-ish
    n_cols: int = 4

    # figure
    fig_width_in: float = 90
    fig_height_in: float = 60
    dpi: int = 300

    # mapping
    value_to_px: float = 1.0
    use_min_link_thickness: bool = True
    min_link_px: float = 5.0

    # mapping / geometry
    min_node_h_px: float = 1.0

    # 节点形状: "rect" | "circle"
    node_shape: str = "rect"

    # 方向: "horizontal"(默认) | "vertical"(上下流向)
    orientation: str = "horizontal"

    # layout
    force_align_top_bottom: bool = False
    # Columns exempted from force_align_top_bottom; these columns use per-column align/gap.
    force_align_exempt_cols: Tuple[int, ...] = tuple()
    stack_mode: str = "center"  # top / center / bottom
    x_min: float = 0.10
    x_max: float = 0.90
    link_node_gap_px: float = 40

    # 布局基准列索引
    layout_ref_col_index: Optional[int] = None

    # 分组间隙
    enable_group_gap: bool = False

    # 空白列占位：把跨列连线拆成相邻连线，并在中间列插入不可见占位节点（用于保留空白列的分隔感）
    enable_node_placeholders: bool = False
    # P值显示
    show_p_value: bool = False
    p_value_font_size: float = 20.0
    p_value_threshold: float = 0.05


    # ---------------- Link percentage labels (per-segment share) ----------------
    # If enabled, each link will show its share within the SAME segment (same src_col -> tgt_col).
    # Example: for all links from col0 -> col1, each link shows w / sum(w in that segment).
    show_link_pct: bool = False

    # Where to place the percentage text:
    # - "source_right": near the source node (right side in horizontal flow)
    # - "middle": middle of the link
    # - "target_left": near the target node (left side in horizontal flow)
    link_pct_position: str = "middle"

    # Which weight to use for percentage computation:
    # - "raw": use edge_w_raw (data weight)
    # - "px":  use edge_w_px  (after value_to_px and min_link_px)
    link_pct_basis: str = "raw"

    # Python format string; available keys:
    # - pct:    0..1
    # - pct100: 0..100
    # - w:      weight used for this link
    # - total:  total weight in this segment
    link_pct_format: str = "{pct100:.1f}%"

    # Text style
    link_pct_font_size: float = 16.0
    link_pct_color: str = "#000000"
    link_pct_bold: bool = False
    link_pct_italic: bool = False

    # Fine offsets (pixels; +x right, +y up)
    link_pct_dx_px: float = 0.0
    link_pct_dy_px: float = 0.0

    # When enable_node_placeholders=True, internal placeholder→placeholder segments can be noisy.
    link_pct_skip_internal_placeholder: bool = True

    # Aggregate identical src->tgt pairs (same segment, same src_node, same tgt_node) into ONE label
    link_pct_aggregate_same_pair: bool = True

    # Auto-hide dense/overlapping percentage labels (greedy by weight) within each segment
    link_pct_enable_density_detection: bool = True

    # Optional minimum separation between percentage labels (px). If None, auto from font size.
    link_pct_min_sep_px: Optional[float] = None


    # alpha / curve
    link_alpha: float = 0.55
    # Optional side outlines for links (only the two thickness-side curves; no end-cap strokes).
    enable_link_side_outline: bool = False
    link_side_outline_color: str = "#000000"
    link_side_outline_alpha: float = 0.35
    link_side_outline_width_px: float = 1.0
    # Optional outline for node shapes.
    enable_node_outline: bool = False
    node_outline_color: str = "#000000"
    node_outline_alpha: float = 0.35
    node_outline_width_px: float = 1.0
    node_alpha: float = 0.7
    curve_ctrl_rel: float = 0.28

    # font
    font_priority: Tuple[str, ...] = ("Arial",)
    # Optional mixed-script priorities. If empty, fallback to font_priority.
    font_zh_priority: Tuple[str, ...] = tuple()
    font_en_priority: Tuple[str, ...] = tuple()
    text_font_size: float = 80

    # canvas y bounds
    y_min: float = 0.03
    y_max: float = 0.97

    # labels default behavior
    label_below_middle: bool = True
    label_offset_px: float = 20
    # Faux-bold enhancement for fonts without true bold weight (e.g., SimSun).
    enable_faux_bold: bool = True
    faux_bold_width_px: float = 0.6
    # Per-column alternating label side (right-left-right-...)
    enable_alternate_label_sides: bool = False
    alternate_label_side_cols: Tuple[int, ...] = tuple()
    # Per-column upright vertical node labels.
    enable_vertical_node_labels: bool = False
    vertical_node_label_cols: Tuple[int, ...] = tuple()

    # CJK auto-wrap (display text level)
    enable_cjk_auto_wrap: bool = False
    cjk_wrap_chars_per_line: int = 8
    # supported targets: "node_label", "legend_label"
    wrap_targets: Tuple[str, ...] = ("node_label",)
    wrap_line_spacing_mult: float = 1.20
    # None/0 means unlimited
    wrap_max_lines: Optional[int] = None

    # density detection for labels
    label_enable_density_detection: bool = True
    label_min_vsep_px: Optional[float] = None
    label_density_priority: str = "height"
    # Empty means all columns keep the legacy density-detection behavior.
    label_density_cols: Tuple[int, ...] = tuple()

    # labels global switch + column grouped config
    show_labels: bool = True
    label_text_color_default: str = "#000000"
    # first-row headers
    enable_header_row: bool = False
    show_headers: bool = True
    header_pos: str = "top"  # top / bottom
    # signed px offset: negative up, positive down
    header_dy_px: float = 0.0
    # None means inherit from label defaults
    header_font_size: Optional[float] = None
    header_text_color: Optional[str] = None
    header_font_priority: Tuple[str, ...] = tuple()
    header_font_zh_priority: Tuple[str, ...] = tuple()
    header_font_en_priority: Tuple[str, ...] = tuple()
    header_bold: Optional[bool] = None
    header_italic: Optional[bool] = None
    header_underline: Optional[bool] = None

    # global plot title
    show_title: bool = False
    title_text: str = ""
    title_dx_px: float = 0.0
    title_dy_px: float = 0.0
    title_font_size: Optional[float] = None
    title_text_color: Optional[str] = None
    title_font_priority: Tuple[str, ...] = tuple()
    title_font_zh_priority: Tuple[str, ...] = tuple()
    title_font_en_priority: Tuple[str, ...] = tuple()
    title_bold: Optional[bool] = None
    title_italic: Optional[bool] = None
    title_underline: Optional[bool] = None

    # 长节点名省略 + 图例（A1/B2…） [方案四]
    # - enable_long_label_legend: 开关
    # - long_label_legend_threshold: 阈值（字符数，严格大于该值才触发）
    enable_long_label_legend: bool = False
    long_label_legend_threshold: int = 30
    legend_force_cols: Tuple[int, ...] = tuple()
    legend_include_auto_hidden: bool = False
    legend_position: str = "right"
    legend_dx_px: float = 0.0
    legend_dy_px: float = 0.0

    # 图例布局模式：
    # - "packed": 不分列，按宽度横向填充，放不下再换行
    # - "by_column_compact": 按列分区（仅对有长名的列分区），分区等宽，不留空白
    legend_layout_mode: str = "packed"

    # by_column_compact 的分区标题：
    # - "letter": A/B/C...
    # - "colnum": 第1列/第2列...
    # - "none": 不显示标题
    legend_column_title_mode: str = "letter"

    # 图例字号（独立于主图标签字号）
    legend_font_size: float = 16.0

    # 图上索引标注（A1/B2...）样式：用于与正常节点名（如 C32/E32）区分
    index_label_color: str = "#4A4A4A"
    index_label_font: str = ""      # 留空=跟随 font_priority
    index_label_bold: bool = False
    index_label_italic: bool = False
    col_label_cfg_rows: Optional[List[Dict[str, Any]]] = None

    # Link color mode: "source" | "target" | "gradient" | "follow_col"
    link_color_mode: str = "source"
    # When link_color_mode == "follow_col", use node color from this stage (0-based)
    link_color_follow_col: int = 0

    # Deprecated UI key (kept for config compatibility only).
    # Visual taper is now controlled by auto_balance_flow.
    enable_link_thickness_gradient: bool = False

    # New: per-stage fixed link color overrides (optional).
    # Each row example: {"stage_index": 0, "enable": True, "color": "#FF0000"}
    # stage_index is 0-based: 0 means between col0->col1; 1 means col1->col2; ...
    # For "skip columns" links (e.g., col1->col3), we attribute it to stage (target_col - 1).
    link_stage_override_rows: Optional[List[Dict[str, Any]]] = None

    # New semantics: visual taper only (do NOT rewrite edge weights or node heights).
    auto_balance_flow: bool = False

    # canvas auto-fit
    enable_auto_fit_canvas: bool = False
    # "manual" | "before_render"
    auto_fit_trigger_mode: str = "manual"
    auto_fit_max_iter: int = 8
    auto_fit_prefer_expand_canvas: bool = True
    auto_fit_consider_legend: bool = True
    auto_fit_consider_link_pct: bool = True

    # 新增：指定列不归一（不合并同名节点）
    no_merge_cols: Tuple[int, ...] = tuple()

    # 新增：末列节点高度可由额外权重列控制（在权重列/可选 p 值列后再追加 1 列）
    use_last_col_weight_override: bool = False

    # manual flip (legacy)
    flip_cols: Tuple[int, ...] = tuple()

    # node ordering optimization
    order_mode: str = "excel"
    order_target_stages: Tuple[int, ...] = tuple()
    order_keep_ratio: float = 0.35

    # per-column layout config
    col_cfg_rows: Optional[List[Dict[str, Any]]] = None
    default_node_width_px: float = 100
    default_gap_px: float = 50


# --------------------- Utils ---------------------
def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
        s = str(x).strip()
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def clamp_link_px(raw_px: float, use_min: bool, min_px: float) -> float:
    if raw_px <= 0:
        return 0.0
    if not use_min:
        return raw_px
    return max(raw_px, float(min_px))


def aggregate_duplicate_edges(
    *,
    cfg: SankeyConfig,
    rows_nodes: List[List[str]],
    edge_row: List[int],
    edge_src_col: List[int],
    edge_tgt_col: List[int],
    edge_src_node: List[str],
    edge_tgt_node: List[str],
    edge_w_raw: List[float],
    edge_base_src_col: List[int],
    edge_base_tgt_col: List[int],
    edge_base_src_node: List[str],
    edge_base_tgt_node: List[str],
    edge_group_src_node: List[str],
) -> Tuple[
    List[int], List[int], List[int], List[str], List[str], List[float], List[float],
    List[int], List[int], List[str], List[str], List[str]
]:
    """
    Merge duplicate edges and recompute edge_w_px from merged raw weights.
    Default key: (src_col, tgt_col, src_node, tgt_node).
    In follow_col mode, key is refined by follow identity from the selected column:
    (src_col, tgt_col, src_node, tgt_node, follow_identity).
    """
    n = len(edge_w_raw)
    if n <= 1:
        edge_w_px = [
            clamp_link_px(max(float(w) * float(cfg.value_to_px), 0.0), cfg.use_min_link_thickness, cfg.min_link_px)
            for w in edge_w_raw
        ]
        return (
            edge_row, edge_src_col, edge_tgt_col, edge_src_node, edge_tgt_node, edge_w_raw, edge_w_px,
            edge_base_src_col, edge_base_tgt_col, edge_base_src_node, edge_base_tgt_node, edge_group_src_node,
        )

    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    order: List[Tuple[Any, ...]] = []
    follow_mode = str(getattr(cfg, "link_color_mode", "source") or "source").lower() in (
        "follow_col", "follow_column", "followcol", "follow"
    )
    follow_col = int(getattr(cfg, "link_color_follow_col", 0) or 0)
    follow_col = max(0, min(int(cfg.n_cols) - 1, follow_col))

    for i in range(n):
        base_key = (
            int(edge_src_col[i]),
            int(edge_tgt_col[i]),
            str(edge_src_node[i]),
            str(edge_tgt_node[i]),
        )
        key = base_key
        if follow_mode:
            follow_identity = ""
            r_i = int(edge_row[i]) if i < len(edge_row) else -1
            if 0 <= r_i < len(rows_nodes):
                rn = rows_nodes[r_i]
                if isinstance(rn, list) and 0 <= follow_col < len(rn):
                    follow_identity = str(rn[follow_col] or "").strip()
            # Fallback to legacy key when identity is missing/empty
            if follow_identity:
                key = base_key + (follow_identity,)

        rec = grouped.get(key)
        if rec is None:
            grouped[key] = {
                "edge_row": int(edge_row[i]),
                "edge_src_col": int(edge_src_col[i]),
                "edge_tgt_col": int(edge_tgt_col[i]),
                "edge_src_node": str(edge_src_node[i]),
                "edge_tgt_node": str(edge_tgt_node[i]),
                "edge_w_raw": float(edge_w_raw[i]),
                "edge_base_src_col": int(edge_base_src_col[i]),
                "edge_base_tgt_col": int(edge_base_tgt_col[i]),
                "edge_base_src_node": str(edge_base_src_node[i]),
                "edge_base_tgt_node": str(edge_base_tgt_node[i]),
                "edge_group_src_node": str(edge_group_src_node[i]),
            }
            order.append(key)
        else:
            rec["edge_w_raw"] += float(edge_w_raw[i])

    out_edge_row: List[int] = []
    out_edge_src_col: List[int] = []
    out_edge_tgt_col: List[int] = []
    out_edge_src_node: List[str] = []
    out_edge_tgt_node: List[str] = []
    out_edge_w_raw: List[float] = []
    out_edge_w_px: List[float] = []
    out_edge_base_src_col: List[int] = []
    out_edge_base_tgt_col: List[int] = []
    out_edge_base_src_node: List[str] = []
    out_edge_base_tgt_node: List[str] = []
    out_edge_group_src_node: List[str] = []

    for key in order:
        rec = grouped[key]
        w_raw = float(rec["edge_w_raw"])
        w_px = clamp_link_px(
            max(w_raw * float(cfg.value_to_px), 0.0),
            cfg.use_min_link_thickness,
            cfg.min_link_px,
        )
        if w_px <= 0:
            continue

        out_edge_row.append(int(rec["edge_row"]))
        out_edge_src_col.append(int(rec["edge_src_col"]))
        out_edge_tgt_col.append(int(rec["edge_tgt_col"]))
        out_edge_src_node.append(str(rec["edge_src_node"]))
        out_edge_tgt_node.append(str(rec["edge_tgt_node"]))
        out_edge_w_raw.append(w_raw)
        out_edge_w_px.append(float(w_px))
        out_edge_base_src_col.append(int(rec["edge_base_src_col"]))
        out_edge_base_tgt_col.append(int(rec["edge_base_tgt_col"]))
        out_edge_base_src_node.append(str(rec["edge_base_src_node"]))
        out_edge_base_tgt_node.append(str(rec["edge_base_tgt_node"]))
        out_edge_group_src_node.append(str(rec["edge_group_src_node"]))

    return (
        out_edge_row, out_edge_src_col, out_edge_tgt_col, out_edge_src_node, out_edge_tgt_node, out_edge_w_raw, out_edge_w_px,
        out_edge_base_src_col, out_edge_base_tgt_col, out_edge_base_src_node, out_edge_base_tgt_node, out_edge_group_src_node,
    )


# --------------------- Node order optimization ---------------------
def apply_node_order_optimization(col_nodes, rows_nodes, weights_px, mode, target_stages, keep_ratio):
    """
    Optimize node order to reduce or increase crossings on selected adjacent stages.
    - mode: "min_cross" | "max_cross" | "excel"
    - target_stages: iterable of stage index s meaning col_s -> col_{s+1}
    - keep_ratio: blend with original excel order in [0,1]
    """
    mode_s = str(mode or "excel").lower().strip()
    if mode_s not in ("min_cross", "max_cross"):
        return col_nodes

    n_cols = len(col_nodes or [])
    if n_cols <= 1:
        return col_nodes

    # rows_weights_raw is passed through legacy arg `weights_px` for compatibility.
    rows_weights_raw = weights_px if isinstance(weights_px, list) else []
    if (not isinstance(rows_nodes, list)) or (not isinstance(rows_weights_raw, list)):
        return col_nodes

    # sanitize stages
    all_stages = list(range(max(0, n_cols - 1)))
    if target_stages:
        st = []
        for x in target_stages:
            try:
                xi = int(x)
                if 0 <= xi < n_cols - 1:
                    st.append(xi)
            except Exception:
                continue
        stages = sorted(set(st)) if st else all_stages
    else:
        stages = all_stages
    if not stages:
        return col_nodes

    try:
        kr = float(keep_ratio)
    except Exception:
        kr = 0.35
    kr = max(0.0, min(1.0, kr))

    # working orders + excel baseline
    orders = [list(col_nodes[j]) for j in range(n_cols)]
    excel_rank = [{n: i for i, n in enumerate(orders[j])} for j in range(n_cols)]

    # Precompute stage edges: (src_node, tgt_node, weight)
    stage_edges: Dict[int, List[Tuple[str, str, float]]] = {s: [] for s in stages}
    for r_i, ns in enumerate(rows_nodes):
        if not isinstance(ns, list):
            continue
        w_row = rows_weights_raw[r_i] if r_i < len(rows_weights_raw) and isinstance(rows_weights_raw[r_i], list) else []
        for s in stages:
            if s >= len(ns) - 1:
                continue
            src = ns[s]
            tgt = ns[s + 1]
            if (not src) or (not tgt):
                continue
            w = 0.0
            if s < len(w_row):
                try:
                    w = float(w_row[s])
                except Exception:
                    w = 0.0
            if w <= 0:
                continue
            stage_edges[s].append((str(src), str(tgt), float(w)))

    def _rank_map(col_idx: int) -> Dict[str, int]:
        return {n: i for i, n in enumerate(orders[col_idx])}

    def _cross_score_for_stage(stage_idx: int) -> float:
        edges = stage_edges.get(stage_idx, [])
        if len(edges) <= 1:
            return 0.0
        rs = _rank_map(stage_idx)
        rt = _rank_map(stage_idx + 1)
        packed = []
        for s_node, t_node, w in edges:
            if (s_node in rs) and (t_node in rt):
                packed.append((rs[s_node], rt[t_node], float(w)))
        m = len(packed)
        if m <= 1:
            return 0.0
        score = 0.0
        for i in range(m):
            si, ti, wi = packed[i]
            for j in range(i + 1, m):
                sj, tj, wj = packed[j]
                if (si - sj) * (ti - tj) < 0:
                    score += wi * wj
        return float(score)

    def _reorder_by_barycenter(stage_idx: int, move_target: bool):
        edges = stage_edges.get(stage_idx, [])
        if not edges:
            return
        fixed_col = stage_idx if move_target else (stage_idx + 1)
        move_col = stage_idx + 1 if move_target else stage_idx
        fixed_rank = _rank_map(fixed_col)

        acc_w: Dict[str, float] = defaultdict(float)
        acc_rw: Dict[str, float] = defaultdict(float)
        for s_node, t_node, w in edges:
            moving = t_node if move_target else s_node
            fixed = s_node if move_target else t_node
            if fixed not in fixed_rank:
                continue
            rr = float(fixed_rank[fixed])
            ww = float(w)
            acc_w[moving] += ww
            acc_rw[moving] += rr * ww

        excel_r = excel_rank[move_col]
        cur = orders[move_col]

        # score: barycenter (or inverse barycenter for max_cross), blended with excel rank
        scored = []
        for n in cur:
            if acc_w.get(n, 0.0) > 0:
                bary = acc_rw[n] / acc_w[n]
            else:
                bary = float(excel_r.get(n, 10 ** 9))
            opt = bary if mode_s == "min_cross" else (-bary)
            final = (1.0 - kr) * opt + kr * float(excel_r.get(n, 10 ** 9))
            scored.append((n, final, float(excel_r.get(n, 10 ** 9))))
        scored.sort(key=lambda x: (x[1], x[2], x[0]))
        orders[move_col] = [x[0] for x in scored]

    def _adjacent_swap_refine(stage_idx: int, move_target: bool, max_pass: int = 2):
        move_col = stage_idx + 1 if move_target else stage_idx
        if len(orders[move_col]) <= 1:
            return
        eps = 1e-12
        for _ in range(max(1, int(max_pass))):
            improved = False
            i = 0
            while i < len(orders[move_col]) - 1:
                base = _cross_score_for_stage(stage_idx)
                arr = orders[move_col]
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                now = _cross_score_for_stage(stage_idx)
                better = (now < base - eps) if mode_s == "min_cross" else (now > base + eps)
                if better:
                    improved = True
                    i += 2
                else:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    i += 1
            if not improved:
                break

    # small iterative sweep to make ordering more natural and stable
    for _ in range(3):
        for s in stages:
            _reorder_by_barycenter(s, move_target=True)
            _adjacent_swap_refine(s, move_target=True, max_pass=2)
        for s in reversed(stages):
            _reorder_by_barycenter(s, move_target=False)
            _adjacent_swap_refine(s, move_target=False, max_pass=2)

    return orders


# --------------------- Graph data container ---------------------
@dataclass
class GraphData:
    # parsed rows
    rows_nodes: List[List[str]]
    rows_colors: List[List[str]]
    rows_weights_raw: List[List[float]]      # per row, per stage
    rows_pvals: List[List[str]]              # per row, per stage (strings)
    rows_last_node_w_raw: List[float]

    # columns
    col_nodes: List[List[str]]
    col_colors: List[Dict[str, str]]
    display_name_by_col: List[Dict[str, str]]
    col_headers: List[Dict[str, str]]  # per col: {"name": str, "color": str}
    no_merge_cols_set: set

    # edges
    edge_row: List[int]
    edge_src_col: List[int]
    edge_tgt_col: List[int]
    edge_src_node: List[str]
    edge_tgt_node: List[str]
    edge_w_raw: List[float]
    edge_w_px: List[float]

    # placeholders (optional)
    placeholder_nodes_set: set
    edge_base_src_col: List[int]
    edge_base_tgt_col: List[int]
    edge_base_src_node: List[str]
    edge_base_tgt_node: List[str]
    edge_group_src_node: List[str]
    skip_spans_count: int


    # derived maps
    col_idx: List[Dict[str, int]]
    out_edges_by_col: List[Dict[str, List[int]]]
    in_edges_by_col: List[Dict[str, List[int]]]

    # node heights (px)
    col_h_px: List[Dict[str, float]]

    # optional last-node override
    last_node_weight_col_idx: Optional[int]
    last_node_override_px: Dict[str, float]

    # diagnostics
    balance_diag: Dict[str, Any]


def _parse_pvalues_for_row(
    r: pd.Series,
    cfg: SankeyConfig,
    stage_count: int,
) -> List[str]:
    out: List[str] = []
    for stage in range(stage_count):
        pval_str = ""
        if cfg.show_p_value:
            p_idx = (2 * cfg.n_cols) + (cfg.n_cols - 1) + stage
            if p_idx in r.index:
                raw_p = r[p_idx]
                if pd.notna(raw_p) and raw_p != "":
                    try:
                        p_float = float(raw_p)
                        if p_float < 0.001:
                            pval_str = "<0.001"
                        else:
                            pval_str = f"{p_float:.3f}"
                    except Exception:
                        pval_str = str(raw_p)
        out.append(pval_str)
    return out


def _infer_last_node_weight_col_idx(df: pd.DataFrame, cfg: SankeyConfig) -> Optional[int]:
    if not bool(getattr(cfg, "use_last_col_weight_override", False)):
        return None

    base_idx = (2 * cfg.n_cols) + max(0, (cfg.n_cols - 1))
    if bool(getattr(cfg, "show_p_value", False)):
        base_idx += max(0, (cfg.n_cols - 1))
    if base_idx in df.columns:
        return int(base_idx)
    return None


def build_graph(df: pd.DataFrame, cfg: SankeyConfig) -> GraphData:
    """
    Parse df -> nodes + skip-blank edges + per-node heights (px), and build maps.
    This function mirrors the original sankey_engine.py behavior.
    """
    if cfg.n_cols < 1:
        raise ValueError("n_cols must be >= 1")

    df = df.dropna(how="all")
    if df.shape[0] == 0:
        raise ValueError("Excel is empty")

    # Ensure required columns exist (nodes/colors + weights)
    need_cols = (2 * cfg.n_cols) + max(0, (cfg.n_cols - 1))
    for c in range(need_cols):
        if c not in df.columns:
            df[c] = ""

    rows_nodes: List[List[str]] = []
    rows_colors: List[List[str]] = []
    rows_weights_raw: List[List[float]] = []
    rows_pvals: List[List[str]] = []
    rows_last_node_w_raw: List[float] = []

    col_nodes: List[List[str]] = [[] for _ in range(cfg.n_cols)]
    col_colors: List[Dict[str, str]] = [{} for _ in range(cfg.n_cols)]
    display_name_by_col: List[Dict[str, str]] = [dict() for _ in range(cfg.n_cols)]
    col_headers: List[Dict[str, str]] = [{"name": "", "color": ""} for _ in range(cfg.n_cols)]
    node_first_row_by_col: List[Dict[str, int]] = [dict() for _ in range(cfg.n_cols)]

    # no_merge columns (row-unique internal node ids)
    no_merge_cols_set = set()
    for x in (cfg.no_merge_cols or ()):
        try:
            xi = int(x)
            if 0 <= xi < cfg.n_cols:
                no_merge_cols_set.add(xi)
        except Exception:
            pass

    # Optional first-row headers: read metadata from row0, then exclude row0 from data parsing.
    df_data = df
    if bool(getattr(cfg, "enable_header_row", False)) and int(df.shape[0]) > 0:
        r0 = df.iloc[0]
        for j in range(cfg.n_cols):
            name_idx = 2 * j
            color_idx = 2 * j + 1
            hname = safe_str(r0[name_idx]) if name_idx in r0.index else ""
            hcolor = safe_str(r0[color_idx]) if color_idx in r0.index else ""
            col_headers[j] = {"name": hname, "color": hcolor}
        df_data = df.iloc[1:].copy()
        if df_data.shape[0] == 0:
            raise ValueError("首行作为表头后，已无可用数据行。")

    last_node_weight_col_idx = _infer_last_node_weight_col_idx(df_data, cfg)

    # -------- Parse rows --------
    for row_i, (_, r) in enumerate(df_data.iterrows()):
        ns: List[str] = []
        cs: List[str] = []
        for j in range(cfg.n_cols):
            name_raw = safe_str(r[2 * j])
            colr = safe_str(r[2 * j + 1])

            if (j in no_merge_cols_set) and name_raw and str(name_raw).lower() != "nan":
                name = f"{name_raw}__r{row_i}"
            else:
                name = name_raw

            ns.append(name)
            cs.append(colr)

            if name and str(name).lower() != "nan":
                display_name_by_col[j][name] = name_raw
                if name not in node_first_row_by_col[j]:
                    node_first_row_by_col[j][name] = row_i

        rows_nodes.append(ns)
        rows_colors.append(cs)

        # weights per stage (raw)
        w_raw: List[float] = []
        for stage in range(cfg.n_cols - 1):
            val = safe_float(r[2 * cfg.n_cols + stage])
            w_raw.append(val)
        rows_weights_raw.append(w_raw)

        # p-values per stage (optional)
        rows_pvals.append(_parse_pvalues_for_row(r, cfg, cfg.n_cols - 1))

        # last-node override per row (optional)
        if last_node_weight_col_idx is not None and last_node_weight_col_idx in df.columns:
            rows_last_node_w_raw.append(safe_float(r[last_node_weight_col_idx]))
        else:
            rows_last_node_w_raw.append(0.0)

        # build col_nodes / col_colors
        for j in range(cfg.n_cols):
            name = ns[j]
            if name and str(name).lower() != "nan":
                if j in no_merge_cols_set:
                    col_nodes[j].append(name)
                else:
                    if name not in col_nodes[j]:
                        col_nodes[j].append(name)
                if name not in col_colors[j]:
                    col_colors[j][name] = cs[j] if cs[j] else ""
                else:
                    if (not col_colors[j][name]) and cs[j]:
                        col_colors[j][name] = cs[j]

    n_rows = len(rows_nodes)

        # -------- Build edges (optionally insert invisible placeholders for blank intermediate columns) --------
    edge_row: List[int] = []
    edge_src_col: List[int] = []
    edge_tgt_col: List[int] = []
    edge_src_node: List[str] = []
    edge_tgt_node: List[str] = []
    edge_w_raw: List[float] = []
    edge_w_px: List[float] = []

    # Base endpoints per edge (so placeholder segments can inherit color / p-value behavior)
    edge_base_src_col: List[int] = []
    edge_base_tgt_col: List[int] = []
    edge_base_src_node: List[str] = []
    edge_base_tgt_node: List[str] = []

    # For grouping gaps: keep "major source" semantics even when an edge is split into segments
    edge_group_src_node: List[str] = []

    # Invisible placeholder nodes used to route links through empty intermediate columns.
    # They are NOT rendered, but they DO participate in layout.
    placeholder_nodes_set: set = set()
    # For each column k, store (row_key, src_col, src_node, pid) so we can insert pid into col_nodes[k] later
    placeholder_reqs_by_col: List[List[Tuple[int, int, str, str]]] = [[] for _ in range(cfg.n_cols)]

    # Diagnostics: number of original "skipping" spans (before splitting into segments)
    skip_spans_count: int = 0

    enable_ph = bool(getattr(cfg, "enable_node_placeholders", False))

    for r_i in range(n_rows):
        ns = rows_nodes[r_i]
        # consecutive non-empty nodes define spans
        valid_idx = [k for k, n in enumerate(ns) if n and str(n).lower() != "nan"]
        if len(valid_idx) < 2:
            continue

        w_stage = rows_weights_raw[r_i] if r_i < len(rows_weights_raw) else []
        pos_vals = [v for v in w_stage if v > 0]
        fill_val = max(pos_vals) if pos_vals else 0.0

        for a, b in zip(valid_idx, valid_idx[1:]):
            if a >= cfg.n_cols - 1:
                continue

            # Choose a representative weight from the spanned stage weights; fallback to row max.
            seg_vals = w_stage[a:b] if (a < len(w_stage)) else []
            seg_max = max(seg_vals) if seg_vals else 0.0
            w = float(seg_max if seg_max > 0 else fill_val)

            raw_px = max(w * cfg.value_to_px, 0.0)
            wpx = clamp_link_px(raw_px, cfg.use_min_link_thickness, cfg.min_link_px)
            if wpx <= 0:
                continue

            base_src_col = int(a)
            base_tgt_col = int(b)
            base_src_node = ns[a]
            base_tgt_node = ns[b]
            base_gap = base_tgt_col - base_src_col

            if (not enable_ph) or (base_gap <= 1):
                # Keep original behavior: a single edge can skip blank columns
                edge_row.append(r_i)
                edge_src_col.append(base_src_col)
                edge_tgt_col.append(base_tgt_col)
                edge_src_node.append(base_src_node)
                edge_tgt_node.append(base_tgt_node)
                edge_w_raw.append(w)
                edge_w_px.append(wpx)

                edge_base_src_col.append(base_src_col)
                edge_base_tgt_col.append(base_tgt_col)
                edge_base_src_node.append(base_src_node)
                edge_base_tgt_node.append(base_tgt_node)

                edge_group_src_node.append(base_src_node)
            else:
                # Split into adjacent segments via invisible placeholders (one per intermediate column)
                skip_spans_count += 1

                prev_col = base_src_col
                prev_node = base_src_node

                for k in range(base_src_col + 1, base_tgt_col):
                    pid = f"__BLANK__r{r_i}_c{k}_a{base_src_col}_b{base_tgt_col}"
                    placeholder_nodes_set.add(pid)
                    placeholder_reqs_by_col[k].append((r_i, base_src_col, base_src_node, pid))

                    # Keep dict keys consistent (labels won't be shown; color is irrelevant for rendering)
                    display_name_by_col[k][pid] = ""
                    if pid not in col_colors[k]:
                        col_colors[k][pid] = "#FFFFFF"

                    # Segment: prev -> placeholder
                    edge_row.append(r_i)
                    edge_src_col.append(prev_col)
                    edge_tgt_col.append(k)
                    edge_src_node.append(prev_node)
                    edge_tgt_node.append(pid)
                    edge_w_raw.append(w)
                    edge_w_px.append(wpx)

                    edge_base_src_col.append(base_src_col)
                    edge_base_tgt_col.append(base_tgt_col)
                    edge_base_src_node.append(base_src_node)
                    edge_base_tgt_node.append(base_tgt_node)

                    edge_group_src_node.append(base_src_node)

                    prev_col = k
                    prev_node = pid

                # Last segment: last placeholder -> real target
                edge_row.append(r_i)
                edge_src_col.append(prev_col)
                edge_tgt_col.append(base_tgt_col)
                edge_src_node.append(prev_node)
                edge_tgt_node.append(base_tgt_node)
                edge_w_raw.append(w)
                edge_w_px.append(wpx)

                edge_base_src_col.append(base_src_col)
                edge_base_tgt_col.append(base_tgt_col)
                edge_base_src_node.append(base_src_node)
                edge_base_tgt_node.append(base_tgt_node)

                edge_group_src_node.append(base_src_node)

    (
        edge_row, edge_src_col, edge_tgt_col, edge_src_node, edge_tgt_node, edge_w_raw, edge_w_px,
        edge_base_src_col, edge_base_tgt_col, edge_base_src_node, edge_base_tgt_node, edge_group_src_node,
    ) = aggregate_duplicate_edges(
        cfg=cfg,
        rows_nodes=rows_nodes,
        edge_row=edge_row,
        edge_src_col=edge_src_col,
        edge_tgt_col=edge_tgt_col,
        edge_src_node=edge_src_node,
        edge_tgt_node=edge_tgt_node,
        edge_w_raw=edge_w_raw,
        edge_base_src_col=edge_base_src_col,
        edge_base_tgt_col=edge_base_tgt_col,
        edge_base_src_node=edge_base_src_node,
        edge_base_tgt_node=edge_base_tgt_node,
        edge_group_src_node=edge_group_src_node,
    )

    n_edges = len(edge_w_px)

    if (not enable_ph) and n_edges > 0:
        skip_spans_count = int(sum(1 for i in range(n_edges) if (edge_tgt_col[i] - edge_src_col[i]) > 1))

    # -------- Visual taper mode (legacy key: auto_balance_flow) --------
    # New semantics: this flag no longer mutates edge weights in graph-building.
    # Node heights remain derived from the original edge_w_px.
    balance_diag: Dict[str, Any] = {
        "enabled": bool(cfg.auto_balance_flow),
        "mode": "visual_taper_only",
    }

    # -------- Apply optional column flips / ordering (node order only) --------
    flip_cols = [j for j in cfg.flip_cols if isinstance(j, int) and 0 <= j < cfg.n_cols]
    for j in flip_cols:
        col_nodes[j] = list(reversed(col_nodes[j]))

    if cfg.order_mode and str(cfg.order_mode).lower().strip() != "excel":
        col_nodes = apply_node_order_optimization(
            col_nodes=col_nodes,
            rows_nodes=rows_nodes,
            weights_px=rows_weights_raw,  # rows stage weights
            mode=str(cfg.order_mode),
            target_stages=tuple(cfg.order_target_stages or ()),
            keep_ratio=float(cfg.order_keep_ratio),
        )

    # -------- Auto-fill node colors (if missing) --------
    auto_fill_missing_node_colors_per_column(col_nodes=col_nodes, col_colors=col_colors, n_cols=cfg.n_cols)
    # -------- Insert invisible placeholders into columns (for layout + routing) --------
    if enable_ph and placeholder_nodes_set:
        flip_set = set(flip_cols)

        try:
            use_excel_like = (str(cfg.order_mode).lower().strip() == "excel")
        except Exception:
            use_excel_like = True

        if use_excel_like:
            # Insert by Excel row order: key = row index where the skipping span appears.
            placeholders_by_col: List[List[Tuple[float, str]]] = [[] for _ in range(cfg.n_cols)]
            for k in range(1, cfg.n_cols - 1):
                for row_key, src_col, src_node, pid in placeholder_reqs_by_col[k]:
                    key = float(row_key)
                    if k in flip_set:
                        key = -key
                    placeholders_by_col[k].append((key, pid))

            for k in range(1, cfg.n_cols - 1):
                if not placeholders_by_col[k]:
                    continue
                placeholders_by_col[k].sort(key=lambda t: (t[0], t[1]))

                keys_now: List[float] = []
                for n in col_nodes[k]:
                    if n in placeholder_nodes_set:
                        continue
                    kk = float(node_first_row_by_col[k].get(n, 10 ** 9))
                    if k in flip_set:
                        kk = -kk
                    keys_now.append(kk)

                for key, pid in placeholders_by_col[k]:
                    ins = bisect_left(keys_now, key)
                    col_nodes[k].insert(ins, pid)
                    keys_now.insert(ins, key)

                    # Ensure dict keys exist
                    if pid not in col_colors[k]:
                        col_colors[k][pid] = "#FFFFFF"
                    display_name_by_col[k][pid] = ""
        else:
            # Non-Excel ordering: project placeholder position from its source node rank.
            base_rank = [{n: i for i, n in enumerate(col_nodes[j])} for j in range(cfg.n_cols)]
            base_len = [len(col_nodes[j]) for j in range(cfg.n_cols)]
            placeholders_by_col: List[List[Tuple[int, str]]] = [[] for _ in range(cfg.n_cols)]

            for k in range(1, cfg.n_cols - 1):
                for row_key, src_col, src_node, pid in placeholder_reqs_by_col[k]:
                    denom = max(1, base_len[src_col] - 1)
                    frac = float(base_rank[src_col].get(src_node, 0)) / float(denom) if denom > 0 else 0.0
                    frac = max(0.0, min(1.0, frac))

                    ins = int(round(frac * float(base_len[k])))
                    ins = max(0, min(base_len[k], ins))
                    placeholders_by_col[k].append((ins, pid))

            for k in range(1, cfg.n_cols - 1):
                if not placeholders_by_col[k]:
                    continue
                placeholders_by_col[k].sort(key=lambda t: (t[0], t[1]))
                offset = 0
                for ins, pid in placeholders_by_col[k]:
                    idx2 = ins + offset
                    col_nodes[k].insert(idx2, pid)
                    offset += 1

                    # Ensure dict keys exist
                    if pid not in col_colors[k]:
                        col_colors[k][pid] = "#FFFFFF"
                    display_name_by_col[k][pid] = ""

    # index maps for ordering
    col_idx = [{n: i for i, n in enumerate(col_nodes[j])} for j in range(cfg.n_cols)]

    # -------- Edge maps by node --------
    out_edges_by_col: List[Dict[str, List[int]]] = [defaultdict(list) for _ in range(cfg.n_cols)]
    in_edges_by_col: List[Dict[str, List[int]]] = [defaultdict(list) for _ in range(cfg.n_cols)]
    for eid in range(n_edges):
        s_col = edge_src_col[eid]
        t_col = edge_tgt_col[eid]
        s = edge_src_node[eid]
        t = edge_tgt_node[eid]
        if 0 <= s_col < cfg.n_cols:
            out_edges_by_col[s_col][s].append(eid)
        if 0 <= t_col < cfg.n_cols:
            in_edges_by_col[t_col][t].append(eid)

    # Sort edges within each node (stable / less crossing)
    for j in range(cfg.n_cols):
        for node, ids in out_edges_by_col[j].items():
            ids.sort(key=lambda eid: (
                edge_tgt_col[eid],
                col_idx[edge_tgt_col[eid]].get(edge_tgt_node[eid], 10 ** 9),
                eid,
            ))
        for node, ids in in_edges_by_col[j].items():
            ids.sort(key=lambda eid: (
                edge_src_col[eid],
                col_idx[edge_src_col[eid]].get(edge_src_node[eid], 10 ** 9),
                eid,
            ))

    # -------- Optional: last column node height override --------
    last_node_override_px: Dict[str, float] = defaultdict(float)
    if last_node_weight_col_idx is not None and cfg.n_cols >= 1:
        last_col = cfg.n_cols - 1
        for r_i in range(n_rows):
            node = rows_nodes[r_i][last_col] if last_col < len(rows_nodes[r_i]) else ""
            if not node:
                continue
            w_raw = rows_last_node_w_raw[r_i] if r_i < len(rows_last_node_w_raw) else 0.0
            if w_raw and float(w_raw) > 0:
                last_node_override_px[node] += max(float(w_raw) * float(cfg.value_to_px), 0.0)

    # -------- Node heights (px) --------
    col_h_px: List[Dict[str, float]] = [{} for _ in range(cfg.n_cols)]
    for j in range(cfg.n_cols):
        for node in col_nodes[j]:
            if cfg.n_cols == 1:
                col_h_px[j][node] = float(cfg.min_node_h_px)
                continue

            s_out = sum(float(edge_w_px[eid]) for eid in out_edges_by_col[j].get(node, []))
            s_in = sum(float(edge_w_px[eid]) for eid in in_edges_by_col[j].get(node, []))

            if j == 0:
                col_h_px[j][node] = max(s_out, float(cfg.min_node_h_px))
            elif j == cfg.n_cols - 1:
                base_h = max(s_in, float(cfg.min_node_h_px))
                if last_node_override_px:
                    ov = float(last_node_override_px.get(node, 0.0))
                    if ov > 0:
                        base_h = max(base_h, ov)
                col_h_px[j][node] = base_h
            else:
                col_h_px[j][node] = max(s_in, s_out, float(cfg.min_node_h_px))

    return GraphData(
        rows_nodes=rows_nodes,
        rows_colors=rows_colors,
        rows_weights_raw=rows_weights_raw,
        rows_pvals=rows_pvals,
        rows_last_node_w_raw=rows_last_node_w_raw,
        col_nodes=col_nodes,
        col_colors=col_colors,
        display_name_by_col=display_name_by_col,
        col_headers=col_headers,
        no_merge_cols_set=no_merge_cols_set,
        edge_row=edge_row,
        edge_src_col=edge_src_col,
        edge_tgt_col=edge_tgt_col,
        edge_src_node=edge_src_node,
        edge_tgt_node=edge_tgt_node,
        edge_w_raw=edge_w_raw,
        edge_w_px=edge_w_px,

placeholder_nodes_set=set(placeholder_nodes_set),
edge_base_src_col=edge_base_src_col,
edge_base_tgt_col=edge_base_tgt_col,
edge_base_src_node=edge_base_src_node,
edge_base_tgt_node=edge_base_tgt_node,
edge_group_src_node=edge_group_src_node,
skip_spans_count=int(skip_spans_count),
        col_idx=col_idx,
        out_edges_by_col=out_edges_by_col,
        in_edges_by_col=in_edges_by_col,
        col_h_px=col_h_px,
        last_node_weight_col_idx=last_node_weight_col_idx,
        last_node_override_px=dict(last_node_override_px),
        balance_diag=balance_diag,
    )
