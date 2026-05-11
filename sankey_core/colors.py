
"""
colors.py
---------
All color utilities and policies:
- Hex <-> RGB(A)
- Auto-fill node colors (diverging palette)
- Link color mode resolution: source/target/gradient/follow_col
- Per-stage fixed link color override (stage_index), including skip-column attribution:
    stage_index = (target_col - 1)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np



from matplotlib import font_manager
# --------------------- Auto colors (when node color code is missing) ---------------------
AUTO_BLUE_LIGHT_TO_DARK = [
    "#C6DBEF", "#9ECAE1", "#6BAED6",
    "#4292C6", "#2171B5", "#08519C", "#08306B",
]
AUTO_RED_LIGHT_TO_DARK = [
    "#FCBBA1", "#FC9272", "#FB6A4A",
    "#EF3B2C", "#CB181D", "#A50F15", "#67000D",
]


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = str(h).strip()
    if not h:
        return (0, 0, 0)
    if h.startswith("#"):
        h = h[1:]
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return (0, 0, 0)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    r = max(0, min(255, int(round(r))))
    g = max(0, min(255, int(round(g))))
    b = max(0, min(255, int(round(b))))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def _blend_hex(a: str, b: str, t: float) -> str:
    ra, ga, ba = _hex_to_rgb(a)
    rb, gb, bb = _hex_to_rgb(b)
    r = ra + (rb - ra) * t
    g = ga + (gb - ga) * t
    b_ = ba + (bb - ba) * t
    return _rgb_to_hex((r, g, b_))


AUTO_NEUTRAL = _blend_hex(AUTO_BLUE_LIGHT_TO_DARK[0], AUTO_RED_LIGHT_TO_DARK[0], 0.5)


def _interp_steps_hex(steps: List[str], t: float) -> str:
    """Interpolate along a discrete light->dark step list, returning a hex color."""
    if not steps:
        return "#999999"
    if len(steps) == 1:
        return steps[0]
    t = max(0.0, min(1.0, float(t)))
    pos = t * (len(steps) - 1)
    i0 = int(math.floor(pos))
    i1 = min(i0 + 1, len(steps) - 1)
    f = pos - i0
    return _blend_hex(steps[i0], steps[i1], f)


def _auto_diverging_color_by_rank(rank: int, n_nodes: int, col_idx: int, n_cols: int) -> str:
    # rank: 0 is the top node in this column
    if n_nodes <= 1:
        return AUTO_NEUTRAL

    mid = (n_nodes - 1) / 2.0
    if mid <= 1e-9:
        return AUTO_NEUTRAL

    # p > 0 => red (upper), p < 0 => blue (lower), p ~ 0 => neutral
    p = (mid - float(rank)) / mid
    m = abs(p)

    # Intensity range grows with how many nodes are in this column:
    base_max_t = min(1.0, mid / float(len(AUTO_RED_LIGHT_TO_DARK) - 1))

    # Column ramp: more columns -> later columns slightly heavier.
    if n_cols > 1:
        col_factor = 0.6 + 0.4 * (float(col_idx) / float(n_cols - 1))
    else:
        col_factor = 1.0

    max_t = min(1.0, base_max_t * col_factor)
    t = m * max_t

    if m < 1e-9:
        return AUTO_NEUTRAL
    return _interp_steps_hex(AUTO_RED_LIGHT_TO_DARK, t) if p > 0 else _interp_steps_hex(AUTO_BLUE_LIGHT_TO_DARK, t)


def auto_fill_missing_node_colors_per_column(col_nodes: List[List[str]], col_colors: List[Dict[str, str]], n_cols: int) -> None:
    """
    Fill missing node colors (empty string) using the diverging scheme per column.
    Keeps any explicit colors provided in the input as-is.
    """
    for j in range(n_cols):
        nodes = col_nodes[j] if j < len(col_nodes) else []
        if not nodes:
            continue
        for i, node in enumerate(nodes):
            if node not in col_colors[j]:
                col_colors[j][node] = ""
            if safe_str(col_colors[j].get(node, "")) == "":
                col_colors[j][node] = _auto_diverging_color_by_rank(i, len(nodes), j, n_cols)


def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return s.strip()


def normalize_hex_or_default(s: Any, default: str) -> str:
    if s is None:
        return default
    t = str(s).strip()
    if t == "" or t.lower() == "nan":
        return default
    return t


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    s = str(hex_color).strip()
    if s == "" or s.lower() == "nan":
        return (0.7, 0.7, 0.7, alpha)
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b, alpha)
    except Exception:
        return (0.7, 0.7, 0.7, alpha)


# --------------------- Link stage color overrides ---------------------
def build_link_stage_color_overrides(link_stage_override_rows: Any) -> Dict[int, str]:
    """
    Build a stage_index -> hex color dict for fixed link-color overrides.

    stage_index is 0-based, corresponding to segments:
      0: col0 -> col1
      1: col1 -> col2
      ...
    Skip-column links are attributed to (target_col - 1) by the caller.
    """
    out: Dict[int, str] = {}
    rows = link_stage_override_rows or []
    if not isinstance(rows, list):
        return out

    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            stage = int(r.get("stage_index", r.get("stage", 0)))
        except Exception:
            continue
        enabled = bool(r.get("enable", r.get("enabled", False)))
        if not enabled:
            continue
        color = normalize_hex_or_default(r.get("color"), "")
        if not color:
            continue
        out[stage] = color
    return out


# --------------------- Link color policy ---------------------
def resolve_link_color_pair(
    *,
    mode: str,
    follow_col: int,
    n_cols: int,
    # current edge context
    row_nodes: List[str],
    row_colors: List[str],
    col_colors: List[Dict[str, str]],
    s_col: int,
    t_col: int,
    src_node: str,
    tgt_node: str,
    # optional stage override
    stage_override: Dict[int, str] | None,
) -> Tuple[str, str]:
    """
    Returns (c_start, c_end) in hex.

    If stage_override has an enabled color for the attributed stage, it overrides BOTH.
    Segment attribution: stage_index = (target_col - 1) clamped to [0, n_cols-2].
    """
    mode = str(mode or "source").lower()

    src_color = col_colors[s_col].get(src_node, (row_colors[s_col] if s_col < len(row_colors) else "") or "#999999")
    tgt_color = col_colors[t_col].get(tgt_node, (row_colors[t_col] if t_col < len(row_colors) else "") or src_color)

    if mode in ("follow_col", "follow_column", "followcol", "follow"):
        follow = int(follow_col or 0)
        follow = max(0, min(n_cols - 1, follow))
        base_node = row_nodes[follow] if follow < len(row_nodes) else ""
        if not base_node:
            base_node = src_node
        base_color = col_colors[follow].get(base_node, (row_colors[follow] if follow < len(row_colors) else "") or src_color or "#999999")
        c_start = base_color
        c_end = base_color
    elif mode == "target":
        c_start = tgt_color
        c_end = tgt_color
    elif mode == "gradient":
        c_start = src_color
        c_end = tgt_color
    else:
        c_start = src_color
        c_end = src_color

    if stage_override:
        stage_idx = int(t_col) - 1
        stage_idx = max(0, min(n_cols - 2, stage_idx))
        forced = stage_override.get(stage_idx)
        if forced:
            c_start = forced
            c_end = forced

    return c_start, c_end
# --------------------- Fonts ---------------------
def choose_font(cands: Tuple[str, ...]) -> str:
    """Return the first available font name from candidates."""
    try:
        avail = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        avail = set()
    for name in cands:
        n = str(name).strip()
        if not n:
            continue
        if n in avail:
            return n
    if "DejaVu Sans" in avail:
        return "DejaVu Sans"
    # fallback to any available font name (avoid generic family "sans-serif" here)
    if avail:
        return sorted(avail)[0]
    return "DejaVu Sans"
