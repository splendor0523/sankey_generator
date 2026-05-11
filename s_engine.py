
"""
sankey_engine.py (facade)
-------------------------
Drop-in replacement for the original monolithic file.

The implementation is now layered:
- graph.py   : parsing + graph construction (nodes/edges/heights)
- layout.py  : geometry/layout (x/y positions, grouping, bands, label density)
- colors.py  : color utilities + link color policies (incl. per-stage overrides)
- render.py  : matplotlib primitives + render_sankey_from_df()

Public API preserved:
- SankeyConfig
- render_sankey_from_df
- infer_n_cols_from_df
- scale_config_for_preview
"""
from __future__ import annotations

from dataclasses import asdict
import pandas as pd

from sankey_core.graph import SankeyConfig
from sankey_core.render import render_sankey_from_df


def infer_n_cols_from_df(df: pd.DataFrame) -> int:
    df2 = df.dropna(how="all", axis=1)
    m = df2.shape[1]
    n = int(round((m + 1) / 3))
    if n < 1:
        return 1
    if (3 * n - 1) > m:
        n = max(1, (m + 1) // 3)
    return max(1, int(n))


def scale_config_for_preview(cfg: SankeyConfig, preview_w_in=16.0, preview_h_in=10.0, preview_dpi=150) -> SankeyConfig:
    """
    Create a scaled-down config for preview rendering.

    Notes:
    - For cfg.orientation == "vertical", the renderer will swap width/height at draw time.
      Here, preview_w_in/preview_h_in refer to the *displayed* preview size.
    """
    ori = str(getattr(cfg, "orientation", "horizontal") or "horizontal").lower().strip()

    # "Displayed" size (what the user sees) depends on orientation.
    if ori == "vertical":
        final_disp_w_px = cfg.fig_height_in * cfg.dpi
        final_disp_h_px = cfg.fig_width_in * cfg.dpi
        prev_disp_w_px = preview_w_in * preview_dpi
        prev_disp_h_px = preview_h_in * preview_dpi

        rx_disp = (prev_disp_w_px / final_disp_w_px) if final_disp_w_px > 0 else 1.0
        ry_disp = (prev_disp_h_px / final_disp_h_px) if final_disp_h_px > 0 else 1.0

        # cfg2 is the config passed to the renderer; the renderer will swap again.
        cfg2 = SankeyConfig(**asdict(cfg))
        cfg2.fig_width_in = float(preview_h_in)
        cfg2.fig_height_in = float(preview_w_in)
        cfg2.dpi = int(preview_dpi)

        # In vertical display: thickness axis = width, flow axis = height
        thickness_scale = rx_disp
        flow_scale = ry_disp
    else:
        final_disp_w_px = cfg.fig_width_in * cfg.dpi
        final_disp_h_px = cfg.fig_height_in * cfg.dpi
        prev_disp_w_px = preview_w_in * preview_dpi
        prev_disp_h_px = preview_h_in * preview_dpi

        rx_disp = (prev_disp_w_px / final_disp_w_px) if final_disp_w_px > 0 else 1.0
        ry_disp = (prev_disp_h_px / final_disp_h_px) if final_disp_h_px > 0 else 1.0

        cfg2 = SankeyConfig(**asdict(cfg))
        cfg2.fig_width_in = float(preview_w_in)
        cfg2.fig_height_in = float(preview_h_in)
        cfg2.dpi = int(preview_dpi)

        # In horizontal display: thickness axis = height, flow axis = width
        thickness_scale = ry_disp
        flow_scale = rx_disp

    # value→pixels & thickness-related thresholds scale with "thickness axis"
    cfg2.value_to_px = cfg.value_to_px * thickness_scale
    cfg2.min_link_px = cfg.min_link_px * thickness_scale
    cfg2.min_node_h_px = cfg.min_node_h_px * thickness_scale

    # label offsets & fonts are screen-vertical concerns (scale with displayed height)
    cfg2.label_offset_px = cfg.label_offset_px * ry_disp
    cfg2.p_value_font_size = cfg.p_value_font_size * ry_disp
    cfg2.text_font_size = cfg.text_font_size * ry_disp
    if hasattr(cfg2, "header_dy_px"):
        cfg2.header_dy_px = float(getattr(cfg, "header_dy_px", 0.0) or 0.0) * ry_disp
    if hasattr(cfg2, "header_font_size") and getattr(cfg, "header_font_size", None) not in (None, "", "nan"):
        try:
            cfg2.header_font_size = float(getattr(cfg, "header_font_size")) * ry_disp
        except Exception:
            pass

    # link % label fonts & offsets
    if hasattr(cfg2, "link_pct_font_size"):
        cfg2.link_pct_font_size = float(getattr(cfg, "link_pct_font_size", 16.0) or 16.0) * ry_disp
    if hasattr(cfg2, "link_pct_dx_px"):
        cfg2.link_pct_dx_px = float(getattr(cfg, "link_pct_dx_px", 0.0) or 0.0) * rx_disp
    if hasattr(cfg2, "link_pct_dy_px"):
        cfg2.link_pct_dy_px = float(getattr(cfg, "link_pct_dy_px", 0.0) or 0.0) * ry_disp


    if hasattr(cfg2, "link_pct_min_sep_px") and getattr(cfg, "link_pct_min_sep_px", None) not in (None, "", "nan"):
        try:
            cfg2.link_pct_min_sep_px = float(getattr(cfg, "link_pct_min_sep_px")) * ry_disp
        except Exception:
            pass


    # legend font size (长节点图例)
    if hasattr(cfg2, "legend_font_size"):
        cfg2.legend_font_size = float(getattr(cfg, "legend_font_size", 16.0) or 16.0) * ry_disp

    # flow-axis spacing
    cfg2.link_node_gap_px = cfg.link_node_gap_px * flow_scale

    # per-column layout rows
    rows = []
    for r in (cfg.col_cfg_rows or []):
        rr = dict(r)
        if "node_width_px" in rr and rr["node_width_px"] is not None:
            rr["node_width_px"] = float(rr["node_width_px"]) * flow_scale
        if "gap_px" in rr and rr["gap_px"] is not None:
            rr["gap_px"] = float(rr["gap_px"]) * thickness_scale
        if "group_gap_px" in rr and rr["group_gap_px"] is not None:
            rr["group_gap_px"] = float(rr["group_gap_px"]) * thickness_scale
        rows.append(rr)
    cfg2.col_cfg_rows = rows

    cfg2.default_node_width_px = cfg.default_node_width_px * flow_scale
    cfg2.default_gap_px = cfg.default_gap_px * thickness_scale

    # label table rows (dx uses displayed width; dy uses displayed height)
    lrows = []
    for r in (cfg.col_label_cfg_rows or []):
        rr = dict(r)
        if "dx_px" in rr and rr["dx_px"] is not None:
            rr["dx_px"] = float(rr["dx_px"]) * rx_disp
        if "dy_px" in rr and rr["dy_px"] is not None:
            rr["dy_px"] = float(rr["dy_px"]) * ry_disp
        if "font_size" in rr and rr["font_size"] not in (None, "", "nan"):
            try:
                rr["font_size"] = float(rr["font_size"]) * ry_disp
            except Exception:
                pass
        lrows.append(rr)
    cfg2.col_label_cfg_rows = lrows

    return cfg2
