
"""
render.py
----------
Matplotlib drawing primitives + full render pipeline.

Public function:
- render_sankey_from_df(df, cfg) -> (pdf_bytes, png_bytes, diag)
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib.path import Path
import matplotlib.patches as patches

from .graph import SankeyConfig, GraphData, build_graph
from .layout import LayoutData, compute_layout, px_to_frac_y, px_to_frac_x
from .colors import (
    choose_font,
    hex_to_rgba,
    build_link_stage_color_overrides,
    resolve_link_color_pair,
    normalize_hex_or_default,
)




from matplotlib.font_manager import FontProperties


# --------------------- Long-label legend helpers ---------------------
def _excel_col_label(idx0: int) -> str:
    """0->A, 1->B, ..., 25->Z, 26->AA ... (Excel-style)."""
    idx = int(idx0)
    if idx < 0:
        return "A"
    s = ""
    while True:
        idx, r = divmod(idx, 26)
        s = chr(65 + r) + s
        if idx == 0:
            break
        idx -= 1
    return s


def _resolve_index_font_name(cfg: SankeyConfig) -> str:
    custom = str(getattr(cfg, "index_label_font", "") or "").strip()
    try:
        fall = tuple(getattr(cfg, "font_priority", ("Arial",)) or ("Arial",))
    except Exception:
        fall = ("Arial",)
    if custom:
        return choose_font((custom,) + fall)
    return choose_font(fall)


def _resolve_global_font_name(cfg: SankeyConfig) -> str:
    try:
        fall = tuple(getattr(cfg, "font_priority", ("Arial",)) or ("Arial",))
    except Exception:
        fall = ("Arial",)
    return choose_font(fall)


def _normalize_font_priority(v) -> Tuple[str, ...]:
    if isinstance(v, (tuple, list)):
        out = tuple([str(x).strip() for x in v if str(x).strip()])
        return out
    if isinstance(v, str):
        s = v.strip()
        return (s,) if s else tuple()
    return tuple()


def _resolve_zh_en_font_names(cfg: SankeyConfig) -> Tuple[str, str]:
    base = _normalize_font_priority(getattr(cfg, "font_priority", ("Arial",))) or ("Arial",)
    zh = _normalize_font_priority(getattr(cfg, "font_zh_priority", tuple()))
    en = _normalize_font_priority(getattr(cfg, "font_en_priority", tuple()))

    zh_chain = zh + base if zh else base
    en_chain = en + base if en else base
    return choose_font(zh_chain), choose_font(en_chain)


def _resolve_header_font_names(cfg: SankeyConfig) -> Tuple[Optional[str], str, str]:
    base = _normalize_font_priority(getattr(cfg, "font_priority", ("Arial",))) or ("Arial",)
    h_main = _normalize_font_priority(getattr(cfg, "header_font_priority", tuple()))
    h_zh = _normalize_font_priority(getattr(cfg, "header_font_zh_priority", tuple()))
    h_en = _normalize_font_priority(getattr(cfg, "header_font_en_priority", tuple()))
    c_main = h_main + base if h_main else base
    c_zh = h_zh + c_main if h_zh else c_main
    c_en = h_en + c_main if h_en else c_main
    main_name = choose_font(c_main) if c_main else None
    zh_name = choose_font(c_zh)
    en_name = choose_font(c_en)
    return main_name, zh_name, en_name


def _resolve_title_font_names(cfg: SankeyConfig) -> Tuple[Optional[str], str, str]:
    base = _normalize_font_priority(getattr(cfg, "font_priority", ("Arial",))) or ("Arial",)
    t_main = _normalize_font_priority(getattr(cfg, "title_font_priority", tuple()))
    t_zh = _normalize_font_priority(getattr(cfg, "title_font_zh_priority", tuple()))
    t_en = _normalize_font_priority(getattr(cfg, "title_font_en_priority", tuple()))
    c_main = t_main + base if t_main else base
    c_zh = t_zh + c_main if t_zh else c_main
    c_en = t_en + c_main if t_en else c_main
    main_name = choose_font(c_main) if c_main else None
    zh_name = choose_font(c_zh)
    en_name = choose_font(c_en)
    return main_name, zh_name, en_name


def is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (
        (0x3400 <= cp <= 0x4DBF) or
        (0x4E00 <= cp <= 0x9FFF) or
        (0xF900 <= cp <= 0xFAFF) or
        (0x20000 <= cp <= 0x2A6DF) or
        (0x2A700 <= cp <= 0x2B73F) or
        (0x2B740 <= cp <= 0x2B81F) or
        (0x2B820 <= cp <= 0x2CEAF) or
        (0x2F800 <= cp <= 0x2FA1F) or
        (0x3000 <= cp <= 0x303F) or
        (0xFF00 <= cp <= 0xFFEF)
    )


def split_by_script(text: str) -> List[Tuple[bool, str]]:
    s = str(text or "")
    if not s:
        return []

    parts: List[Tuple[bool, str]] = []
    cur_is_cjk = is_cjk_char(s[0])
    cur = [s[0]]
    for ch in s[1:]:
        ch_is_cjk = is_cjk_char(ch)
        if ch_is_cjk == cur_is_cjk:
            cur.append(ch)
        else:
            parts.append((cur_is_cjk, "".join(cur)))
            cur_is_cjk = ch_is_cjk
            cur = [ch]
    parts.append((cur_is_cjk, "".join(cur)))
    return parts


def _normalize_wrap_targets(v) -> set:
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
    elif isinstance(v, (tuple, list, set)):
        parts = [str(p).strip() for p in v]
    else:
        parts = []
    return {p for p in parts if p}


def _cjk_visible_char_count(text: str) -> int:
    s = str(text or "")
    return sum(1 for ch in s if (not ch.isspace()) and is_cjk_char(ch))


def _tokenize_for_cjk_wrap(line: str) -> List[Tuple[str, str]]:
    """
    Tokenize line for mixed CJK/English wrapping:
    - Keep English/alnum runs together (avoid hard-cut words like "ABC123")
    - Keep other chars as single-char tokens
    """
    s = str(line or "")
    out: List[Tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        m = re.match(r"[A-Za-z0-9][A-Za-z0-9_\-./]*", s[i:])
        if m:
            tok = m.group(0)
            out.append(("eng", tok))
            i += len(tok)
            continue
        ch = s[i]
        out.append(("cjk" if is_cjk_char(ch) else "other", ch))
        i += 1
    return out


def _wrap_one_line_by_cjk_chars(line: str, chars_per_line: int) -> List[str]:
    s = str(line or "")
    if (not s) or (chars_per_line <= 0):
        return [s]

    tokens = _tokenize_for_cjk_wrap(s)
    if not tokens:
        return [s]

    lines: List[str] = []
    cur_tokens: List[str] = []
    cur_cjk = 0

    for _typ, tok in tokens:
        tok_cjk = _cjk_visible_char_count(tok)
        need_break = (cur_tokens and tok_cjk > 0 and (cur_cjk + tok_cjk) > int(chars_per_line))
        if need_break:
            lines.append("".join(cur_tokens).rstrip())
            cur_tokens = []
            cur_cjk = 0

        cur_tokens.append(tok)
        cur_cjk += tok_cjk

    if cur_tokens:
        lines.append("".join(cur_tokens).rstrip())
    return lines or [s]


def cjk_auto_wrap_text(text: str, *, chars_per_line: int, max_lines: Optional[int] = None) -> str:
    """
    Wrap by CJK visible char count while preserving manual line breaks.
    English/alnum runs are kept as whole tokens where possible.
    """
    s = str(text or "")
    if (not s) or (chars_per_line <= 0):
        return s

    all_lines: List[str] = []
    for raw_line in s.split("\n"):
        all_lines.extend(_wrap_one_line_by_cjk_chars(raw_line, int(chars_per_line)))

    if max_lines is not None:
        try:
            ml = int(max_lines)
        except Exception:
            ml = 0
        if ml > 0 and len(all_lines) > ml:
            head = all_lines[: ml - 1]
            tail = "".join(all_lines[ml - 1 :])
            all_lines = head + [tail]

    return "\n".join(all_lines)


def _pick_script_font(is_cjk: bool, zh_family: Optional[str], en_family: Optional[str], fallback_family: Optional[str]) -> Optional[str]:
    if is_cjk and zh_family:
        return zh_family
    if (not is_cjk) and en_family:
        return en_family
    return fallback_family


def _apply_faux_bold_to_text_artist(t, *, enabled: bool, width_px: float, dpi: int):
    if (not enabled) or t is None:
        return
    try:
        fw = str(t.get_fontweight() or "").lower().strip()
    except Exception:
        fw = ""
    if fw not in ("bold", "heavy", "black", "semibold", "demibold"):
        return
    try:
        color = t.get_color()
    except Exception:
        color = "#000000"
    w_px = max(0.0, float(width_px or 0.0))
    if w_px <= 0:
        return
    lw_pt = max(0.1, w_px * 72.0 / float(max(1, int(dpi))))
    try:
        t.set_path_effects([path_effects.withStroke(linewidth=lw_pt, foreground=color)])
    except Exception:
        return


def _apply_faux_bold_to_axis_texts(ax, *, enabled: bool, width_px: float, dpi: int):
    if (not enabled) or ax is None:
        return
    for t in list(getattr(ax, "texts", []) or []):
        _apply_faux_bold_to_text_artist(t, enabled=True, width_px=width_px, dpi=dpi)


def _draw_mixed_text_single_line(
    ax,
    renderer,
    *,
    x: float,
    y: float,
    text: str,
    ha: str,
    va: str,
    fs: float,
    color: str,
    weight: str = "normal",
    style: str = "normal",
    zh_family: Optional[str] = None,
    en_family: Optional[str] = None,
    fallback_family: Optional[str] = None,
    transform=None,
    zorder: int = 20,
    gid: Optional[str] = None,
) -> List[Any]:
    s = str(text or "")
    if s == "":
        return []

    tr = transform if transform is not None else ax.transData
    parts = split_by_script(s)
    if (renderer is None) or (not parts):
        t = ax.text(
            x, y, s,
            transform=tr,
            ha=ha, va=va,
            fontsize=float(fs),
            color=color,
            fontweight=weight,
            fontstyle=style,
            fontfamily=(fallback_family or en_family or zh_family),
            zorder=zorder,
        )
        if gid:
            try:
                t.set_gid(gid)
            except Exception:
                pass
        return [t]

    widths_px: List[float] = []
    families: List[Optional[str]] = []
    for is_cjk, seg in parts:
        fam = _pick_script_font(is_cjk, zh_family, en_family, fallback_family)
        w_px, _h_px = _text_wh_px(renderer, seg, fs, weight=weight, style=style, family=fam)
        widths_px.append(w_px)
        families.append(fam)

    total_w_px = float(sum(widths_px))
    x_disp, y_disp = tr.transform((float(x), float(y)))
    if ha == "right":
        start_x_disp = x_disp - total_w_px
    elif ha == "center":
        start_x_disp = x_disp - total_w_px / 2.0
    else:
        start_x_disp = x_disp

    artists: List[Any] = []
    cur_x_disp = start_x_disp
    for idx, (_is_cjk, seg) in enumerate(parts):
        x_seg, y_seg = tr.inverted().transform((cur_x_disp, y_disp))
        t = ax.text(
            x_seg, y_seg, seg,
            transform=tr,
            ha="left", va=va,
            fontsize=float(fs),
            color=color,
            fontweight=weight,
            fontstyle=style,
            fontfamily=families[idx],
            zorder=zorder,
        )
        if gid:
            try:
                t.set_gid(gid)
            except Exception:
                pass
        artists.append(t)
        cur_x_disp += widths_px[idx]
    return artists


def _draw_mixed_text(
    ax,
    renderer,
    *,
    x: float,
    y: float,
    text: str,
    ha: str,
    va: str,
    fs: float,
    color: str,
    weight: str = "normal",
    style: str = "normal",
    zh_family: Optional[str] = None,
    en_family: Optional[str] = None,
    fallback_family: Optional[str] = None,
    transform=None,
    zorder: int = 20,
    line_spacing: float = 1.20,
    gid: Optional[str] = None,
) -> List[Any]:
    s = str(text or "")
    if s == "":
        return []

    tr = transform if transform is not None else ax.transData
    lines = s.split("\n")
    if len(lines) <= 1:
        return _draw_mixed_text_single_line(
            ax, renderer, x=x, y=y, text=s, ha=ha, va=va, fs=fs, color=color,
            weight=weight, style=style, zh_family=zh_family, en_family=en_family,
            fallback_family=fallback_family, transform=tr, zorder=zorder, gid=gid
        )

    # Multiline block layout in display coords, then draw each line as single-line mixed text.
    base_h_px = 0.0
    for ln in lines:
        _w0, h0 = _text_wh_px_mixed(
            renderer, ln, fs, weight=weight, style=style,
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
        )
        base_h_px = max(base_h_px, float(h0))
    if base_h_px <= 0:
        base_h_px = float(fs) * 1.2

    step_px = base_h_px * max(1.0, float(line_spacing))
    block_h_px = base_h_px + step_px * max(0, len(lines) - 1)
    x_disp, y_disp = tr.transform((float(x), float(y)))

    va0 = str(va or "center").lower().strip()
    if va0 == "top":
        first_center_y = y_disp - base_h_px / 2.0
    elif va0 == "bottom":
        first_center_y = y_disp + block_h_px - base_h_px / 2.0
    else:
        first_center_y = y_disp + block_h_px / 2.0 - base_h_px / 2.0

    out: List[Any] = []
    for i, ln in enumerate(lines):
        line_center_y = first_center_y - i * step_px
        x_line, y_line = tr.inverted().transform((x_disp, line_center_y))
        out.extend(
            _draw_mixed_text_single_line(
                ax, renderer, x=float(x_line), y=float(y_line), text=ln, ha=ha, va="center",
                fs=fs, color=color, weight=weight, style=style,
                zh_family=zh_family, en_family=en_family, fallback_family=fallback_family,
                transform=tr, zorder=zorder, gid=gid
            )
        )
    return out


def _verticalize_label_text(text: str, *, reverse: bool = False) -> str:
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    chars = [ch for ch in s if ch != "\n"]
    if reverse:
        chars = list(reversed(chars))
    return "\n".join(chars)


def _text_wh_px(renderer, text: str, fs: float, *, weight="normal", style="normal", family: Optional[str] = None) -> Tuple[float, float]:
    if renderer is None:
        s = str(text or "")
        est_w = float(len(s)) * float(fs) * 0.6
        est_h = float(fs) * 1.2
        return est_w, est_h
    prop = FontProperties(size=float(fs), weight=weight, style=style)
    if family:
        try:
            prop.set_family(family)
        except Exception:
            pass
    w, h, _d = renderer.get_text_width_height_descent(str(text), prop, ismath=False)
    return float(w), float(h)


def _text_wh_px_mixed(renderer, text: str, fs: float, *, weight="normal", style="normal",
                      family: Optional[str] = None, zh_family: Optional[str] = None,
                      en_family: Optional[str] = None, fallback_family: Optional[str] = None) -> Tuple[float, float]:
    if zh_family is None and en_family is None:
        return _text_wh_px(renderer, text, fs, weight=weight, style=style, family=family)

    parts = split_by_script(str(text or ""))
    if not parts:
        return 0.0, 0.0

    total_w = 0.0
    max_h = 0.0
    for is_cjk, seg in parts:
        fam = _pick_script_font(is_cjk, zh_family, en_family, fallback_family or family)
        w, h = _text_wh_px(renderer, seg, fs, weight=weight, style=style, family=fam)
        total_w += w
        if h > max_h:
            max_h = h
    return float(total_w), float(max_h)


def _text_block_wh_px_mixed(renderer, text: str, fs: float, *, weight="normal", style="normal",
                            family: Optional[str] = None, zh_family: Optional[str] = None,
                            en_family: Optional[str] = None, fallback_family: Optional[str] = None,
                            line_spacing: float = 1.20) -> Tuple[float, float]:
    lines = str(text or "").split("\n")
    if not lines:
        return 0.0, 0.0
    max_w = 0.0
    base_h = 0.0
    for ln in lines:
        w, h = _text_wh_px_mixed(
            renderer, ln, fs, weight=weight, style=style,
            family=family, zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
        )
        max_w = max(max_w, float(w))
        base_h = max(base_h, float(h))
    if base_h <= 0:
        base_h = float(fs) * 1.2
    step = base_h * max(1.0, float(line_spacing))
    total_h = base_h + step * max(0, len(lines) - 1)
    return float(max_w), float(total_h)


def _wrap_text_to_width_px(renderer, text: str, max_px: float, *, fs: float, family: Optional[str] = None,
                           weight="normal", style="normal", zh_family: Optional[str] = None,
                           en_family: Optional[str] = None, fallback_family: Optional[str] = None) -> List[str]:
    """Wrap text into multiple lines so each line width <= max_px. Never uses ellipsis."""
    s = str(text)
    if "\n" in s:
        out_manual: List[str] = []
        for sub in s.split("\n"):
            out_manual.extend(
                _wrap_text_to_width_px(
                    renderer, sub, max_px, fs=fs, family=family, weight=weight, style=style,
                    zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
                )
            )
        return out_manual if out_manual else [""]
    if max_px <= 1:
        return [s] if s else [""]
    # quick accept
    w, _h = _text_wh_px_mixed(
        renderer, s, fs, weight=weight, style=style, family=family,
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
    )
    if w <= max_px:
        return [s]

    # word wrap if spaces exist
    if " " in s:
        words = s.split(" ")
        lines = []
        cur = ""
        for w0 in words:
            cand = (cur + " " + w0).strip() if cur else w0
            w_c, _ = _text_wh_px_mixed(
                renderer, cand, fs, weight=weight, style=style, family=family,
                zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
            )
            if w_c <= max_px or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w0
        if cur:
            lines.append(cur)
        # if still too wide (very long token), fall back to char-wrap per line
        out = []
        for ln in lines:
            w_ln, _ = _text_wh_px_mixed(
                renderer, ln, fs, weight=weight, style=style, family=family,
                zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
            )
            if w_ln <= max_px:
                out.append(ln)
            else:
                out.extend(
                    _wrap_text_to_width_px(
                        renderer, ln, max_px, fs=fs, family=family, weight=weight, style=style,
                        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
                    )
                )
        return out

    # char wrap
    lines = []
    cur = ""
    for ch in s:
        cand = cur + ch
        w_c, _ = _text_wh_px_mixed(
            renderer, cand, fs, weight=weight, style=style, family=family,
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
        )
        if w_c <= max_px or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _compute_sankey_x_bounds(ax, g: GraphData, lay: LayoutData, cfg: SankeyConfig, is_vertical: bool) -> Tuple[float, float]:
    placeholder = getattr(g, "placeholder_nodes_set", set()) or set()

    if not is_vertical:
        left, right = 1.0, 0.0
        for j in range(cfg.n_cols):
            w = lay.node_w_frac_col[j]
            left = min(left, lay.x_pos[j] - w / 2)
            right = max(right, lay.x_pos[j] + w / 2)
    else:
        left, right = 1.0, 0.0
        for j in range(cfg.n_cols):
            for node, cx, w_frac in zip(g.col_nodes[j], lay.col_centers[j], lay.col_h_frac[j]):
                if node in placeholder:
                    continue
                left = min(left, cx - w_frac / 2)
                right = max(right, cx + w_frac / 2)

    if not (left < right):
        left, right = 0.0, 1.0
    left = max(0.0, float(left))
    right = min(1.0, float(right))
    return left, right


def _draw_right_color_legend(
    ax,
    renderer,
    *,
    items: List[Tuple[int, str, str, str]],
    fs: float,
    position: str = "right",
    dx_px: float = 0.0,
    dy_px: float = 0.0,
    family: Optional[str],
    zh_family: Optional[str],
    en_family: Optional[str],
    fallback_family: Optional[str],
    line_spacing: float,
):
    if not items:
        return
    fig_w_px = max(1.0, float(ax.bbox.width))
    fig_h_px = max(1.0, float(ax.bbox.height))
    swatch_h_px = max(1.0, float(fs) * float(ax.figure.dpi) / 72.0)
    swatch_w_px = swatch_h_px * 0.8
    row_gap_px = swatch_h_px * 1.25
    swatch_w_axes = swatch_w_px / fig_w_px
    swatch_h_axes = swatch_h_px / fig_h_px
    text_gap_axes = (swatch_h_px * 0.35) / fig_w_px
    item_gap_axes = (swatch_h_px * 0.9) / fig_w_px
    dx_axes = float(dx_px or 0.0) / fig_w_px
    dy_axes = float(dy_px or 0.0) / fig_h_px
    pos = str(position or "right").lower().strip()
    if pos not in ("left", "right", "bottom"):
        pos = "right"

    unique_items = []
    seen = set()
    for col_idx, node_key, label, color in items:
        dedup_key = (int(col_idx), str(node_key))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique_items.append((col_idx, node_key, label, color))

    if pos == "bottom":
        widths_axes = []
        for _col_idx, _node_key, label, _color in unique_items:
            txt_w_px, _txt_h_px = _text_wh_px_mixed(
                renderer,
                str(label),
                float(fs),
                zh_family=zh_family,
                en_family=en_family,
                fallback_family=fallback_family or family,
            )
            widths_axes.append(swatch_w_axes + text_gap_axes + (txt_w_px / fig_w_px))
        total_w_axes = sum(widths_axes) + max(0, len(widths_axes) - 1) * item_gap_axes
        x = 0.5 - total_w_axes / 2.0 + dx_axes
        y_center = -0.08 - dy_axes
        for idx, (_col_idx, _node_key, label, color) in enumerate(unique_items):
            rect = patches.Rectangle(
                (x, y_center - swatch_h_axes / 2.0),
                swatch_w_axes,
                swatch_h_axes,
                transform=ax.transAxes,
                facecolor=hex_to_rgba(normalize_hex_or_default(color, "#999999"), 1.0),
                edgecolor="none",
                linewidth=0,
                clip_on=False,
                zorder=80,
            )
            try:
                rect.set_gid("legend")
            except Exception:
                pass
            ax.add_patch(rect)
            _draw_mixed_text(
                ax,
                renderer,
                x=x + swatch_w_axes + text_gap_axes,
                y=y_center,
                text=str(label),
                ha="left",
                va="center",
                fs=float(fs),
                color="#000000",
                weight="normal",
                style="normal",
                zh_family=zh_family,
                en_family=en_family,
                fallback_family=fallback_family or family,
                transform=ax.transAxes,
                zorder=81,
                line_spacing=line_spacing,
                gid="legend",
            )
            x += widths_axes[idx] + item_gap_axes
        return

    x0 = (1.02 if pos == "right" else -0.08) + dx_axes
    y = 0.98 - dy_axes
    for col_idx, node_key, label, color in unique_items:
        y_center = y - swatch_h_axes / 2.0
        rect = patches.Rectangle(
            (x0, y_center - swatch_h_axes / 2.0),
            swatch_w_axes,
            swatch_h_axes,
            transform=ax.transAxes,
            facecolor=hex_to_rgba(normalize_hex_or_default(color, "#999999"), 1.0),
            edgecolor="none",
            linewidth=0,
            clip_on=False,
            zorder=80,
        )
        try:
            rect.set_gid("legend")
        except Exception:
            pass
        ax.add_patch(rect)
        _draw_mixed_text(
            ax,
            renderer,
            x=x0 + swatch_w_axes + text_gap_axes,
            y=y_center,
            text=str(label),
            ha="left",
            va="center",
            fs=float(fs),
            color="#000000",
            weight="normal",
            style="normal",
            zh_family=zh_family,
            en_family=en_family,
            fallback_family=fallback_family or family,
            transform=ax.transAxes,
            zorder=81,
            line_spacing=line_spacing,
            gid="legend",
        )
        y -= row_gap_px / fig_h_px


def _compute_current_bottom_y_axes(ax, renderer) -> float:
    bboxes = []
    for artist in list(ax.patches) + list(ax.texts):
        try:
            if hasattr(artist, "get_visible") and (not artist.get_visible()):
                continue
            bb = artist.get_window_extent(renderer=renderer)
            if bb is not None:
                bboxes.append(bb)
        except Exception:
            continue
    if not bboxes:
        return 0.0
    y0_disp = min(bb.y0 for bb in bboxes)
    _x_axes, y0_axes = ax.transAxes.inverted().transform((0.0, y0_disp))
    return float(y0_axes)


def _compute_current_top_y_axes(ax, renderer) -> float:
    bboxes = []
    for artist in list(ax.patches) + list(ax.texts):
        try:
            if hasattr(artist, "get_visible") and (not artist.get_visible()):
                continue
            bb = artist.get_window_extent(renderer=renderer)
            if bb is not None:
                bboxes.append(bb)
        except Exception:
            continue
    if not bboxes:
        return 1.0
    y1_disp = max(bb.y1 for bb in bboxes)
    _x_axes, y1_axes = ax.transAxes.inverted().transform((0.0, y1_disp))
    return float(y1_axes)


def _is_valid_hex_color(s: str) -> bool:
    t = str(s or "").strip()
    if not t:
        return False
    return re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", t) is not None


def _iter_text_artists_for_autofit(ax, *, consider_legend: bool, consider_link_pct: bool) -> List[Any]:
    out: List[Any] = []
    for t in list(ax.texts):
        try:
            if hasattr(t, "get_visible") and (not t.get_visible()):
                continue
            txt = str(t.get_text() or "").strip()
            if txt == "":
                continue
            gid = str(getattr(t, "get_gid", lambda: "")() or "")
            if (not consider_legend) and gid.startswith("legend_"):
                continue
            if (not consider_link_pct) and gid == "link_pct":
                continue
            out.append(t)
        except Exception:
            continue
    return out


def _bbox_overlap_px(a, b) -> float:
    dx = min(float(a.x1), float(b.x1)) - max(float(a.x0), float(b.x0))
    dy = min(float(a.y1), float(b.y1)) - max(float(a.y0), float(b.y0))
    if dx <= 0 or dy <= 0:
        return 0.0
    return float(dx * dy)


def _measure_text_overlap_metrics(ax, renderer, *, consider_legend: bool, consider_link_pct: bool) -> Dict[str, float]:
    texts = _iter_text_artists_for_autofit(
        ax,
        consider_legend=bool(consider_legend),
        consider_link_pct=bool(consider_link_pct),
    )
    bboxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
            if bb is not None:
                bboxes.append(bb)
        except Exception:
            continue

    overlap_pairs = 0
    overlap_area_px = 0.0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            ov = _bbox_overlap_px(bboxes[i], bboxes[j])
            if ov > 0:
                overlap_pairs += 1
                overlap_area_px += ov

    axbb = ax.bbox
    overflow_count = 0
    overflow_x_px = 0.0
    overflow_y_px = 0.0
    for bb in bboxes:
        left_ov = max(0.0, float(axbb.x0) - float(bb.x0))
        right_ov = max(0.0, float(bb.x1) - float(axbb.x1))
        bottom_ov = max(0.0, float(axbb.y0) - float(bb.y0))
        top_ov = max(0.0, float(bb.y1) - float(axbb.y1))
        if (left_ov + right_ov + bottom_ov + top_ov) > 0:
            overflow_count += 1
            overflow_x_px += (left_ov + right_ov)
            overflow_y_px += (bottom_ov + top_ov)

    return {
        "text_count": float(len(bboxes)),
        "overlap_pairs": float(overlap_pairs),
        "overlap_area_px": float(overlap_area_px),
        "overflow_count": float(overflow_count),
        "overflow_x_px": float(overflow_x_px),
        "overflow_y_px": float(overflow_y_px),
        "ax_w_px": float(max(1.0, axbb.width)),
        "ax_h_px": float(max(1.0, axbb.height)),
    }


def _clone_cfg_for_autofit(cfg: SankeyConfig) -> SankeyConfig:
    try:
        return SankeyConfig(**asdict(cfg))
    except Exception:
        return SankeyConfig(**dict(cfg.__dict__))


def _scale_gap_rows(rows: Optional[List[Dict[str, Any]]], factor: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in (rows or []):
        rr = dict(r)
        if "gap_px" in rr and rr["gap_px"] not in (None, "", "nan"):
            try:
                rr["gap_px"] = float(rr["gap_px"]) * factor
            except Exception:
                pass
        out.append(rr)
    return out


def _auto_fit_next_cfg(cfg: SankeyConfig, metrics: Dict[str, float], *, iteration: int, is_vertical: bool) -> SankeyConfig:
    nxt = _clone_cfg_for_autofit(cfg)
    ax_w = float(metrics.get("ax_w_px", 1.0) or 1.0)
    ax_h = float(metrics.get("ax_h_px", 1.0) or 1.0)
    ov_x = float(metrics.get("overflow_x_px", 0.0) or 0.0)
    ov_y = float(metrics.get("overflow_y_px", 0.0) or 0.0)
    overlap_pairs = float(metrics.get("overlap_pairs", 0.0) or 0.0)

    # Prefer expanding canvas first.
    kx = 1.0 + max(0.08, min(0.60, (ov_x / ax_w) + 0.04 * min(5.0, overlap_pairs)))
    ky = 1.0 + max(0.08, min(0.60, (ov_y / ax_h) + 0.04 * min(5.0, overlap_pairs)))
    if bool(getattr(cfg, "auto_fit_prefer_expand_canvas", True)):
        if not is_vertical:
            nxt.fig_width_in = float(nxt.fig_width_in) * kx
            nxt.fig_height_in = float(nxt.fig_height_in) * ky
        else:
            # render canvas swaps width/height in vertical mode
            nxt.fig_height_in = float(nxt.fig_height_in) * kx
            nxt.fig_width_in = float(nxt.fig_width_in) * ky

    # Then micro-tune spacing and label offset (without touching fonts).
    tune_phase = (iteration >= max(1, int(getattr(cfg, "auto_fit_max_iter", 8) or 8) // 2)) or (
        not bool(getattr(cfg, "auto_fit_prefer_expand_canvas", True))
    )
    if tune_phase:
        gap_factor = 1.0 + min(0.25, 0.05 + 0.03 * max(0, int(iteration)))
        nxt.default_gap_px = float(nxt.default_gap_px) * gap_factor
        nxt.col_cfg_rows = _scale_gap_rows(getattr(nxt, "col_cfg_rows", None), gap_factor)
        nxt.label_offset_px = float(nxt.label_offset_px) + max(2.0, float(nxt.text_font_size) * 0.08)

    return nxt


def _draw_legend_item_wrapped(ax, renderer, *, x: float, y_top: float, max_w_axes: float,
                              code: str, name: str, fs: float, family: Optional[str],
                              zh_family: Optional[str] = None, en_family: Optional[str] = None,
                              fallback_family: Optional[str] = None,
                              line_spacing: float = 1.25, zorder: int = 80) -> float:
    """
    Draw one legend item (code bold + name normal) wrapped within max_w_axes.
    Returns height used in Axes coords (positive).
    """
    code = str(code)
    name = str(name)

    # measure code width (with trailing space)
    code_part = code + " "
    code_px, h_px = _text_wh_px_mixed(
        renderer, code_part, fs, weight="bold", family=family,
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
    )
    code_w_axes = code_px / ax.bbox.width
    line_h_axes = (h_px * line_spacing) / ax.bbox.height

    avail_name_px = max(1.0, (max_w_axes - code_w_axes) * ax.bbox.width)
    name_lines = _wrap_text_to_width_px(
        renderer, name, avail_name_px, fs=fs, family=family, weight="normal",
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
    )

    # draw first line with code
    _draw_mixed_text(
        ax, renderer, x=x, y=y_top, text=code, transform=ax.transAxes,
        ha="left", va="top", fs=fs, color="black", weight="bold", style="normal",
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family or family, zorder=zorder,
        line_spacing=line_spacing, gid="legend_index"
    )
    _draw_mixed_text(
        ax, renderer, x=x + code_w_axes, y=y_top, text=(name_lines[0] if name_lines else ""), transform=ax.transAxes,
        ha="left", va="top", fs=fs, color="black", weight="normal", style="normal",
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family or family, zorder=zorder,
        line_spacing=line_spacing, gid="legend_label"
    )

    # continuation lines aligned under name start
    for i in range(1, len(name_lines)):
        yy = y_top - line_h_axes * i
        _draw_mixed_text(
            ax, renderer, x=x + code_w_axes, y=yy, text=name_lines[i], transform=ax.transAxes,
            ha="left", va="top", fs=fs, color="black", weight="normal", style="normal",
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family or family, zorder=zorder,
            line_spacing=line_spacing, gid="legend_label"
        )

    used = line_h_axes * max(1, len(name_lines))
    return float(used)


def _draw_legend_packed(ax, renderer, items: List[Tuple[str, str]], *,
                        x_left: float, x_right: float, top_y: float,
                        fs: float, family: Optional[str], item_gap_spaces: int = 3,
                        zh_family: Optional[str] = None, en_family: Optional[str] = None,
                        fallback_family: Optional[str] = None,
                        line_spacing: float = 1.25) -> None:
    if not items:
        return
    gap_txt = " " * max(1, int(item_gap_spaces))
    gap_px, _ = _text_wh_px_mixed(
        renderer, gap_txt, fs, family=family,
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
    )
    gap_axes = gap_px / ax.bbox.width

    x = float(x_left)
    y = float(top_y)
    max_w = float(x_right - x_left)

    for code, name in items:
        # estimate one-line width
        code_part = str(code) + " "
        code_px, h_px = _text_wh_px_mixed(
            renderer, code_part, fs, weight="bold", family=family,
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
        )
        name_px, _ = _text_wh_px_mixed(
            renderer, str(name), fs, weight="normal", family=family,
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
        )
        item_w_axes = (code_px + name_px) / ax.bbox.width

        # if doesn't fit remaining, wrap to next line
        if (x > x_left) and ((x + item_w_axes) > x_right):
            # move to new line
            line_h_axes = (h_px * 1.25) / ax.bbox.height
            y -= line_h_axes
            x = float(x_left)

        # if still too wide at line start, draw wrapped block spanning full width (no ellipsis)
        if (x == x_left) and (item_w_axes > max_w):
            used = _draw_legend_item_wrapped(
                ax, renderer,
                x=x, y_top=y, max_w_axes=max_w,
                code=code, name=name, fs=fs, family=family,
                zh_family=zh_family, en_family=en_family, fallback_family=fallback_family,
                line_spacing=line_spacing
            )
            y -= used
            x = float(x_left)
            continue

        # draw one-line item
        code_w_axes = code_px / ax.bbox.width
        _draw_mixed_text(
            ax, renderer, x=x, y=y, text=str(code), transform=ax.transAxes,
            ha="left", va="top", fs=fs, color="black", weight="bold", style="normal",
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family or family, zorder=80,
            line_spacing=line_spacing, gid="legend_index"
        )
        _draw_mixed_text(
            ax, renderer, x=x + code_w_axes, y=y, text=str(name), transform=ax.transAxes,
            ha="left", va="top", fs=fs, color="black", weight="normal", style="normal",
            zh_family=zh_family, en_family=en_family, fallback_family=fallback_family or family, zorder=80,
            line_spacing=line_spacing, gid="legend_label"
        )
        x = x + item_w_axes + gap_axes


def _draw_legend_by_column_compact(ax, renderer, items_by_col: Dict[int, List[Tuple[str, str]]], *,
                                  active_cols: List[int],
                                  x_left: float, x_right: float, top_y: float,
                                  fs: float, family: Optional[str],
                                  title_mode: str = "letter",
                                  zh_family: Optional[str] = None, en_family: Optional[str] = None,
                                  fallback_family: Optional[str] = None,
                                  line_spacing: float = 1.25) -> None:
    if not active_cols:
        return
    n = len(active_cols)
    total_w = float(x_right - x_left)
    seg_w = total_w / n if n > 0 else total_w

    # title settings
    title_fs = max(1.0, fs * 0.9)
    # measure title height
    _w_t, h_t = _text_wh_px_mixed(
        renderer, "Ag", title_fs, weight="bold", family=family,
        zh_family=zh_family, en_family=en_family, fallback_family=fallback_family
    )
    title_line_h_axes = (h_t * 1.2) / ax.bbox.height

    for i, col_idx in enumerate(active_cols):
        x0 = float(x_left + i * seg_w)
        max_w = float(seg_w)

        y = float(top_y)

        # title
        mode = (title_mode or "letter").lower().strip()
        if mode != "none":
            if mode == "colnum":
                title = f"Col {col_idx + 1}"
            else:
                title = _excel_col_label(col_idx)
            _draw_mixed_text(
                ax, renderer, x=x0, y=y, text=title, transform=ax.transAxes,
                ha="left", va="top", fs=title_fs, color="black", weight="bold", style="normal",
                zh_family=zh_family, en_family=en_family, fallback_family=fallback_family or family, zorder=81,
                line_spacing=line_spacing, gid="legend_title"
            )
            y -= title_line_h_axes

        for code, name in items_by_col.get(col_idx, []):
            used = _draw_legend_item_wrapped(
                ax, renderer,
                x=x0, y_top=y, max_w_axes=max_w,
                code=code, name=name, fs=fs, family=family,
                zh_family=zh_family, en_family=en_family, fallback_family=fallback_family,
                line_spacing=line_spacing
            )
            y -= used


# --------------------- Drawing primitives ---------------------
def _sample_cubic_bezier(p0, p1, p2, p3, n: int = 48):
    t = np.linspace(0.0, 1.0, max(8, int(n)))
    omt = (1.0 - t)
    x = (omt ** 3) * p0[0] + 3 * (omt ** 2) * t * p1[0] + 3 * omt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
    y = (omt ** 3) * p0[1] + 3 * (omt ** 2) * t * p1[1] + 3 * omt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
    return x, y


def _draw_link_side_outlines(
    ax,
    top_p0, top_p1, top_p2, top_p3,
    bot_p0, bot_p1, bot_p2, bot_p3,
    color_hex: str,
    alpha: float,
    width_px: float,
    dpi: int,
):
    wpx = float(width_px or 0.0)
    if wpx <= 0:
        return
    col = normalize_hex_or_default(color_hex, "#000000")
    # matplotlib line width uses points; convert from requested pixels.
    lw_pt = max(0.1, wpx * 72.0 / float(max(1, int(dpi))))
    rgba = hex_to_rgba(col, max(0.0, min(1.0, float(alpha))))

    tx, ty = _sample_cubic_bezier(top_p0, top_p1, top_p2, top_p3)
    bx, by = _sample_cubic_bezier(bot_p0, bot_p1, bot_p2, bot_p3)
    ax.plot(tx, ty, color=rgba, linewidth=lw_pt, solid_capstyle="round", zorder=2)
    ax.plot(bx, by, color=rgba, linewidth=lw_pt, solid_capstyle="round", zorder=2)


def draw_trapezoid_link(
    ax,
    x0, x1,
    y0, y1,
    h0, h1,
    color_start,
    color_end=None,
    curve_ctrl_rel=0.28,
    alpha=0.55,
    outline_enabled: bool = False,
    outline_color: str = "#000000",
    outline_alpha: float = 0.35,
    outline_width_px: float = 1.0,
    dpi: int = 300,
    p_value_text: str = None,
    text_font_size: float = 12.0
):
    if color_end is None:
        color_end = color_start

    use_gradient = (color_start != color_end)

    top0 = y0 + h0 / 2
    bot0 = y0 - h0 / 2
    top1 = y1 + h1 / 2
    bot1 = y1 - h1 / 2

    ctrl = curve_ctrl_rel
    c1_top = (x0 + ctrl * (x1 - x0), top0)
    c2_top = (x1 - ctrl * (x1 - x0), top1)
    c1_bot = (x0 + ctrl * (x1 - x0), bot0)
    c2_bot = (x1 - ctrl * (x1 - x0), bot1)

    verts = [
        (x0, top0),
        c1_top, c2_top, (x1, top1),
        (x1, bot1),
        c2_bot, c1_bot, (x0, bot0),
        (x0, top0),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY
    ]

    path = Path(verts, codes)

    if not use_gradient:
        ax.add_patch(
            patches.PathPatch(
                path,
                facecolor=hex_to_rgba(color_start, alpha),
                edgecolor="none",
                linewidth=0,
            )
        )
    else:
        clip_patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(clip_patch)

        c_s = np.array(hex_to_rgba(color_start, alpha))
        c_e = np.array(hex_to_rgba(color_end, alpha))

        N = 64
        t = np.linspace(0, 1, N)
        gradient = np.zeros((1, N, 4))
        for i in range(4):
            gradient[0, :, i] = c_s[i] * (1 - t) + c_e[i] * t

        min_y = min(bot0, bot1)
        max_y = max(top0, top1)

        im = ax.imshow(
            gradient,
            aspect='auto',
            extent=[x0, x1, min_y, max_y],
            interpolation='bilinear',
            zorder=1
        )
        im.set_clip_path(clip_patch)

    if bool(outline_enabled):
        _draw_link_side_outlines(
            ax,
            (x0, top0), c1_top, c2_top, (x1, top1),
            (x0, bot0), c1_bot, c2_bot, (x1, bot1),
            color_hex=outline_color,
            alpha=outline_alpha,
            width_px=outline_width_px,
            dpi=dpi,
        )

    if p_value_text:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(
            mx, my, p_value_text,
            ha='center', va='center',
            fontsize=text_font_size,
            color='black',
            zorder=50
        )



def draw_trapezoid_link_vertical(
    ax,
    x0, x1,
    y0, y1,
    h0, h1,
    color_start,
    color_end=None,
    curve_ctrl_rel=0.28,
    alpha=0.55,
    outline_enabled: bool = False,
    outline_color: str = "#000000",
    outline_alpha: float = 0.35,
    outline_width_px: float = 1.0,
    dpi: int = 300,
    p_value_text: str = None,
    text_font_size: float = 12.0
):
    """
    Vertical-flow version of draw_trapezoid_link:

    - Flow axis: y (from y0 to y1)
    - Thickness axis: x (centered at x0/x1, with thickness h0/h1)
    """
    if color_end is None:
        color_end = color_start

    use_gradient = (color_start != color_end)

    right0 = x0 + h0 / 2
    left0 = x0 - h0 / 2
    right1 = x1 + h1 / 2
    left1 = x1 - h1 / 2

    ctrl = curve_ctrl_rel
    c1_r = (right0, y0 + ctrl * (y1 - y0))
    c2_r = (right1, y1 - ctrl * (y1 - y0))
    c1_l = (left0, y0 + ctrl * (y1 - y0))
    c2_l = (left1, y1 - ctrl * (y1 - y0))

    verts = [
        (right0, y0),
        c1_r, c2_r, (right1, y1),
        (left1, y1),
        c2_l, c1_l, (left0, y0),
        (right0, y0),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY
    ]

    path = Path(verts, codes)

    if not use_gradient:
        ax.add_patch(
            patches.PathPatch(
                path,
                facecolor=hex_to_rgba(color_start, alpha),
                edgecolor="none",
                linewidth=0,
                zorder=1,
            )
        )
    else:
        clip_patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(clip_patch)

        c_s = np.array(hex_to_rgba(color_start, alpha))
        c_e = np.array(hex_to_rgba(color_end, alpha))

        N = 64
        t = np.linspace(0, 1, N)
        gradient = np.zeros((N, 1, 4))
        for i in range(4):
            gradient[:, 0, i] = c_s[i] * (1 - t) + c_e[i] * t

        min_x = min(left0, left1)
        max_x = max(right0, right1)

        im = ax.imshow(
            gradient,
            aspect='auto',
            extent=[min_x, max_x, y0, y1],
            interpolation='bilinear',
            zorder=1
        )
        im.set_clip_path(clip_patch)

    if bool(outline_enabled):
        _draw_link_side_outlines(
            ax,
            (right0, y0), c1_r, c2_r, (right1, y1),
            (left0, y0), c1_l, c2_l, (left1, y1),
            color_hex=outline_color,
            alpha=outline_alpha,
            width_px=outline_width_px,
            dpi=dpi,
        )

    if p_value_text:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(
            mx, my, p_value_text,
            ha='center', va='center',
            fontsize=text_font_size,
            color='black',
            zorder=50
        )

def draw_node(
    ax,
    x, cy, h, w_frac,
    rect_color, node_alpha,
    label: str,
    align: str,
    auto_label_pos: str,
    label_offset_frac: float,
    text_font_size: float,
    show_label: bool,
    text_color: str,
    pos_override: str,
    dx_frac: float,
    dy_frac: float,
    font_size_override: Optional[float],
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    font_family: Optional[str] = None,
    font_zh_family: Optional[str] = None,
    font_en_family: Optional[str] = None,
    renderer=None,
    node_shape: str = "rect",
    label_margin: float = 0.01,
    line_spacing: float = 1.20,
    text_gid: Optional[str] = None,
    outline_enabled: bool = False,
    outline_color: str = "#000000",
    outline_alpha: float = 0.35,
    outline_width_px: float = 1.0,
    dpi: float = 300.0,
    outline_sides: Optional[Tuple[bool, bool, bool, bool]] = None,  # top, right, bottom, left
    vertical_text: bool = False,
    vertical_text_reverse: bool = False,
):
    node_outline_on = bool(outline_enabled) and float(outline_width_px) > 0.0
    node_outline_rgba = hex_to_rgba(normalize_hex_or_default(outline_color, "#000000"), max(0.0, min(1.0, float(outline_alpha))))
    node_outline_lw_pt = (float(outline_width_px) * 72.0 / max(float(dpi), 1.0)) if node_outline_on else 0.0

    # node body
    if node_shape == "circle":
        radius = h / 2
        ax.add_patch(
            patches.Circle(
                (x, cy),
                radius,
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                linewidth=0,
                zorder=9,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x, cy),
                radius,
                facecolor=hex_to_rgba(rect_color, node_alpha),
                edgecolor=(node_outline_rgba if node_outline_on else "none"),
                linewidth=node_outline_lw_pt,
                zorder=10
            )
        )
    else:
        ax.add_patch(
            patches.Rectangle(
                (x - w_frac / 2, cy - h / 2),
                w_frac, h,
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                linewidth=0,
                zorder=9,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x - w_frac / 2, cy - h / 2),
                w_frac, h,
                facecolor=hex_to_rgba(rect_color, node_alpha),
                edgecolor="none",
                linewidth=0,
                zorder=10
            )
        )
        if node_outline_on:
            sides = outline_sides if outline_sides is not None else (True, True, True, True)
            top_on, right_on, bottom_on, left_on = [bool(v) for v in sides]
            x0 = x - w_frac / 2.0
            x1 = x + w_frac / 2.0
            y0 = cy - h / 2.0
            y1 = cy + h / 2.0
            kw = dict(color=node_outline_rgba, linewidth=node_outline_lw_pt, solid_capstyle="butt", zorder=11)
            if top_on:
                ax.plot([x0, x1], [y1, y1], **kw)
            if right_on:
                ax.plot([x1, x1], [y0, y1], **kw)
            if bottom_on:
                ax.plot([x0, x1], [y0, y0], **kw)
            if left_on:
                ax.plot([x0, x0], [y0, y1], **kw)

    if (not show_label) or (pos_override.lower() == "none"):
        return

    fs = font_size_override if font_size_override is not None else text_font_size

    pos = pos_override.lower().strip()
    if pos == "auto":
        if align == "left":
            base_x, base_y, ha, va = x - w_frac / 2 - label_margin, cy, "right", "center"
        elif align == "right":
            base_x, base_y, ha, va = x + w_frac / 2 + label_margin, cy, "left", "center"
        else:
            if auto_label_pos == "below":
                base_x, base_y, ha, va = x, cy - h / 2 - label_offset_frac, "center", "top"
            else:
                base_x, base_y, ha, va = x, cy, "center", "center"
    else:
        if pos == "inside":
            base_x, base_y, ha, va = x, cy, "center", "center"
        elif pos == "below":
            base_x, base_y, ha, va = x, cy - h / 2 - label_offset_frac, "center", "top"
        elif pos == "above":
            base_x, base_y, ha, va = x, cy + h / 2 + label_offset_frac, "center", "bottom"
        elif pos == "left":
            base_x, base_y, ha, va = x - w_frac / 2 - label_margin, cy, "right", "center"
        elif pos == "right":
            base_x, base_y, ha, va = x + w_frac / 2 + label_margin, cy, "left", "center"
        else:
            base_x, base_y, ha, va = x, cy, "center", "center"

    draw_label = _verticalize_label_text(label, reverse=vertical_text_reverse) if vertical_text else label

    txts = _draw_mixed_text(
        ax,
        renderer,
        x=base_x + dx_frac,
        y=base_y + dy_frac,
        text=draw_label,
        ha=ha,
        va=va,
        fs=fs,
        color=text_color,
        weight=("bold" if bold else "normal"),
        style=("italic" if italic else "normal"),
        zh_family=(font_family or font_zh_family),
        en_family=(font_family or font_en_family),
        fallback_family=font_family,
        zorder=20,
        line_spacing=line_spacing,
        gid=text_gid,
    )
    if underline:
        for txt in txts:
            try:
                txt.set_underline(True)
            except Exception:
                pass


# --------------------- Core render ---------------------
def render_sankey_from_df(
    df: pd.DataFrame,
    cfg: SankeyConfig,
    force_auto_fit: bool = False,
    _auto_fit_depth: int = 0,
    _auto_fit_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """
    Render a multi-column Sankey diagram.

    - cfg.orientation == "horizontal" (default): left → right flow
    - cfg.orientation == "vertical": top → bottom flow (canvas width/height swapped)
    """
    plt.rcParams["font.sans-serif"] = [choose_font(cfg.font_priority)]
    plt.rcParams["axes.unicode_minus"] = False

    ori = str(getattr(cfg, "orientation", "horizontal") or "horizontal").lower().strip()
    is_vertical = (ori == "vertical")

    # -------- Auto-fit options --------
    auto_fit_enabled = bool(getattr(cfg, "enable_auto_fit_canvas", False))
    auto_fit_trigger_mode = str(getattr(cfg, "auto_fit_trigger_mode", "manual") or "manual").lower().strip()
    auto_fit_max_iter = max(0, int(getattr(cfg, "auto_fit_max_iter", 8) or 8))
    auto_fit_consider_legend = bool(getattr(cfg, "auto_fit_consider_legend", True))
    auto_fit_consider_link_pct = bool(getattr(cfg, "auto_fit_consider_link_pct", True))
    auto_fit_wanted = bool(auto_fit_enabled and (force_auto_fit or auto_fit_trigger_mode == "before_render"))
    if _auto_fit_state is None:
        _auto_fit_state = {"history": [], "triggered": False, "manual": bool(force_auto_fit)}
    else:
        _auto_fit_state["manual"] = bool(_auto_fit_state.get("manual", False) or force_auto_fit)

    g = build_graph(df, cfg)
    lay = compute_layout(g, cfg)

    # -------- Node legend: hide selected node labels and list them as color swatches on the right --------
    enable_ll = bool(getattr(cfg, "enable_long_label_legend", False))
    ll_threshold = int(getattr(cfg, "long_label_legend_threshold", 0) or 0)
    legend_layout_mode = str(getattr(cfg, "legend_layout_mode", "packed") or "packed").lower().strip()
    legend_title_mode = str(getattr(cfg, "legend_column_title_mode", "letter") or "letter").lower().strip()
    legend_font_size = float(getattr(cfg, "legend_font_size", 16.0) or 16.0)
    legend_force_cols = set()
    for _x in (getattr(cfg, "legend_force_cols", tuple()) or tuple()):
        try:
            ix = int(_x)
        except Exception:
            continue
        if 0 <= ix < int(cfg.n_cols):
            legend_force_cols.add(ix)
    legend_include_auto_hidden = bool(getattr(cfg, "legend_include_auto_hidden", False))
    legend_position = str(getattr(cfg, "legend_position", "right") or "right").lower().strip()
    if legend_position not in ("left", "right", "bottom"):
        legend_position = "right"
    legend_dx_px = float(getattr(cfg, "legend_dx_px", 0.0) or 0.0)
    legend_dy_px = float(getattr(cfg, "legend_dy_px", 0.0) or 0.0)

    index_label_color = normalize_hex_or_default(str(getattr(cfg, "index_label_color", "#4A4A4A") or "#4A4A4A"), "#4A4A4A")
    index_label_font_name = _resolve_index_font_name(cfg)
    index_label_bold = bool(getattr(cfg, "index_label_bold", False))
    index_label_italic = bool(getattr(cfg, "index_label_italic", False))

    legend_font_family = _resolve_global_font_name(cfg)
    zh_font_family, en_font_family = _resolve_zh_en_font_names(cfg)
    header_font_family, header_zh_font_family, header_en_font_family = _resolve_header_font_names(cfg)
    title_font_family, title_zh_font_family, title_en_font_family = _resolve_title_font_names(cfg)
    enable_vertical_node_labels = bool(getattr(cfg, "enable_vertical_node_labels", False))
    vertical_node_label_cols = set()
    for _x in (getattr(cfg, "vertical_node_label_cols", tuple()) or tuple()):
        try:
            ix = int(_x)
        except Exception:
            continue
        if 0 <= ix < int(cfg.n_cols):
            vertical_node_label_cols.add(ix)

    # -------- CJK auto-wrap --------
    enable_cjk_auto_wrap = bool(getattr(cfg, "enable_cjk_auto_wrap", False))
    cjk_wrap_chars_per_line = int(getattr(cfg, "cjk_wrap_chars_per_line", 8) or 8)
    wrap_targets = _normalize_wrap_targets(getattr(cfg, "wrap_targets", ("node_label",)))
    wrap_line_spacing_mult = float(getattr(cfg, "wrap_line_spacing_mult", 1.20) or 1.20)
    wrap_max_lines_raw = getattr(cfg, "wrap_max_lines", None)
    try:
        wrap_max_lines = None if wrap_max_lines_raw in (None, "", 0, "0", "nan") else int(wrap_max_lines_raw)
    except Exception:
        wrap_max_lines = None
    wrap_node_label = bool(enable_cjk_auto_wrap and ("node_label" in wrap_targets))
    wrap_legend_label = bool(enable_cjk_auto_wrap and ("legend_label" in wrap_targets))
    enable_alternate_label_sides = bool(getattr(cfg, "enable_alternate_label_sides", False))
    raw_alt_cols = getattr(cfg, "alternate_label_side_cols", tuple()) or tuple()
    alternate_label_side_cols: set[int] = set()
    for _v in raw_alt_cols:
        try:
            _j = int(_v)
            if 0 <= _j < int(cfg.n_cols):
                alternate_label_side_cols.add(_j)
        except Exception:
            continue
    enable_faux_bold = bool(getattr(cfg, "enable_faux_bold", True))
    faux_bold_width_px = max(0.0, float(getattr(cfg, "faux_bold_width_px", 0.6) or 0.0))

    # -------- Column headers from first row metadata --------
    enable_header_row = bool(getattr(cfg, "enable_header_row", False))
    show_headers = bool(getattr(cfg, "show_headers", True))
    header_pos = str(getattr(cfg, "header_pos", "top") or "top").lower().strip()
    if header_pos not in ("top", "bottom"):
        header_pos = "top"
    header_font_size_cfg = getattr(cfg, "header_font_size", None)
    header_font_size = float(header_font_size_cfg) if header_font_size_cfg not in (None, "", "nan") else float(cfg.text_font_size)
    header_text_color_cfg = getattr(cfg, "header_text_color", None)
    header_text_color_default = normalize_hex_or_default(
        (header_text_color_cfg if header_text_color_cfg not in (None, "") else cfg.label_text_color_default),
        cfg.label_text_color_default,
    )
    header_bold_cfg = getattr(cfg, "header_bold", None)
    header_italic_cfg = getattr(cfg, "header_italic", None)
    header_underline_cfg = getattr(cfg, "header_underline", None)
    header_bold = bool(header_bold_cfg) if header_bold_cfg is not None else False
    header_italic = bool(header_italic_cfg) if header_italic_cfg is not None else False
    header_underline = bool(header_underline_cfg) if header_underline_cfg is not None else False

    # -------- Global plot title --------
    show_title = bool(getattr(cfg, "show_title", False))
    title_text = str(getattr(cfg, "title_text", "") or "").strip()
    title_font_size_cfg = getattr(cfg, "title_font_size", None)
    title_font_size = float(title_font_size_cfg) if title_font_size_cfg not in (None, "", "nan") else float(cfg.text_font_size)
    title_text_color_cfg = getattr(cfg, "title_text_color", None)
    title_text_color = normalize_hex_or_default(
        (title_text_color_cfg if title_text_color_cfg not in (None, "") else cfg.label_text_color_default),
        cfg.label_text_color_default,
    )
    title_bold_cfg = getattr(cfg, "title_bold", None)
    title_italic_cfg = getattr(cfg, "title_italic", None)
    title_underline_cfg = getattr(cfg, "title_underline", None)
    title_bold = bool(title_bold_cfg) if title_bold_cfg is not None else False
    title_italic = bool(title_italic_cfg) if title_italic_cfg is not None else False
    title_underline = bool(title_underline_cfg) if title_underline_cfg is not None else False

    legend_items: List[Tuple[int, str, str, str]] = []  # (col_idx, node_key, full_name, color)

    # -------- Draw canvas --------
    if not is_vertical:
        fig = plt.figure(figsize=(cfg.fig_width_in, cfg.fig_height_in), dpi=cfg.dpi)
    else:
        # Swap width/height to behave like a true 90° rotated diagram.
        fig = plt.figure(figsize=(cfg.fig_height_in, cfg.fig_width_in), dpi=cfg.dpi)

    fig.subplots_adjust(left=0.03, right=0.99, top=0.985, bottom=0.02)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    # Keep a normal matplotlib y-axis (0 bottom → 1 top) so that label positions
    # like 'above'/'below' behave intuitively in vertical mode.
    ax.set_ylim(0, 1)
    ax.axis("off")

    # stage overrides (optional)
    stage_override = build_link_stage_color_overrides(getattr(cfg, "link_stage_override_rows", None))

    # -------- Link % labels (per-segment share) --------
    show_link_pct = bool(getattr(cfg, "show_link_pct", False))
    link_pct_position = str(getattr(cfg, "link_pct_position", "middle") or "middle").lower().strip()
    link_pct_basis = str(getattr(cfg, "link_pct_basis", "raw") or "raw").lower().strip()
    link_pct_format = str(getattr(cfg, "link_pct_format", "{pct100:.1f}%") or "{pct100:.1f}%")
    link_pct_font_size = float(getattr(cfg, "link_pct_font_size", 16.0) or 16.0)
    link_pct_color = normalize_hex_or_default(getattr(cfg, "link_pct_color", "#000000"), "#000000")
    link_pct_bold = bool(getattr(cfg, "link_pct_bold", False))
    link_pct_italic = bool(getattr(cfg, "link_pct_italic", False))
    link_pct_skip_internal_ph = bool(getattr(cfg, "link_pct_skip_internal_placeholder", True))

    # New (v3): aggregate identical src->tgt pairs into ONE label + density auto-hide
    link_pct_aggregate = bool(getattr(cfg, "link_pct_aggregate_same_pair", True))
    link_pct_density_on = bool(getattr(cfg, "link_pct_enable_density_detection", True))
    link_pct_min_sep_px_cfg = getattr(cfg, "link_pct_min_sep_px", None)

    placeholder_set = getattr(g, "placeholder_nodes_set", set()) or set()

    # Convert pixel offsets to Axes fraction in the ACTUAL draw canvas.
    if not is_vertical:
        _draw_w_in = float(cfg.fig_width_in)
        _draw_h_in = float(cfg.fig_height_in)
    else:
        _draw_w_in = float(cfg.fig_height_in)
        _draw_h_in = float(cfg.fig_width_in)

    link_pct_dx_frac = px_to_frac_x(float(getattr(cfg, "link_pct_dx_px", 0.0) or 0.0), _draw_w_in, cfg.dpi)
    link_pct_dy_frac = px_to_frac_y(float(getattr(cfg, "link_pct_dy_px", 0.0) or 0.0), _draw_h_in, cfg.dpi)
    link_side_outline_on = bool(getattr(cfg, "enable_link_side_outline", False))
    link_side_outline_color = normalize_hex_or_default(getattr(cfg, "link_side_outline_color", "#000000"), "#000000")
    link_side_outline_alpha = max(0.0, min(1.0, float(getattr(cfg, "link_side_outline_alpha", 0.35) or 0.35)))
    link_side_outline_width_px = max(0.0, float(getattr(cfg, "link_side_outline_width_px", 1.0) or 0.0))
    node_outline_on = bool(getattr(cfg, "enable_node_outline", False))
    node_outline_color = normalize_hex_or_default(getattr(cfg, "node_outline_color", "#000000"), "#000000")
    node_outline_alpha = max(0.0, min(1.0, float(getattr(cfg, "node_outline_alpha", 0.35) or 0.35)))
    node_outline_width_px = max(0.0, float(getattr(cfg, "node_outline_width_px", 1.0) or 0.0))

    # Precompute per-segment totals and per-(segment, src, tgt) aggregated weights.
    # NOTE: even if we still render individual edges, we label aggregated "merged-looking" links to avoid massive overlap.
    from collections import defaultdict
    pct_total_by_seg: Dict[Tuple[int, int], float] = {}
    pct_group_w: Dict[Tuple, float] = {}
    pct_group_seg: Dict[Tuple, Tuple[int, int]] = {}

    if show_link_pct:
        seg_total = defaultdict(float)
        group_w = defaultdict(float)

        n_edges_tmp = len(g.edge_w_px)
        for _eid in range(n_edges_tmp):
            s = int(g.edge_src_col[_eid])
            t = int(g.edge_tgt_col[_eid])
            if not (0 <= s < cfg.n_cols and 0 <= t < cfg.n_cols):
                continue

            is_s_ph = g.edge_src_node[_eid] in placeholder_set
            is_t_ph = g.edge_tgt_node[_eid] in placeholder_set
            if link_pct_skip_internal_ph and is_s_ph and is_t_ph:
                continue

            w = float(g.edge_w_px[_eid]) if link_pct_basis == "px" else float(g.edge_w_raw[_eid])
            if w <= 0:
                continue

            seg_key = (s, t)
            seg_total[seg_key] += w

            if link_pct_aggregate:
                gkey = (s, t, g.edge_src_node[_eid], g.edge_tgt_node[_eid])
            else:
                gkey = (s, t, _eid)
            group_w[gkey] += w
            pct_group_seg[gkey] = seg_key

        pct_total_by_seg = dict(seg_total)
        pct_group_w = dict(group_w)

    # Accumulate label positions while drawing edges; draw labels after links are rendered.
    # key -> {w_sum, xw_sum, yw_sum, seg_key, x_anchor, y_anchor}
    pct_label_acc: Dict[Tuple, Dict[str, float]] = {}
    pct_label_xy: Dict[Tuple, Tuple[float, float]] = {}



    # In vertical Sankey mode, labels placed above/below a stage reserve real
    # stage-axis space: offset + rendered text height + offset.
    vertical_label_gap_before_col = [0.0 for _ in range(cfg.n_cols)]
    vertical_label_gap_after_col = [0.0 for _ in range(cfg.n_cols)]
    if ori == "vertical":
        for _j in range(cfg.n_cols):
            _cdef = lay.col_label_cfg.get(_j, {})
            _col_show = bool(_cdef.get("show", True))
            if (not bool(cfg.show_labels)) or (not _col_show):
                continue

            _col_pos = str(_cdef.get("pos", "auto"))
            _pos_lower = _col_pos.lower().strip()
            if _pos_lower == "auto":
                if _j == 0:
                    _pos_final = "above"
                elif _j == cfg.n_cols - 1:
                    _pos_final = "below"
                else:
                    _pos_final = "inside"
            else:
                _pos_final = _col_pos

            if bool(enable_vertical_node_labels and (_j in vertical_node_label_cols)):
                continue

            _pos_final_lower = str(_pos_final).lower().strip()
            if _pos_final_lower not in ("above", "below"):
                continue

            _fs = _cdef.get("font_size", None)
            if isinstance(_fs, str) and _fs.lower() == "nan":
                _fs = None
            _fs_for_gap = float(_fs) if _fs is not None else float(cfg.text_font_size)

            _max_lines = 1
            for _node in g.col_nodes[_j]:
                if _node in getattr(g, "placeholder_nodes_set", set()):
                    continue
                _disp = str(g.display_name_by_col[_j].get(_node, _node) or "")
                if not _disp.strip():
                    continue
                if wrap_node_label:
                    _disp = cjk_auto_wrap_text(
                        _disp,
                        chars_per_line=cjk_wrap_chars_per_line,
                        max_lines=wrap_max_lines,
                    )
                _max_lines = max(_max_lines, len(str(_disp).splitlines()) or 1)

            _font_px = max(0.0, _fs_for_gap * float(cfg.dpi) / 72.0)
            _text_h_px = _font_px * (1.0 + max(0, _max_lines - 1) * float(wrap_line_spacing_mult))
            _extra_px = max(0.0, float(cfg.label_offset_px) + _text_h_px + float(cfg.label_offset_px))
            _extra_frac = px_to_frac_x(_extra_px, float(cfg.fig_width_in), cfg.dpi)
            if _pos_final_lower == "above":
                vertical_label_gap_before_col[_j] = _extra_frac
            else:
                vertical_label_gap_after_col[_j] = _extra_frac

    # -------- Draw links (edges) --------
    n_edges = len(g.edge_w_px)
    if cfg.n_cols >= 2 and n_edges > 0:
        edge_ids = list(range(n_edges))
        edge_ids.sort(key=lambda eid: (g.edge_src_col[eid], g.edge_row[eid], g.edge_tgt_col[eid], eid))

        for eid in edge_ids:
            wpx = float(g.edge_w_px[eid])
            if wpx <= 0:
                continue

            s_col = g.edge_src_col[eid]
            t_col = g.edge_tgt_col[eid]
            if not (0 <= s_col < cfg.n_cols and 0 <= t_col < cfg.n_cols):
                continue

            is_src_ph = g.edge_src_node[eid] in getattr(g, "placeholder_nodes_set", set())
            is_tgt_ph = g.edge_tgt_node[eid] in getattr(g, "placeholder_nodes_set", set())

            r_i = g.edge_row[eid]
            src_node = g.edge_src_node[eid]
            tgt_node = g.edge_tgt_node[eid]

            # Use base endpoints for color semantics (so placeholder segments inherit original src/tgt colors)
            b_s_col = g.edge_base_src_col[eid] if hasattr(g, "edge_base_src_col") else s_col
            b_t_col = g.edge_base_tgt_col[eid] if hasattr(g, "edge_base_tgt_col") else t_col
            b_src_node = g.edge_base_src_node[eid] if hasattr(g, "edge_base_src_node") else src_node
            b_tgt_node = g.edge_base_tgt_node[eid] if hasattr(g, "edge_base_tgt_node") else tgt_node

            c_start, c_end = resolve_link_color_pair(
                mode=str(cfg.link_color_mode or "source"),
                follow_col=int(getattr(cfg, "link_color_follow_col", 0) or 0),
                n_cols=cfg.n_cols,
                row_nodes=g.rows_nodes[r_i],
                row_colors=g.rows_colors[r_i],
                col_colors=g.col_colors,
                s_col=b_s_col,
                t_col=b_t_col,
                src_node=b_src_node,
                tgt_node=b_tgt_node,
                stage_override=stage_override,
            )

            hh0 = lay.edge_h_frac_out[eid] if hasattr(lay, "edge_h_frac_out") else lay.edge_h_frac[eid]
            hh1 = lay.edge_h_frac_in[eid] if hasattr(lay, "edge_h_frac_in") else lay.edge_h_frac[eid]

            # Keep legacy p-value text disabled; this renderer uses link percentage labels.
            p_txt = None

            if not is_vertical:
                # horizontal: endpoints are on x, thickness is on y
                y0 = lay.out_band_y[s_col].get(eid)
                y1 = lay.in_band_y[t_col].get(eid)
                if y0 is None or y1 is None:
                    continue

                wL = 0.0 if is_src_ph else lay.node_w_frac_col[s_col]
                wR = 0.0 if is_tgt_ph else lay.node_w_frac_col[t_col]

                dist = lay.x_pos[t_col] - lay.x_pos[s_col]
                avail = max(0.0, dist - (wL / 2 + wR / 2) - 1e-6)
                base_gap = min(lay.gapx0, avail / 2) if avail > 0 else 0.0

                gap_left = 0.0 if is_src_ph else base_gap
                gap_right = 0.0 if is_tgt_ph else base_gap

                cur_x0 = lay.x_pos[s_col] + wL / 2 + gap_left
                cur_x1 = lay.x_pos[t_col] - wR / 2 - gap_right

                draw_trapezoid_link(
                    ax, cur_x0, cur_x1, y0, y1, hh0, hh1,
                    color_start=c_start,
                    color_end=c_end,
                    curve_ctrl_rel=cfg.curve_ctrl_rel,
                    alpha=cfg.link_alpha,
                    outline_enabled=link_side_outline_on,
                    outline_color=link_side_outline_color,
                    outline_alpha=link_side_outline_alpha,
                    outline_width_px=link_side_outline_width_px,
                    dpi=cfg.dpi,
                    p_value_text=p_txt,
                    text_font_size=cfg.p_value_font_size
                )

                # ---- Accumulate link % label position (per-segment share; merged by identical src→tgt) ----
                if show_link_pct:
                    _is_s_ph = is_src_ph
                    _is_t_ph = is_tgt_ph
                    if not (link_pct_skip_internal_ph and _is_s_ph and _is_t_ph):
                        if link_pct_aggregate:
                            _gkey = (int(s_col), int(t_col), src_node, tgt_node)
                        else:
                            _gkey = (int(s_col), int(t_col), int(eid))

                        _w = float(g.edge_w_px[eid]) if link_pct_basis == "px" else float(g.edge_w_raw[eid])
                        _tot = float(pct_total_by_seg.get((int(s_col), int(t_col)), 0.0))
                        if _w > 0 and _tot > 0 and (_gkey in pct_group_w):
                            pos = link_pct_position
                            if pos == "source_right":
                                _x = lay.x_pos[s_col] + wL / 2 + max(0.0, gap_left * 0.5)
                                _y = y0
                            elif pos == "target_left":
                                _x = lay.x_pos[t_col] - wR / 2 - max(0.0, gap_right * 0.5)
                                _y = y1
                            else:
                                _x = (cur_x0 + cur_x1) / 2
                                _y = (y0 + y1) / 2

                            acc = pct_label_acc.get(_gkey)
                            if acc is None:
                                pct_label_acc[_gkey] = {"w": _w, "xw": _x * _w, "yw": _y * _w}
                            else:
                                acc["w"] += _w
                                acc["xw"] += _x * _w
                                acc["yw"] += _y * _w


            else:
                # vertical: endpoints are on y, thickness is on x
                x0 = lay.out_band_y[s_col].get(eid)
                x1 = lay.in_band_y[t_col].get(eid)
                if x0 is None or x1 is None:
                    continue

                wT = 0.0 if is_src_ph else lay.node_w_frac_col[s_col]
                wB = 0.0 if is_tgt_ph else lay.node_w_frac_col[t_col]

                dist = lay.x_pos[t_col] - lay.x_pos[s_col]
                avail = max(0.0, dist - (wT / 2 + wB / 2) - 1e-6)
                base_gap = min(lay.gapx0, avail / 2) if avail > 0 else 0.0

                gap_top = 0.0 if is_src_ph else base_gap + vertical_label_gap_after_col[s_col]
                gap_bot = 0.0 if is_tgt_ph else base_gap + vertical_label_gap_before_col[t_col]
                gap_sum = gap_top + gap_bot
                if avail > 0 and gap_sum > avail:
                    scale = avail / gap_sum
                    gap_top *= scale
                    gap_bot *= scale

                # Map stage-axis (layout uses x_pos in [0,1]) onto y so that stage 0 is on top.
                # Using y = 1 - x_pos keeps the axis non-inverted, which makes label 'above/below' intuitive.
                y_src = lay.x_pos[s_col] + wT / 2 + gap_top
                y_tgt = lay.x_pos[t_col] - wB / 2 - gap_bot
                cur_y0 = 1.0 - y_src
                cur_y1 = 1.0 - y_tgt

                draw_trapezoid_link_vertical(
                    ax, x0, x1, cur_y0, cur_y1, hh0, hh1,
                    color_start=c_start,
                    color_end=c_end,
                    curve_ctrl_rel=cfg.curve_ctrl_rel,
                    alpha=cfg.link_alpha,
                    outline_enabled=link_side_outline_on,
                    outline_color=link_side_outline_color,
                    outline_alpha=link_side_outline_alpha,
                    outline_width_px=link_side_outline_width_px,
                    dpi=cfg.dpi,
                    p_value_text=p_txt,
                    text_font_size=cfg.p_value_font_size
                )

                # ---- Accumulate link % label position (per-segment share; merged by identical src→tgt) [vertical] ----
                if show_link_pct:
                    _is_s_ph = is_src_ph
                    _is_t_ph = is_tgt_ph
                    if not (link_pct_skip_internal_ph and _is_s_ph and _is_t_ph):
                        if link_pct_aggregate:
                            _gkey = (int(s_col), int(t_col), src_node, tgt_node)
                        else:
                            _gkey = (int(s_col), int(t_col), int(eid))

                        _w = float(g.edge_w_px[eid]) if link_pct_basis == "px" else float(g.edge_w_raw[eid])
                        _tot = float(pct_total_by_seg.get((int(s_col), int(t_col)), 0.0))
                        if _w > 0 and _tot > 0 and (_gkey in pct_group_w):
                            pos = link_pct_position
                            if pos == "source_right":
                                _x = x0 + hh0 / 2 + 0.003
                                _y = cur_y0
                            elif pos == "target_left":
                                _x = x1 - hh1 / 2 - 0.003
                                _y = cur_y1
                            else:
                                _x = (x0 + x1) / 2
                                _y = (cur_y0 + cur_y1) / 2

                            acc = pct_label_acc.get(_gkey)
                            if acc is None:
                                pct_label_acc[_gkey] = {"w": _w, "xw": _x * _w, "yw": _y * _w}
                            else:
                                acc["w"] += _w
                                acc["xw"] += _x * _w
                                acc["yw"] += _y * _w





    # -------- Draw link % labels (after links; merged + density-controlled) --------
    if show_link_pct and pct_label_acc and pct_group_w:
        from collections import defaultdict

        candidates_by_seg = defaultdict(list)

        for gkey, acc in pct_label_acc.items():
            wsum = float(acc.get("w", 0.0) or 0.0)
            if wsum <= 0:
                continue

            x = float(acc.get("xw", 0.0) or 0.0) / wsum
            y = float(acc.get("yw", 0.0) or 0.0) / wsum

            seg_key = pct_group_seg.get(gkey)
            if not seg_key:
                continue

            total = float(pct_total_by_seg.get(seg_key, 0.0) or 0.0)
            w_group = float(pct_group_w.get(gkey, 0.0) or 0.0)
            if total <= 0 or w_group <= 0:
                continue

            pct = w_group / total

            # optional: expose src/tgt node names in format string when aggregated
            if link_pct_aggregate and isinstance(gkey, tuple) and len(gkey) >= 4:
                src_node_name = gkey[2]
                tgt_node_name = gkey[3]
            else:
                src_node_name = ""
                tgt_node_name = ""

            try:
                txt = link_pct_format.format(
                    pct=pct,
                    pct100=pct * 100.0,
                    w=w_group,
                    total=total,
                    src_col=int(seg_key[0]),
                    tgt_col=int(seg_key[1]),
                    src_node=src_node_name,
                    tgt_node=tgt_node_name,
                )
            except Exception:
                txt = f"{pct * 100.0:.1f}%"

            txt = str(txt).strip()
            if not txt:
                continue

            coord = y if not is_vertical else x  # thickness axis
            candidates_by_seg[tuple(seg_key)].append({
                "coord": float(coord),
                "x": float(x),
                "y": float(y),
                "w": float(w_group),
                "txt": txt,
            })

        # Density filter (1D greedy) within each segment to avoid over-plotting.
        kept = []
        if link_pct_density_on:
            font_px = float(link_pct_font_size) * (float(cfg.dpi) / 72.0)
            try:
                min_sep_px = float(link_pct_min_sep_px_cfg) if link_pct_min_sep_px_cfg not in (None, "", "nan") else (font_px * 0.9)
            except Exception:
                min_sep_px = font_px * 0.9

            axis_px = float(ax.bbox.height if not is_vertical else ax.bbox.width)
            min_sep = (min_sep_px / axis_px) if axis_px > 1e-9 else 0.0

            for seg_key, lst in candidates_by_seg.items():
                lst2 = sorted(lst, key=lambda d: (-d["w"], d["coord"]))
                selected = []
                for c in lst2:
                    if any(abs(c["coord"] - s["coord"]) < min_sep for s in selected):
                        continue
                    selected.append(c)
                kept.extend(selected)
        else:
            for lst in candidates_by_seg.values():
                kept.extend(lst)

        # alignment depends on chosen position
        if link_pct_position == "source_right":
            ha, va = "left", "center"
        elif link_pct_position == "target_left":
            ha, va = "right", "center"
        else:
            ha, va = "center", "center"

        for c in kept:
            t_pct = ax.text(
                c["x"] + link_pct_dx_frac,
                c["y"] + link_pct_dy_frac,
                c["txt"],
                ha=ha,
                va=va,
                fontsize=float(link_pct_font_size),
                color=link_pct_color,
                fontweight=("bold" if link_pct_bold else "normal"),
                fontstyle=("italic" if link_pct_italic else "normal"),
                fontfamily=legend_font_family,
                zorder=60,
            )
            try:
                t_pct.set_gid("link_pct")
            except Exception:
                pass


    # -------- Draw nodes & labels --------
    try:
        renderer_for_text = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        renderer_for_text = fig.canvas.get_renderer()

    placeholder_nodes = getattr(g, "placeholder_nodes_set", set()) or set()

    def _select_legend_labels_by_density(j: int, *, vertical_mode: bool) -> set:
        if not (enable_ll and legend_include_auto_hidden and (int(j) in legend_force_cols)):
            return set(g.col_nodes[j])
        font_px = float(legend_font_size) * (float(cfg.dpi) / 72.0)
        min_sep_px = font_px * 0.9
        axis_px = float(ax.bbox.width if vertical_mode else ax.bbox.height)
        candidates = []
        for orig_idx, (node, h_frac, center) in enumerate(zip(g.col_nodes[j], lay.col_h_frac[j], lay.col_centers[j])):
            if node in placeholder_nodes:
                continue
            coord_px = float(center) * axis_px
            node_w = float(g.col_h_px[j].get(node, 0.0) or 0.0)
            candidates.append({"name": node, "orig_idx": orig_idx, "y_px": coord_px, "priority": node_w})
        cand_sorted = sorted(candidates, key=lambda c: (-c["priority"], c["orig_idx"]))
        accepted = []
        for c in cand_sorted:
            if any(abs(float(c["y_px"]) - float(a["y_px"])) < min_sep_px for a in accepted):
                continue
            accepted.append(c)
        return set(c["name"] for c in accepted)

    legend_accepted_labels_per_col = [
        _select_legend_labels_by_density(j, vertical_mode=is_vertical)
        for j in range(cfg.n_cols)
    ]

    def _legend_display_name(disp_name: str) -> str:
        if wrap_legend_label and disp_name:
            return cjk_auto_wrap_text(
                disp_name,
                chars_per_line=cjk_wrap_chars_per_line,
                max_lines=wrap_max_lines,
            )
        return str(disp_name or "")

    def _node_uses_right_legend(j: int, node: str, disp_name: str, col_show_label_global: bool) -> bool:
        if (not enable_ll) or (node in placeholder_nodes) or (not str(disp_name or "").strip()):
            return False
        by_long_name = bool(ll_threshold > 0 and len(str(disp_name)) > ll_threshold)
        by_forced_col = (int(j) in legend_force_cols) and (not legend_include_auto_hidden)
        by_auto_hidden = (
            legend_include_auto_hidden
            and (int(j) in legend_force_cols)
            and bool(col_show_label_global)
            and (node not in legend_accepted_labels_per_col[j])
        )
        return bool(by_long_name or by_forced_col or by_auto_hidden)

    if not is_vertical:
        for j in range(cfg.n_cols):
            w_frac = lay.node_w_frac_col[j]

            align = "center"
            if j == 0:
                align = "left"
            elif j == cfg.n_cols - 1:
                align = "right"

            auto_label_pos = "inside"
            if (align == "center") and cfg.label_below_middle:
                auto_label_pos = "below"

            cdef = lay.col_label_cfg.get(j, {})
            col_show = bool(cdef.get("show", True))
            col_pos = str(cdef.get("pos", "auto"))
            col_text_color = str(cdef.get("text_color", cfg.label_text_color_default))
            col_fs = cdef.get("font_size", None)
            if isinstance(col_fs, str) and col_fs.lower() == "nan":
                col_fs = None

            col_use_node_color = bool(cdef.get("use_node_color", False))
            col_bold = bool(cdef.get("bold", False))
            col_italic = bool(cdef.get("italic", False))
            col_underline = bool(cdef.get("underline", False))
            vertical_text_on_col = bool(enable_vertical_node_labels and (j in vertical_node_label_cols))
            alt_on_col = bool(enable_alternate_label_sides and (j in alternate_label_side_cols))
            alt_k = 0

            dx_frac = lay.dx_frac_per_col[j]
            dy_frac = lay.dy_frac_per_col[j]

            col_show_label_global = cfg.show_labels and col_show and (alt_on_col or (col_pos.lower() != "none"))
            # For rectangular nodes with outline enabled, deduplicate shared border when adjacent gap is ~0.
            suppress_top_orig_idx = set()
            if node_outline_on and str(cfg.node_shape).lower() != "circle":
                eps_y = px_to_frac_y(0.5, float(cfg.fig_height_in), cfg.dpi)
                visible_rows = []
                for orig_idx, (node, h_frac, cy) in enumerate(zip(g.col_nodes[j], lay.col_h_frac[j], lay.col_centers[j])):
                    if node in getattr(g, 'placeholder_nodes_set', set()):
                        continue
                    visible_rows.append((orig_idx, float(h_frac), float(cy)))
                for k in range(1, len(visible_rows)):
                    prev_idx, prev_h, prev_cy = visible_rows[k - 1]
                    cur_idx, cur_h, cur_cy = visible_rows[k]
                    prev_bottom = prev_cy - prev_h / 2.0
                    cur_top = cur_cy + cur_h / 2.0
                    if abs(cur_top - prev_bottom) <= eps_y:
                        # keep previous bottom, suppress current top
                        suppress_top_orig_idx.add(cur_idx)

            for orig_idx, (node, h_frac, cy) in enumerate(zip(g.col_nodes[j], lay.col_h_frac[j], lay.col_centers[j])):
                if node in getattr(g, 'placeholder_nodes_set', set()):
                    continue
                pos_override_final = ("right" if (alt_k % 2 == 0) else "left") if alt_on_col else str(col_pos)
                alt_k += 1
                disp_name = g.display_name_by_col[j].get(node, node)
                use_right_legend = _node_uses_right_legend(j, node, disp_name, col_show_label_global)

                if use_right_legend:
                    show_label_final = False
                    label_text = ""
                    text_color_for_node = col_text_color
                    bold_final = False
                    italic_final = False
                    underline_final = False
                    font_family_final = None
                    legend_items.append((j, node, _legend_display_name(disp_name), g.col_colors[j].get(node, "#999999")))
                    text_gid = "node_label"
                else:
                    show_label_final = col_show_label_global and (node in lay.accepted_labels_per_col[j])
                    label_text = (
                        cjk_auto_wrap_text(
                            disp_name,
                            chars_per_line=cjk_wrap_chars_per_line,
                            max_lines=wrap_max_lines,
                        )
                        if (wrap_node_label and disp_name)
                        else disp_name
                    )
                    text_color_for_node = g.col_colors[j].get(node, col_text_color) if col_use_node_color else col_text_color
                    bold_final = col_bold
                    italic_final = col_italic
                    underline_final = col_underline
                    font_family_final = None
                    text_gid = "node_label"

                draw_node(
                    ax=ax,
                    x=lay.x_pos[j],
                    cy=cy,
                    h=h_frac,
                    w_frac=w_frac,
                    rect_color=g.col_colors[j].get(node, "#999999"),
                    node_alpha=cfg.node_alpha,
                    outline_enabled=node_outline_on,
                    outline_color=node_outline_color,
                    outline_alpha=node_outline_alpha,
                    outline_width_px=node_outline_width_px,
                    dpi=cfg.dpi,
                    outline_sides=(False if (orig_idx in suppress_top_orig_idx) else True, True, True, True),
                    label=label_text,
                    align=align,
                    auto_label_pos=auto_label_pos,
                    label_offset_frac=lay.label_offset_frac,
                    text_font_size=cfg.text_font_size,
                    show_label=show_label_final,
                    text_color=text_color_for_node,
                    pos_override=pos_override_final,
                    dx_frac=dx_frac,
                    dy_frac=dy_frac,
                    font_size_override=(float(col_fs) if col_fs is not None else None),
                    bold=bold_final,
                    italic=italic_final,
                    underline=underline_final,
                    font_family=font_family_final,
                    font_zh_family=zh_font_family,
                    font_en_family=en_font_family,
                    renderer=renderer_for_text,
                    node_shape=cfg.node_shape,
                    line_spacing=wrap_line_spacing_mult,
                    text_gid=text_gid,
                    vertical_text=vertical_text_on_col,
                    vertical_text_reverse=False,
                )
    else:
        # In vertical mode:
        # - stage axis is y (based on lay.x_pos)
        # - thickness/value axis is x (based on lay.col_centers / lay.col_h_frac)
        # Layout computed offsets were based on the original canvas; adjust for swapped canvas.
        w_over_h = (cfg.fig_width_in / cfg.fig_height_in) if cfg.fig_height_in > 0 else 1.0
        h_over_w = (cfg.fig_height_in / cfg.fig_width_in) if cfg.fig_width_in > 0 else 1.0

        for j in range(cfg.n_cols):
            # Stage 0 should be at the top in vertical mode.
            stage_y = 1.0 - lay.x_pos[j]

            cdef = lay.col_label_cfg.get(j, {})
            col_show = bool(cdef.get("show", True))
            col_pos = str(cdef.get("pos", "auto"))
            col_text_color = str(cdef.get("text_color", cfg.label_text_color_default))
            col_fs = cdef.get("font_size", None)
            if isinstance(col_fs, str) and col_fs.lower() == "nan":
                col_fs = None

            col_use_node_color = bool(cdef.get("use_node_color", False))
            col_bold = bool(cdef.get("bold", False))
            col_italic = bool(cdef.get("italic", False))
            col_underline = bool(cdef.get("underline", False))
            vertical_text_on_col = bool(enable_vertical_node_labels and (j in vertical_node_label_cols))
            alt_on_col = bool(enable_alternate_label_sides and (j in alternate_label_side_cols))
            alt_k = 0

            # adjust dx/dy because fig width/height are swapped at draw time
            dx_frac = lay.dx_frac_per_col[j] * w_over_h
            dy_frac = lay.dy_frac_per_col[j] * h_over_w
            label_offset_frac = lay.label_offset_frac * h_over_w

            # auto position mapping (vertical):
            # - first stage labels: above
            # - middle stages: inside
            # - last stage labels: below
            pos_lower = col_pos.lower().strip()
            if pos_lower == "auto":
                if j == 0:
                    col_pos_final = "above"
                elif j == cfg.n_cols - 1:
                    col_pos_final = "below"
                else:
                    col_pos_final = "inside"
            else:
                col_pos_final = col_pos

            col_show_label_global = cfg.show_labels and col_show and (alt_on_col or (str(col_pos_final).lower() != "none"))
            suppress_left_orig_idx = set()
            if node_outline_on and str(cfg.node_shape).lower() != "circle":
                # In vertical mode, x-axis fractions are based on swapped draw width (cfg.fig_height_in).
                eps_x = px_to_frac_x(0.5, float(cfg.fig_height_in), cfg.dpi)
                visible_rows = []
                for orig_idx, (node, w_frac_node, cx) in enumerate(zip(g.col_nodes[j], lay.col_h_frac[j], lay.col_centers[j])):
                    if node in getattr(g, 'placeholder_nodes_set', set()):
                        continue
                    visible_rows.append((orig_idx, float(w_frac_node), float(cx)))
                for k in range(1, len(visible_rows)):
                    prev_idx, prev_w, prev_cx = visible_rows[k - 1]
                    cur_idx, cur_w, cur_cx = visible_rows[k]
                    prev_right = prev_cx + prev_w / 2.0
                    cur_left = cur_cx - cur_w / 2.0
                    if abs(cur_left - prev_right) <= eps_x:
                        # keep previous right, suppress current left
                        suppress_left_orig_idx.add(cur_idx)

            for orig_idx, (node, h_frac, cx) in enumerate(zip(g.col_nodes[j], lay.col_h_frac[j], lay.col_centers[j])):
                if node in getattr(g, 'placeholder_nodes_set', set()):
                    continue
                pos_override_final = ("right" if (alt_k % 2 == 0) else "left") if alt_on_col else str(col_pos_final)
                alt_k += 1
                disp_name = g.display_name_by_col[j].get(node, node)
                use_right_legend = _node_uses_right_legend(j, node, disp_name, col_show_label_global)

                if use_right_legend:
                    show_label_final = False
                    label_text = ""
                    text_color_for_node = col_text_color
                    bold_final = False
                    italic_final = False
                    underline_final = False
                    font_family_final = None
                    legend_items.append((j, node, _legend_display_name(disp_name), g.col_colors[j].get(node, "#999999")))
                    text_gid = "node_label"
                else:
                    show_label_final = col_show_label_global and (node in lay.accepted_labels_per_col[j])
                    label_text = (
                        cjk_auto_wrap_text(
                            disp_name,
                            chars_per_line=cjk_wrap_chars_per_line,
                            max_lines=wrap_max_lines,
                        )
                        if (wrap_node_label and disp_name)
                        else disp_name
                    )
                    text_color_for_node = g.col_colors[j].get(node, col_text_color) if col_use_node_color else col_text_color
                    bold_final = col_bold
                    italic_final = col_italic
                    underline_final = col_underline
                    font_family_final = None
                    text_gid = "node_label"

                # vertical bars: width=NodeHeight(h_frac) along x, height=NodeWidth(w_frac) along y
                label_offset_for_node = label_offset_frac

                draw_node(
                    ax=ax,
                    x=cx,
                    cy=stage_y,
                    h=lay.node_w_frac_col[j],
                    w_frac=h_frac,
                    rect_color=g.col_colors[j].get(node, "#999999"),
                    node_alpha=cfg.node_alpha,
                    outline_enabled=node_outline_on,
                    outline_color=node_outline_color,
                    outline_alpha=node_outline_alpha,
                    outline_width_px=node_outline_width_px,
                    dpi=cfg.dpi,
                    outline_sides=(True, True, True, False if (orig_idx in suppress_left_orig_idx) else True),
                    label=label_text,
                    align="center",
                    auto_label_pos="inside",
                    label_offset_frac=label_offset_for_node,
                    text_font_size=cfg.text_font_size,
                    show_label=show_label_final,
                    text_color=text_color_for_node,
                    pos_override=pos_override_final,
                    dx_frac=dx_frac,
                    dy_frac=dy_frac,
                    font_size_override=(float(col_fs) if col_fs is not None else None),
                    bold=bold_final,
                    italic=italic_final,
                    underline=underline_final,
                    font_family=font_family_final,
                    font_zh_family=zh_font_family,
                    font_en_family=en_font_family,
                    renderer=renderer_for_text,
                    node_shape=cfg.node_shape,
                    line_spacing=wrap_line_spacing_mult,
                    text_gid=text_gid,
                    vertical_text=vertical_text_on_col,
                    vertical_text_reverse=False,
                )

    # -------- Draw column headers (from first-row metadata) --------
    if enable_header_row and show_headers:
        try:
            fig.canvas.draw()
            renderer_header = fig.canvas.get_renderer()
        except Exception:
            renderer_header = renderer_for_text

        if not is_vertical:
            _draw_h_in = float(cfg.fig_height_in)
        else:
            _draw_h_in = float(cfg.fig_width_in)
        header_dy_frac = px_to_frac_y(float(getattr(cfg, "header_dy_px", 0.0) or 0.0), _draw_h_in, cfg.dpi)

        top_y_axes = _compute_current_top_y_axes(ax, renderer_header)
        bottom_y_axes = _compute_current_bottom_y_axes(ax, renderer_header)
        _w_h, h_px = _text_wh_px(renderer_header, "Ag", header_font_size, family=(header_font_family or header_en_font_family))
        gap_axes = (h_px / ax.bbox.height) * 0.6 if ax.bbox.height > 0 else 0.02

        if header_pos == "bottom":
            y_base = bottom_y_axes - gap_axes
        else:
            y_base = top_y_axes + gap_axes
        y_final = y_base - header_dy_frac

        for j in range(cfg.n_cols):
            hdr = {}
            if j < len(getattr(g, "col_headers", [])):
                hdr = getattr(g, "col_headers", [])[j] or {}
            txt = str(hdr.get("name", "") or "").strip()
            if not txt:
                continue
            c_raw = str(hdr.get("color", "") or "").strip()
            # Header cell color must be HEX; invalid input falls back to black.
            if c_raw:
                c_use = c_raw if _is_valid_hex_color(c_raw) else "#000000"
            else:
                c_use = header_text_color_default if _is_valid_hex_color(header_text_color_default) else "#000000"

            txts = _draw_mixed_text(
                ax,
                renderer_header,
                x=lay.x_pos[j],
                y=y_final,
                text=txt,
                ha="center",
                va=("top" if header_pos == "bottom" else "bottom"),
                fs=header_font_size,
                color=c_use,
                weight=("bold" if header_bold else "normal"),
                style=("italic" if header_italic else "normal"),
                zh_family=(header_font_family or header_zh_font_family),
                en_family=(header_font_family or header_en_font_family),
                fallback_family=header_font_family,
                transform=ax.transAxes,
                zorder=70,
                line_spacing=wrap_line_spacing_mult,
                gid="header_label",
            )
            if header_underline:
                for tt in txts:
                    try:
                        tt.set_underline(True)
                    except Exception:
                        pass

    # -------- Draw global title --------
    if show_title and title_text:
        try:
            fig.canvas.draw()
            renderer_title = fig.canvas.get_renderer()
        except Exception:
            renderer_title = renderer_for_text

        top_y_axes = _compute_current_top_y_axes(ax, renderer_title)
        _w_t, title_h_px = _text_wh_px(
            renderer_title,
            "Ag",
            title_font_size,
            family=(title_font_family or title_en_font_family),
        )
        gap_axes = (title_h_px / ax.bbox.height) * 0.6 if ax.bbox.height > 0 else 0.02
        dx_axes = (float(getattr(cfg, "title_dx_px", 0.0) or 0.0) / ax.bbox.width) if ax.bbox.width > 0 else 0.0
        dy_axes = (float(getattr(cfg, "title_dy_px", 0.0) or 0.0) / ax.bbox.height) if ax.bbox.height > 0 else 0.0

        txts = _draw_mixed_text(
            ax,
            renderer_title,
            x=0.5 + dx_axes,
            y=top_y_axes + gap_axes - dy_axes,
            text=title_text,
            ha="center",
            va="bottom",
            fs=title_font_size,
            color=title_text_color,
            weight=("bold" if title_bold else "normal"),
            style=("italic" if title_italic else "normal"),
            zh_family=(title_font_family or title_zh_font_family),
            en_family=(title_font_family or title_en_font_family),
            fallback_family=title_font_family,
            transform=ax.transAxes,
            zorder=80,
            line_spacing=wrap_line_spacing_mult,
            gid="plot_title",
        )
        if title_underline:
            for tt in txts:
                try:
                    tt.set_underline(True)
                except Exception:
                    pass

    # -------- Draw right-side color-block node legend --------
    if enable_ll and legend_items:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        _draw_right_color_legend(
            ax,
            renderer,
            items=legend_items,
            fs=legend_font_size,
            position=legend_position,
            dx_px=legend_dx_px,
            dy_px=legend_dy_px,
            family=legend_font_family,
            zh_family=zh_font_family,
            en_family=en_font_family,
            fallback_family=legend_font_family,
            line_spacing=wrap_line_spacing_mult,
        )

    # -------- Faux-bold post process (for fonts without real bold weight) --------
    _apply_faux_bold_to_axis_texts(
        ax,
        enabled=enable_faux_bold,
        width_px=faux_bold_width_px,
        dpi=cfg.dpi,
    )

    # -------- Auto-fit (bbox-driven iterative refine) --------
    fit_metrics: Optional[Dict[str, float]] = None
    if auto_fit_wanted:
        fig.canvas.draw()
        renderer_fit = fig.canvas.get_renderer()
        fit_metrics = _measure_text_overlap_metrics(
            ax,
            renderer_fit,
            consider_legend=auto_fit_consider_legend,
            consider_link_pct=auto_fit_consider_link_pct,
        )
        has_issue = (
            float(fit_metrics.get("overlap_pairs", 0.0)) > 0.0
            or float(fit_metrics.get("overflow_count", 0.0)) > 0.0
        )
        if has_issue and (_auto_fit_depth < auto_fit_max_iter):
            _auto_fit_state["triggered"] = True
            _auto_fit_state["history"].append({
                "iter": int(_auto_fit_depth),
                "metrics": dict(fit_metrics),
                "fig_width_in": float(cfg.fig_width_in),
                "fig_height_in": float(cfg.fig_height_in),
                "default_gap_px": float(cfg.default_gap_px),
                "label_offset_px": float(cfg.label_offset_px),
            })
            next_cfg = _auto_fit_next_cfg(
                cfg,
                fit_metrics,
                iteration=int(_auto_fit_depth + 1),
                is_vertical=bool(is_vertical),
            )
            plt.close(fig)
            return render_sankey_from_df(
                df,
                next_cfg,
                force_auto_fit=bool(_auto_fit_state.get("manual", False)),
                _auto_fit_depth=int(_auto_fit_depth + 1),
                _auto_fit_state=_auto_fit_state,
            )

    pdf_buf = io.BytesIO()
    png_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        png_buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=cfg.dpi,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)

    diag = {
        "n_rows": len(g.rows_nodes),
        "n_cols": cfg.n_cols,
        "n_edges": len(g.edge_w_px),
        "skip_edges": int(getattr(g, "skip_spans_count", 0) or sum(1 for i in range(len(g.edge_w_px)) if (g.edge_tgt_col[i] - g.edge_src_col[i]) > 1)),
        "nodes_per_col": [len(g.col_nodes[j]) for j in range(cfg.n_cols)],
        "scale_applied_to_fit": lay.scale_applied,
        "orientation": ori,
        "fig_size_in": [float(cfg.fig_width_in), float(cfg.fig_height_in)],
        "render_fig_size_in": [float(cfg.fig_height_in), float(cfg.fig_width_in)] if is_vertical else [float(cfg.fig_width_in), float(cfg.fig_height_in)],
        "link_color_mode": str(cfg.link_color_mode or "source"),
        "link_color_follow_col": int(getattr(cfg, "link_color_follow_col", 0) or 0),
        "link_stage_override_rows": list(getattr(cfg, "link_stage_override_rows", None) or []),
        "enable_header_row": bool(getattr(cfg, "enable_header_row", False)),
        "show_headers": bool(getattr(cfg, "show_headers", True)),
        "use_last_col_weight_override": bool(getattr(cfg, "use_last_col_weight_override", False)),
        "enable_cjk_auto_wrap": bool(getattr(cfg, "enable_cjk_auto_wrap", False)),
        "cjk_wrap_chars_per_line": int(getattr(cfg, "cjk_wrap_chars_per_line", 8) or 8),
        "wrap_targets": list(getattr(cfg, "wrap_targets", ("node_label",)) or ()),
        "wrap_line_spacing_mult": float(getattr(cfg, "wrap_line_spacing_mult", 1.20) or 1.20),
        "wrap_max_lines": (
            None
            if getattr(cfg, "wrap_max_lines", None) in (None, "", 0, "0", "nan")
            else int(getattr(cfg, "wrap_max_lines", None))
        ),
        "auto_fit_enabled": bool(getattr(cfg, "enable_auto_fit_canvas", False)),
        "auto_fit_trigger_mode": str(getattr(cfg, "auto_fit_trigger_mode", "manual") or "manual"),
        "auto_fit_iterations": int(_auto_fit_depth),
        "auto_fit_metrics": (fit_metrics or {}),
        "auto_fit_triggered": bool((_auto_fit_state or {}).get("triggered", False)),
        "auto_fit_final_params": {
            "fig_width_in": float(cfg.fig_width_in),
            "fig_height_in": float(cfg.fig_height_in),
            "default_gap_px": float(cfg.default_gap_px),
            "label_offset_px": float(cfg.label_offset_px),
        },
    }
    return pdf_buf.getvalue(), png_buf.getvalue(), diag


def auto_fit_canvas(df: pd.DataFrame, cfg: SankeyConfig) -> Tuple[SankeyConfig, Dict[str, Any]]:
    """
    Convenience entrypoint for manual auto-fit without changing the render API.
    Returns an updated config (fig size/gap/offset) and diagnostics.
    """
    cfg0 = _clone_cfg_for_autofit(cfg)
    cfg0.enable_auto_fit_canvas = True
    _pdf, _png, diag = render_sankey_from_df(df, cfg0, force_auto_fit=True)
    out = _clone_cfg_for_autofit(cfg0)
    p = dict(diag.get("auto_fit_final_params", {}) or {})
    if "fig_width_in" in p:
        out.fig_width_in = float(p["fig_width_in"])
    if "fig_height_in" in p:
        out.fig_height_in = float(p["fig_height_in"])
    if "default_gap_px" in p:
        out.default_gap_px = float(p["default_gap_px"])
    if "label_offset_px" in p:
        out.label_offset_px = float(p["label_offset_px"])
    return out, diag
