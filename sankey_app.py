# app.py
import io
import json
import os
import glob
import hashlib
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from matplotlib import font_manager

from s_engine import (
    SankeyConfig,
    infer_n_cols_from_df,
    render_sankey_from_df,
    scale_config_for_preview,
)


# ---------- 自动注册用户字体目录（解决装在 AppData 里的字体找不到的问题）----------
@st.cache_resource
def _register_user_fonts():
    """将��户本地字体目录（AppData/Local/.../Fonts）里的字体注册到 matplotlib。"""
    user_font_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")
    if not os.path.isdir(user_font_dir):
        return
    existing_fnames = {f.fname for f in font_manager.fontManager.ttflist}
    patterns = ["*.ttf", "*.otf", "*.ttc"]
    added = 0
    for pat in patterns:
        for fp in glob.glob(os.path.join(user_font_dir, pat)):
            if fp not in existing_fnames:
                try:
                    font_manager.fontManager.addfont(fp)
                    added += 1
                except Exception:
                    pass
    return added

_register_user_fonts()


def _safe_rerun():
    """Compatibility rerun across Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # Fallback for Streamlit versions without rerun APIs
        st.stop()


def _font_cfg_to_text(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ",".join([str(x).strip() for x in v if str(x).strip()])
    if isinstance(v, str):
        return v
    return ""


def _safe_hex_color(v: Any, default: str = "#000000") -> str:
    s = str(v or "").strip()
    if s.startswith("#") and len(s) in (4, 7, 9):
        return s
    return str(default)


def _normalize_df_for_signature(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy().fillna("")
    for c in df_norm.columns:
        df_norm[c] = (
            df_norm[c]
            .map(lambda x: str(x))
            .map(lambda s: s.replace("\r\n", "\n").replace("\r", "\n"))
        )
    return df_norm


def _build_col_structure_summary(df: pd.DataFrame, n_cols: int) -> List[Dict[str, Any]]:
    m = int(df.shape[1])
    out: List[Dict[str, Any]] = []
    max_core_col = max(0, 3 * int(n_cols) - 1)  # [name,color,weight] * n - 1
    for i in range(min(m, max_core_col)):
        mod = i % 3
        role = "node_name" if mod == 0 else ("node_color" if mod == 1 else "edge_weight")
        out.append({"col_index": int(i), "role": role})
    if m > max_core_col:
        out.append({"col_index": int(max_core_col), "role": "p_value_or_extra"})
    if m > (max_core_col + 1):
        out.append({"col_index": int(max_core_col + 1), "role": "last_col_weight_override_or_extra"})
    return out


def build_data_signature(df: pd.DataFrame, sheet_name: str, n_cols: int) -> Dict[str, Any]:
    df_norm = _normalize_df_for_signature(df)
    payload = {
        "shape": [int(df_norm.shape[0]), int(df_norm.shape[1])],
        "rows": df_norm.values.tolist(),
    }
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    data_hash = hashlib.sha256(payload_bytes).hexdigest()
    return {
        "sheet_name": str(sheet_name),
        "n_cols": int(n_cols),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "col_structure": _build_col_structure_summary(df, int(n_cols)),
        "data_hash_sha256": data_hash,
    }


def _signature_match(current_sig: Dict[str, Any], template_sig: Dict[str, Any]) -> bool:
    if not isinstance(template_sig, dict):
        return False
    required = ("sheet_name", "n_cols", "shape", "data_hash_sha256")
    if any(k not in template_sig for k in required):
        return False
    return (
        str(current_sig.get("sheet_name", "")) == str(template_sig.get("sheet_name", ""))
        and int(current_sig.get("n_cols", -1)) == int(template_sig.get("n_cols", -2))
        and list(current_sig.get("shape", [])) == list(template_sig.get("shape", []))
        and str(current_sig.get("data_hash_sha256", "")) == str(template_sig.get("data_hash_sha256", ""))
    )


# 常见中文字体的本地化名称（英文内部名 -> 显示名）
_ZH_FONT_NAMES = {
    "SimSun": "宋体", "NSimSun": "新宋体", "SimSun-ExtB": "宋体-ExtB", "SimSun-ExtG": "宋体-ExtG",
    "SimHei": "黑体",
    "Microsoft YaHei": "微软雅黑", "Microsoft YaHei UI": "微软雅黑 UI",
    "KaiTi": "楷体", "KaiTi_GB2312": "楷体_GB2312",
    "FangSong": "仿宋", "FangSong_GB2312": "仿宋_GB2312",
    "DengXian": "等线", "DengXian Light": "等线 Light",
    "LiSu": "隶书", "YouYuan": "幼圆",
    "STSong": "华文宋体", "STZhongsong": "华文中宋", "STKaiti": "华文楷体",
    "STHeiti": "华文黑体", "STFangsong": "华文仿宋", "STXihei": "华文细黑",
    "STXingkai": "华文行楷", "STCaiyun": "华文彩云", "STHupo": "华文琥珀",
    "STLiti": "华文隶书", "STXinwei": "华文新魏",
    "FZShuTi": "方正舒体", "FZYaoti": "方正姚体",
    "Adobe Heiti Std": "Adobe黑体 Std", "Adobe Kaiti Std": "Adobe楷体 Std",
    "Adobe Song Std": "Adobe宋体 Std", "Adobe Fangsong Std": "Adobe仿宋 Std",
}


@st.cache_data(show_spinner="正在加载字体列表…")
def _build_font_display_map() -> dict:
    """
    返回 {英文字体名: 显示名} 的映射。
    优先用 fonttools 读取字体文件里的本地化名（中文字体读中文名）；
    fonttools 不可用时退回静态映射表。
    """
    eng_to_display: dict = {}
    try:
        from fonttools import ttLib  # type: ignore
        # 按文件路径去重，避免重复打开同一字体文件
        seen_files: set = set()
        for fe in font_manager.fontManager.ttflist:
            if fe.fname in seen_files:
                # 同文件不同 weight/style，直接复用已有结果
                if fe.name not in eng_to_display and fe.name in eng_to_display:
                    pass
                seen_files.add(fe.fname)
                continue
            seen_files.add(fe.fname)
            if fe.name in eng_to_display:
                continue
            try:
                tt = ttLib.TTFont(fe.fname, fontNumber=0)
                names_by_lang: dict = {}
                for rec in tt["name"].names:
                    if rec.nameID in (1, 16):   # Family / Preferred Family
                        try:
                            text = rec.toUnicode()
                            if text and rec.langID not in names_by_lang:
                                names_by_lang[rec.langID] = text
                        except Exception:
                            pass
                tt.close()
                # 优先：简体中文(2052) > 繁体中文(1028) > 英文(1033) > 静态表 > 原名
                display = (names_by_lang.get(2052)
                           or names_by_lang.get(1028)
                           or names_by_lang.get(1033)
                           or _ZH_FONT_NAMES.get(fe.name)
                           or fe.name)
                eng_to_display[fe.name] = display
            except Exception:
                eng_to_display[fe.name] = _ZH_FONT_NAMES.get(fe.name, fe.name)
    except ImportError:
        # fonttools 未安装，仅用静态映射表
        for fe in font_manager.fontManager.ttflist:
            if fe.name not in eng_to_display:
                eng_to_display[fe.name] = _ZH_FONT_NAMES.get(fe.name, fe.name)
    return eng_to_display


st.set_page_config(page_title="Sankey Trapezoid Builder", layout="wide")

st.title("丫丫的Sankey生成器")
st.caption("上传 Excel → 调参（含每列 x/宽度/gap + 每列标签颜色/位置）→ 预览 PNG → 导出 PDF+PNG")

# ---------- Upload ----------
uploaded = st.file_uploader("上传 Excel（.xlsx / .xls）", type=["xlsx", "xls"])
if not uploaded:
    st.stop()

excel_bytes = uploaded.getvalue()
xls = pd.ExcelFile(io.BytesIO(excel_bytes))
if ("sheet_select" not in st.session_state) or (st.session_state.get("sheet_select") not in xls.sheet_names):
    st.session_state.sheet_select = xls.sheet_names[0]
sheet = st.selectbox("选择工作表（sheet）", xls.sheet_names, key="sheet_select")

df = pd.read_excel(
    io.BytesIO(excel_bytes),
    sheet_name=sheet,
    header=None,
    keep_default_na=False,
).dropna(how="all")


with st.expander("查看数据预览（前 20 行）", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ---------- Infer N_COLS ----------
guess_n = infer_n_cols_from_df(df)

# ---------- Apply pending JSON config (must run BEFORE sidebar widgets) ----------
# When importing a config JSON, we cannot directly modify st.session_state keys that are bound
# to widgets after those widgets are created. So we stash the JSON in session_state and apply
# it here, before any sidebar widgets instantiate.
if ("_pending_cfg_payload" in st.session_state) or ("_pending_cfg_json" in st.session_state):
    try:
        if "_pending_cfg_payload" in st.session_state:
            _pending_payload = st.session_state.pop("_pending_cfg_payload")
            loaded = (_pending_payload or {}).get("config", {})
            apply_mode = str((_pending_payload or {}).get("mode", "patch") or "patch").lower()
        else:
            loaded = st.session_state.pop("_pending_cfg_json")
            apply_mode = "full"
        apply_mode = "full" if apply_mode not in ("patch", "full") else apply_mode
        is_full_import = (apply_mode == "full")

        # n_cols
        if is_full_import or ("n_cols" in loaded):
            st.session_state.n_cols = int(loaded.get("n_cols", guess_n))
        n_loaded = int(st.session_state.get("n_cols", guess_n))

        # tables
        if is_full_import or ("col_cfg_rows" in loaded):
            col_rows = loaded.get("col_cfg_rows")
            if isinstance(col_rows, list) and len(col_rows) > 0:
                st.session_state.col_table = pd.DataFrame(col_rows)
            elif is_full_import:
                xs = [0.10 + (0.90 - 0.10) * j / (n_loaded - 1) for j in range(n_loaded)] if n_loaded > 1 else [0.50]
                st.session_state.col_table = pd.DataFrame({
                    "col_index": list(range(n_loaded)),
                    "x": xs,
                    "node_width_px": [100.0] * n_loaded,
                    "gap_px": [300.0] * n_loaded,
                    "align": ["center"] * n_loaded,
                    "group_gap_on": [True] * n_loaded,
                    "group_gap_px": [20.0] * n_loaded,
                })

        if is_full_import or ("col_label_cfg_rows" in loaded):
            label_rows = loaded.get("col_label_cfg_rows")
            if isinstance(label_rows, list) and len(label_rows) > 0:
                st.session_state.label_table = pd.DataFrame(label_rows)
            elif is_full_import:
                st.session_state.label_table = pd.DataFrame({
                    "col_index": list(range(n_loaded)),
                    "show": [True] * n_loaded,
                    "pos": ["auto"] * n_loaded,
                    "text_color": ["#000000"] * n_loaded,
                    "dx_px": [0.0] * n_loaded,
                    "dy_px": [0.0] * n_loaded,
                    "font_size": [None] * n_loaded,
                    "use_node_color": [False] * n_loaded,
                    "bold": [False] * n_loaded,
                    "italic": [False] * n_loaded,
                    "underline": [False] * n_loaded,
                })


        # link stage overrides table (per-segment fixed link color)
        if is_full_import or ("link_stage_override_rows" in loaded) or ("link_stage_color_override_rows" in loaded):
            stage_rows = loaded.get("link_stage_override_rows") or loaded.get("link_stage_color_override_rows")
            stage_n = max(0, n_loaded - 1)
            if isinstance(stage_rows, list) and len(stage_rows) > 0:
                st.session_state.link_stage_table = pd.DataFrame(stage_rows)
            elif is_full_import:
                st.session_state.link_stage_table = pd.DataFrame({
                    "stage_index": list(range(stage_n)),
                    "segment": [f"第{si+1}段：第{si+1}列→第{si+2}列" for si in range(stage_n)],
                    "enable": [False] * stage_n,
                    "color": ["#999999"] * stage_n,
                })

        g = loaded.get("global", {}) or {}

        # These keys are bound to widgets; must be set before widgets instantiate.
        if "font_priority" in g:
            _v = g.get("font_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.font_priority_select = _f or "SimSun"
        if "font_zh_priority" in g:
            _v = g.get("font_zh_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.font_zh_priority_select = _f or "（不指定）"
        if "font_en_priority" in g:
            _v = g.get("font_en_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.font_en_priority_select = _f or "（不指定）"
        if "header_font_priority" in g:
            _v = g.get("header_font_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.header_font_priority_select = _f or "（不指定）"
        if "header_font_zh_priority" in g:
            _v = g.get("header_font_zh_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.header_font_zh_priority_select = _f or "（不指定）"
        if "header_font_en_priority" in g:
            _v = g.get("header_font_en_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.header_font_en_priority_select = _f or "（不指定）"
        if "title_font_priority" in g:
            _v = g.get("title_font_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.title_font_priority_select = _f or "（不指定）"
        if "title_font_zh_priority" in g:
            _v = g.get("title_font_zh_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.title_font_zh_priority_select = _f or "（不指定）"
        if "title_font_en_priority" in g:
            _v = g.get("title_font_en_priority", [])
            _f = (str(_v[0]).strip() if isinstance(_v, (list, tuple)) and _v
                  else str(_v).split(",")[0].strip() if isinstance(_v, str) else "")
            st.session_state.title_font_en_priority_select = _f or "（不指定）"
        if "link_color_follow_col" in g:
            st.session_state.link_color_follow_col = int(g.get("link_color_follow_col", 0))
        if "auto_balance_flow" in g:
            st.session_state.auto_balance_flow = bool(g.get("auto_balance_flow", False))
        if "disable_merge_toggle" in g:
            st.session_state.disable_merge_toggle = bool(g.get("disable_merge_toggle", False))
        if "no_merge_cols" in g:
            st.session_state.no_merge_cols = [int(x) for x in (g.get("no_merge_cols", []) or [])]
        if "use_last_col_weight_override" in g:
            st.session_state.use_last_col_weight_override = bool(g.get("use_last_col_weight_override", False))
        if "enable_node_placeholders" in g:
            st.session_state.enable_node_placeholders = bool(g.get("enable_node_placeholders", False))


        # Link % labels (per-segment share)
        if "show_link_pct" in g:
            st.session_state.show_link_pct = bool(g.get("show_link_pct", False))
        if "link_pct_position" in g:
            st.session_state.link_pct_position = str(g.get("link_pct_position", "middle") or "middle")
        if "link_pct_basis" in g:
            st.session_state.link_pct_basis = str(g.get("link_pct_basis", "raw") or "raw")
        if "link_pct_format" in g:
            st.session_state.link_pct_format = str(g.get("link_pct_format", "{pct100:.1f}%") or "{pct100:.1f}%")
        if "link_pct_font_size" in g:
            st.session_state.link_pct_font_size = float(g.get("link_pct_font_size", 16.0) or 16.0)
        if "link_pct_color" in g:
            st.session_state.link_pct_color = str(g.get("link_pct_color", "#000000") or "#000000")
        if "link_pct_bold" in g:
            st.session_state.link_pct_bold = bool(g.get("link_pct_bold", False))
        if "link_pct_italic" in g:
            st.session_state.link_pct_italic = bool(g.get("link_pct_italic", False))
        if "link_pct_dx_px" in g:
            st.session_state.link_pct_dx_px = float(g.get("link_pct_dx_px", 0.0) or 0.0)
        if "link_pct_dy_px" in g:
            st.session_state.link_pct_dy_px = float(g.get("link_pct_dy_px", 0.0) or 0.0)
        if "link_pct_skip_internal_placeholder" in g:
            st.session_state.link_pct_skip_internal_placeholder = bool(g.get("link_pct_skip_internal_placeholder", True))

        if "link_pct_aggregate_same_pair" in g:
            st.session_state.link_pct_aggregate_same_pair = bool(g.get("link_pct_aggregate_same_pair", True))
        if "link_pct_enable_density_detection" in g:
            st.session_state.link_pct_enable_density_detection = bool(g.get("link_pct_enable_density_detection", True))
        if "link_pct_min_sep_px" in g:
            st.session_state.link_pct_min_sep_px = float(g.get("link_pct_min_sep_px", 0.0) or 0.0)
        if "enable_header_row" in g:
            st.session_state.enable_header_row = bool(g.get("enable_header_row", False))
        if "show_headers" in g:
            st.session_state.show_headers = bool(g.get("show_headers", True))
        if "header_pos" in g:
            st.session_state.header_pos = str(g.get("header_pos", "top") or "top")
        if "header_dy_px" in g:
            st.session_state.header_dy_px = float(g.get("header_dy_px", 0.0) or 0.0)
        if "header_font_size" in g:
            _v = g.get("header_font_size", None)
            st.session_state.header_font_size = "" if _v in (None, "") else str(_v)
        if "header_text_color" in g:
            st.session_state.header_text_color = str(g.get("header_text_color", "") or "")
        if "header_bold" in g:
            _v = g.get("header_bold", None)
            st.session_state.header_bold_mode = "继承默认" if _v is None else ("是" if bool(_v) else "否")
        if "header_italic" in g:
            _v = g.get("header_italic", None)
            st.session_state.header_italic_mode = "继承默认" if _v is None else ("是" if bool(_v) else "否")
        if "header_underline" in g:
            _v = g.get("header_underline", None)
            st.session_state.header_underline_mode = "继承默认" if _v is None else ("是" if bool(_v) else "否")
        if "show_title" in g:
            st.session_state.show_title = bool(g.get("show_title", False))
        if "title_text" in g:
            st.session_state.title_text = str(g.get("title_text", "") or "")
        if "title_dx_px" in g:
            st.session_state.title_dx_px = float(g.get("title_dx_px", 0.0) or 0.0)
        if "title_dy_px" in g:
            st.session_state.title_dy_px = float(g.get("title_dy_px", 0.0) or 0.0)
        if "title_font_size" in g:
            _v = g.get("title_font_size", None)
            st.session_state.title_font_size = "" if _v in (None, "") else str(_v)
        if "title_text_color" in g:
            st.session_state.title_text_color = str(g.get("title_text_color", "") or "")
        if "title_bold" in g:
            _v = g.get("title_bold", None)
            st.session_state.title_bold_mode = "继承默认" if _v is None else ("是" if bool(_v) else "否")
        if "title_italic" in g:
            _v = g.get("title_italic", None)
            st.session_state.title_italic_mode = "继承默认" if _v is None else ("是" if bool(_v) else "否")
        if "title_underline" in g:
            _v = g.get("title_underline", None)
            st.session_state.title_underline_mode = "继承默认" if _v is None else ("是" if bool(_v) else "否")

        if "enable_cjk_auto_wrap" in g:
            st.session_state.enable_cjk_auto_wrap = bool(g.get("enable_cjk_auto_wrap", False))
        if "cjk_wrap_chars_per_line" in g:
            st.session_state.cjk_wrap_chars_per_line = int(g.get("cjk_wrap_chars_per_line", 8) or 8)
        if "wrap_targets" in g:
            _wt = g.get("wrap_targets", ["node_label"])
            if isinstance(_wt, str):
                st.session_state.wrap_targets = [x.strip() for x in _wt.split(",") if str(x).strip()]
            else:
                st.session_state.wrap_targets = [str(x).strip() for x in (_wt or []) if str(x).strip()]
        if "wrap_line_spacing_mult" in g:
            st.session_state.wrap_line_spacing_mult = float(g.get("wrap_line_spacing_mult", 1.2) or 1.2)
        if "wrap_max_lines" in g:
            _v = g.get("wrap_max_lines", None)
            st.session_state.wrap_max_lines = 0 if _v in (None, "", "nan") else int(_v)

        if "enable_auto_fit_canvas" in g:
            st.session_state.enable_auto_fit_canvas = bool(g.get("enable_auto_fit_canvas", False))
        if "auto_fit_trigger_mode" in g:
            st.session_state.auto_fit_trigger_mode = str(g.get("auto_fit_trigger_mode", "manual") or "manual")
        if "auto_fit_max_iter" in g:
            st.session_state.auto_fit_max_iter = int(g.get("auto_fit_max_iter", 8) or 8)
        if "auto_fit_prefer_expand_canvas" in g:
            st.session_state.auto_fit_prefer_expand_canvas = bool(g.get("auto_fit_prefer_expand_canvas", True))
        if "auto_fit_consider_legend" in g:
            st.session_state.auto_fit_consider_legend = bool(g.get("auto_fit_consider_legend", True))
        if "auto_fit_consider_link_pct" in g:
            st.session_state.auto_fit_consider_link_pct = bool(g.get("auto_fit_consider_link_pct", True))
        if "orientation" in g:
            st.session_state.orientation = str(g.get("orientation", "horizontal") or "horizontal")
        if "fig_width_in" in g:
            st.session_state.fig_w = float(g.get("fig_width_in", 20.0))
        if "fig_height_in" in g:
            st.session_state.fig_h = float(g.get("fig_height_in", 15.0))
        if "dpi" in g:
            st.session_state.dpi = int(g.get("dpi", 300))
        if "preview_fig_width_in" in g:
            st.session_state.preview_fig_w = float(g.get("preview_fig_width_in", 16.0))
        if "preview_fig_height_in" in g:
            st.session_state.preview_fig_h = float(g.get("preview_fig_height_in", 10.0))
        if "preview_dpi" in g:
            st.session_state.preview_dpi = int(g.get("preview_dpi", 150))
        if "value_to_px" in g:
            st.session_state.value_to_px = float(g.get("value_to_px", 1.0))
        if "use_min_link_thickness" in g:
            st.session_state.use_min_link = bool(g.get("use_min_link_thickness", True))
        if "min_link_px" in g:
            st.session_state.min_link_px = float(g.get("min_link_px", 5.0))
        if "min_node_h_px" in g:
            st.session_state.min_node_h_px = float(g.get("min_node_h_px", 1.0))
        if "force_align_top_bottom" in g:
            st.session_state.force_align = bool(g.get("force_align_top_bottom", False))
        if "force_align_exempt_cols" in g:
            st.session_state.force_align_exempt_cols = [int(x) for x in (g.get("force_align_exempt_cols", []) or [])]
        if "layout_ref_col_index" in g:
            st.session_state.layout_ref_col_index = int(g.get("layout_ref_col_index", -1))
        if "enable_group_gap" in g:
            st.session_state.enable_group_gap = bool(g.get("enable_group_gap", False))
        if "stack_mode" in g:
            st.session_state.stack_mode = str(g.get("stack_mode", "center"))
        if "y_min" in g:
            st.session_state.y_min = float(g.get("y_min", 0.03))
        if "y_max" in g:
            st.session_state.y_max = float(g.get("y_max", 0.97))
        if "x_min" in g:
            st.session_state.x_min = float(g.get("x_min", 0.10))
        if "x_max" in g:
            st.session_state.x_max = float(g.get("x_max", 0.90))
        if "link_node_gap_px" in g:
            st.session_state.link_node_gap_px = float(g.get("link_node_gap_px", 20.0))
        if "curve_ctrl_rel" in g:
            st.session_state.curve_ctrl_rel = float(g.get("curve_ctrl_rel", 0.28))
        if "link_alpha" in g:
            st.session_state.link_alpha = float(g.get("link_alpha", 0.55))
        if "enable_link_side_outline" in g:
            st.session_state.enable_link_side_outline = bool(g.get("enable_link_side_outline", False))
        if "link_side_outline_color" in g:
            st.session_state.link_side_outline_color = _safe_hex_color(g.get("link_side_outline_color", "#000000"), "#000000")
        if "link_side_outline_alpha" in g:
            st.session_state.link_side_outline_alpha = float(g.get("link_side_outline_alpha", 0.35))
        if "link_side_outline_width_px" in g:
            st.session_state.link_side_outline_width_px = float(g.get("link_side_outline_width_px", 1.0))
        if "enable_node_outline" in g:
            st.session_state.enable_node_outline = bool(g.get("enable_node_outline", False))
        if "node_outline_color" in g:
            st.session_state.node_outline_color = _safe_hex_color(g.get("node_outline_color", "#000000"), "#000000")
        if "node_outline_alpha" in g:
            st.session_state.node_outline_alpha = float(g.get("node_outline_alpha", 0.35))
        if "node_outline_width_px" in g:
            st.session_state.node_outline_width_px = float(g.get("node_outline_width_px", 1.0))
        if "link_color_mode" in g:
            st.session_state.link_color_mode = str(g.get("link_color_mode", "source"))
        if "node_alpha" in g:
            st.session_state.node_alpha = float(g.get("node_alpha", 0.70))
        if "show_labels" in g:
            st.session_state.show_labels = bool(g.get("show_labels", True))
        if "text_font_size" in g:
            st.session_state.text_font_size = float(g.get("text_font_size", 20.0))
        if "label_below_middle" in g:
            st.session_state.label_below_middle = bool(g.get("label_below_middle", True))
        if "label_offset_px" in g:
            st.session_state.label_offset_px = float(g.get("label_offset_px", 20.0))
        if "enable_faux_bold" in g:
            st.session_state.enable_faux_bold = bool(g.get("enable_faux_bold", True))
        if "faux_bold_width_px" in g:
            st.session_state.faux_bold_width_px = float(g.get("faux_bold_width_px", 0.6))
        if "enable_alternate_label_sides" in g:
            st.session_state.enable_alternate_label_sides = bool(g.get("enable_alternate_label_sides", False))
        if "alternate_label_side_cols" in g:
            st.session_state.alternate_label_side_cols = [int(x) for x in (g.get("alternate_label_side_cols", []) or [])]
        if "enable_vertical_node_labels" in g:
            st.session_state.enable_vertical_node_labels = bool(g.get("enable_vertical_node_labels", False))
        if "vertical_node_label_cols" in g:
            st.session_state.vertical_node_label_cols = [int(x) for x in (g.get("vertical_node_label_cols", []) or [])]
        if "label_text_color_default" in g:
            st.session_state.label_text_color_default = str(g.get("label_text_color_default", "#000000"))
        if "label_enable_density_detection" in g:
            st.session_state.label_enable_density_detection = bool(g.get("label_enable_density_detection", False))
        if "label_density_cols" in g:
            st.session_state.label_density_cols = [int(x) for x in (g.get("label_density_cols", []) or [])]
        if "label_density_priority" in g:
            st.session_state.label_density_priority = str(g.get("label_density_priority", "height"))
        if "min_gap_px" in g:
            st.session_state.min_gap_px = float(g.get("min_gap_px", 0.0))
        if "enable_long_label_legend" in g:
            st.session_state.enable_long_label_legend = bool(g.get("enable_long_label_legend", False))
        if "long_label_legend_threshold" in g:
            st.session_state.long_label_legend_threshold = int(g.get("long_label_legend_threshold", 30))
        if "legend_force_cols" in g:
            st.session_state.legend_force_cols = [int(x) for x in (g.get("legend_force_cols", []) or [])]
        if "legend_include_auto_hidden" in g:
            st.session_state.legend_include_auto_hidden = bool(g.get("legend_include_auto_hidden", False))
        if "legend_position" in g:
            st.session_state.legend_position = str(g.get("legend_position", "right") or "right")
        if "legend_dx_px" in g:
            st.session_state.legend_dx_px = float(g.get("legend_dx_px", 0.0))
        if "legend_dy_px" in g:
            st.session_state.legend_dy_px = float(g.get("legend_dy_px", 0.0))
        if "legend_font_size" in g:
            st.session_state.legend_font_size = float(g.get("legend_font_size", 16.0))
        if "legend_layout_mode" in g:
            st.session_state.legend_layout_mode = str(g.get("legend_layout_mode", "packed"))
        if "legend_column_title_mode" in g:
            st.session_state.legend_column_title_mode = str(g.get("legend_column_title_mode", "letter"))
        if "index_label_color" in g:
            st.session_state.index_label_color = str(g.get("index_label_color", "#4A4A4A"))
        if "index_label_font" in g:
            st.session_state.index_label_font = str(g.get("index_label_font", ""))
        if "index_label_bold" in g:
            st.session_state.index_label_bold = bool(g.get("index_label_bold", False))
        if "index_label_italic" in g:
            st.session_state.index_label_italic = bool(g.get("index_label_italic", False))
        if "order_mode" in g:
            st.session_state.order_mode = str(g.get("order_mode", "excel"))
        if "order_target_stages" in g:
            st.session_state.order_target_stages = [int(x) for x in (g.get("order_target_stages", []) or [])]
        if "order_keep_ratio" in g:
            st.session_state.order_keep_ratio = float(g.get("order_keep_ratio", 0.35))
        if "flip_cols" in g:
            st.session_state.flip_cols = [int(x) for x in (g.get("flip_cols", []) or [])]
        if "ui_selected_cols_for_node_color" in g:
            st.session_state.ui_selected_cols_for_node_color = [int(x) for x in (g.get("ui_selected_cols_for_node_color", []) or [])]
        st.session_state._cfg_loaded_ok = True
    except Exception as e:
        st.session_state._cfg_loaded_err = str(e)

    _safe_rerun()

if "n_cols" not in st.session_state:
    st.session_state.n_cols = guess_n

_state_defaults = {
    "value_to_px": 1.0,
    "use_min_link": True,
    "min_link_px": 5.0,
    "min_node_h_px": 1.0,
    "force_align": False,
    "force_align_exempt_cols": [],
    "layout_ref_col_index": -1,
    "enable_group_gap": False,
    "stack_mode": "center",
    "y_min": 0.03,
    "y_max": 0.97,
    "x_min": 0.10,
    "x_max": 0.90,
    "link_node_gap_px": 20.0,
    "curve_ctrl_rel": 0.28,
    "link_color_mode": "source",
    "link_alpha": 0.55,
    "enable_link_side_outline": False,
    "link_side_outline_color": "#000000",
    "link_side_outline_alpha": 0.35,
    "link_side_outline_width_px": 1.0,
    "enable_node_outline": False,
    "node_outline_color": "#000000",
    "node_outline_alpha": 0.35,
    "node_outline_width_px": 1.0,
    "node_alpha": 0.70,
    "show_labels": True,
    "text_font_size": 20.0,
    "label_below_middle": True,
    "label_offset_px": 20.0,
    "enable_faux_bold": True,
    "faux_bold_width_px": 0.6,
    "enable_alternate_label_sides": False,
    "alternate_label_side_cols": [],
    "enable_vertical_node_labels": False,
    "vertical_node_label_cols": [],
    "label_text_color_default": "#000000",
    "show_title": False,
    "title_text": "",
    "title_dx_px": 0.0,
    "title_dy_px": 0.0,
    "title_font_size": "",
    "title_text_color": "",
    "enable_long_label_legend": False,
    "long_label_legend_threshold": 30,
    "legend_force_cols": [],
    "legend_include_auto_hidden": False,
    "legend_position": "right",
    "legend_dx_px": 0.0,
    "legend_dy_px": 0.0,
    "legend_font_size": 16.0,
    "legend_layout_mode": "packed",
    "legend_column_title_mode": "letter",
    "index_label_color": "#4A4A4A",
    "index_label_font": "",
    "index_label_bold": False,
    "index_label_italic": False,
    "label_enable_density_detection": False,
    "label_density_cols": [],
    "label_density_priority": "height",
    "min_gap_px": 0.0,
    "ui_selected_cols_for_node_color": [],
    "order_mode": "excel",
    "order_target_stages": [],
    "order_keep_ratio": 0.35,
    "flip_cols": [],
}
for _k, _v in _state_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.sidebar.header("全局参数")

# --- 方向切换（按钮纵向排列）---
if "orientation" not in st.session_state:
    st.session_state.orientation = "horizontal"

st.sidebar.subheader("桑基图方向")
if st.sidebar.button("横向排列", use_container_width=True):
    st.session_state.orientation = "horizontal"
    _safe_rerun()
if st.sidebar.button("竖向排列", use_container_width=True):
    st.session_state.orientation = "vertical"
    _safe_rerun()
st.sidebar.caption(f"当前方向：{'竖向' if st.session_state.orientation == 'vertical' else '横向'}")

n_cols = st.sidebar.number_input(
    "N_COLS（列数）",
    min_value=1,
    max_value=30,
    value=int(st.session_state.n_cols),
    step=1
)
st.session_state.n_cols = int(n_cols)


# ---------- Input enhancements ----------
with st.sidebar.expander("输入增强", expanded=False):
    st.caption("新增：① 指定列不归一（同名节点不合并）；② 末列节点高度可由额外权重列控制。")

    st.checkbox(
        "指定列不归一（同名节点不合并）",
        value=bool(st.session_state.get("disable_merge_toggle", False)),
        key="disable_merge_toggle",
        help="开启后，你选择的列会把同名节点拆成多个独立节点（按行区分），用于避免自动归一/合并。"
    )
    prev_nm = [int(c) for c in st.session_state.get("no_merge_cols", []) if str(c).strip() != ""]
    prev_nm = [c for c in prev_nm if 0 <= c < int(n_cols)]

    no_merge_cols = st.multiselect(
        "不归一的列（可多选）",
        options=list(range(int(n_cols))),
        default=prev_nm,
        disabled=not bool(st.session_state.get("disable_merge_toggle", False)),
        format_func=lambda i: f"第{i+1}列 (stage {i})",
        help="仅对这里勾选的列生效；未勾选的列仍会合并同名节点。"
    )
    st.session_state.no_merge_cols = list(map(int, no_merge_cols))

    st.checkbox(
        "末列节点高度取自额外权重列（在权重列/可选 p 值列后再追加 1 列）",
        value=bool(st.session_state.get("use_last_col_weight_override", True)),
        key="use_last_col_weight_override",
        help="当 Excel 在默认权重列（以及启用 p-value 时的 p-value 列）后面再多一列数值时，"
             "将该列按末列节点汇总，作为末列节点的目标高度（像素=数值×VALUE_TO_PX）。"
             "注意：如果开启了最小线宽，进入末列的线条可能被抬高；节点高度不会小于进入线条总厚度。"
    )


# ---------- Session tables init ----------
def init_col_layout_table(n: int, x_min=0.10, x_max=0.90) -> pd.DataFrame:
    xs = [x_min + (x_max - x_min) * j / (n - 1) for j in range(n)] if n > 1 else [(x_min + x_max) / 2]
    return pd.DataFrame({
        "col_index": list(range(n)),
        "x": xs,
        "node_width_px": [100.0] * n,
        "gap_px": [300.0] * n,
        "align": ["center"] * n,
        # --- 新增列级分组控制 ---
        "group_gap_on": [True] * n,
        "group_gap_px": [20.0] * n,
    })


def init_col_label_table(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "col_index": list(range(n)),
        "show": [True] * n,
        "pos": ["auto"] * n,
        "text_color": ["#000000"] * n,
        "dx_px": [0.0] * n,
        "dy_px": [0.0] * n,
        "font_size": [None] * n,
        "use_node_color": [False] * n,
        "bold": [False] * n,
        "italic": [False] * n,
        "underline": [False] * n,
    })

def init_link_stage_table(n_cols: int) -> pd.DataFrame:
    stage_n = max(0, int(n_cols) - 1)
    return pd.DataFrame({
        "stage_index": list(range(stage_n)),
        "segment": [f"第{si+1}段：第{si+1}列→第{si+2}列" for si in range(stage_n)],
        "enable": [False] * stage_n,
        "color": ["#999999"] * stage_n,
    })



if "col_table" not in st.session_state or len(st.session_state.col_table) != n_cols:
    st.session_state.col_table = init_col_layout_table(n_cols)

if "label_table" not in st.session_state or len(st.session_state.label_table) != n_cols:
    st.session_state.label_table = init_col_label_table(n_cols)

# ---------- Link stage override table init ----------
stage_n = max(0, int(n_cols) - 1)
if "link_stage_table" not in st.session_state:
    st.session_state.link_stage_table = init_link_stage_table(n_cols)
elif len(st.session_state.link_stage_table) != stage_n:
    # Resize while preserving existing settings by stage_index
    old = st.session_state.link_stage_table.copy()
    new = init_link_stage_table(n_cols)
    try:
        for _, rr in old.iterrows():
            try:
                stg = int(rr.get("stage_index", 0))
            except Exception:
                continue
            if 0 <= stg < stage_n:
                new.loc[new["stage_index"] == stg, "enable"] = bool(rr.get("enable", False))
                new.loc[new["stage_index"] == stg, "color"] = str(rr.get("color", "#999999"))
    except Exception:
        pass
    st.session_state.link_stage_table = new


# ---------- Sidebar controls ----------

with st.sidebar.expander("字体", expanded=False):
    all_fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
    font_opts_optional = ["（不指定）"] + all_fonts
    _font_display = _build_font_display_map()
    # format_func：下拉列表显示本地化名（中文字体→中文，英文字体→英文）
    def _fmt_font(eng: str) -> str:
        native = _font_display.get(eng, eng)
        return native if native == eng else f"{native}  ({eng})"
    def _fmt_font_opt(eng: str) -> str:
        if eng == "（不指定）":
            return "（不指定）"
        return _fmt_font(eng)

    st.caption("提示：点击下拉框可直接输入字体名称搜索（支持中文名或英文名）；字体须已安装在运行环境中。")

    # 初始化主字体
    if "font_priority_select" not in st.session_state:
        st.session_state.font_priority_select = "SimSun" if "SimSun" in all_fonts else (all_fonts[0] if all_fonts else "")
    elif st.session_state.font_priority_select not in all_fonts and all_fonts:
        st.session_state.font_priority_select = all_fonts[0]

    # 初始化中文/英文字体（可不指定）
    if "font_zh_priority_select" not in st.session_state:
        st.session_state.font_zh_priority_select = "（不指定）"
    elif st.session_state.font_zh_priority_select not in font_opts_optional:
        st.session_state.font_zh_priority_select = "（不指定）"

    if "font_en_priority_select" not in st.session_state:
        st.session_state.font_en_priority_select = "（不指定）"
    elif st.session_state.font_en_priority_select not in font_opts_optional:
        st.session_state.font_en_priority_select = "（不指定）"

    sel_font = st.selectbox(
        "通用字体（主字体）",
        options=all_fonts,
        key="font_priority_select",
        format_func=_fmt_font,
        help="支持输入中文名（如：宋体）或英文名（如：SimSun）搜索",
    )
    sel_zh = st.selectbox(
        "中文字体（可选，优先于主字体）",
        options=font_opts_optional,
        key="font_zh_priority_select",
        format_func=_fmt_font_opt,
        help="仅用于中文字符的专用字体；选 '（不指定）' 则回退到主字体",
    )
    sel_en = st.selectbox(
        "英文字体（可选，优先于主字体）",
        options=font_opts_optional,
        key="font_en_priority_select",
        format_func=_fmt_font_opt,
        help="仅用于英文字符的专用字体；选 '（不指定）' 则回退到主字体",
    )

    font_priority = (sel_font,) if sel_font else ("SimSun",)
    font_zh_priority = () if sel_zh == "（不指定）" else (sel_zh,)
    font_en_priority = () if sel_en == "（不指定）" else (sel_en,)

with st.sidebar.expander("表头（首行元数据）", expanded=False):
    if "enable_header_row" not in st.session_state:
        st.session_state.enable_header_row = False
    if "show_headers" not in st.session_state:
        st.session_state.show_headers = True
    if "header_pos" not in st.session_state:
        st.session_state.header_pos = "top"
    if "header_dy_px" not in st.session_state:
        st.session_state.header_dy_px = 0.0
    if "header_font_size" not in st.session_state:
        st.session_state.header_font_size = ""
    if "header_text_color" not in st.session_state:
        st.session_state.header_text_color = ""
    if "header_font_priority_select" not in st.session_state:
        st.session_state.header_font_priority_select = "（不指定）"
    elif st.session_state.header_font_priority_select not in font_opts_optional:
        st.session_state.header_font_priority_select = "（不指定）"
    if "header_font_zh_priority_select" not in st.session_state:
        st.session_state.header_font_zh_priority_select = "（不指定）"
    elif st.session_state.header_font_zh_priority_select not in font_opts_optional:
        st.session_state.header_font_zh_priority_select = "（不指定）"
    if "header_font_en_priority_select" not in st.session_state:
        st.session_state.header_font_en_priority_select = "（不指定）"
    elif st.session_state.header_font_en_priority_select not in font_opts_optional:
        st.session_state.header_font_en_priority_select = "（不指定）"
    if "header_bold_mode" not in st.session_state:
        st.session_state.header_bold_mode = "继承默认"
    if "header_italic_mode" not in st.session_state:
        st.session_state.header_italic_mode = "继承默认"
    if "header_underline_mode" not in st.session_state:
        st.session_state.header_underline_mode = "继承默认"

    st.checkbox(
        "启用首行表头（第1行仅作为每列标题与颜色，不参与流图计算）",
        key="enable_header_row",
        value=bool(st.session_state.get("enable_header_row", False)),
    )
    st.checkbox(
        "显示表头文字（独立于 show_labels）",
        key="show_headers",
        value=bool(st.session_state.get("show_headers", True)),
    )
    st.selectbox(
        "表头位置",
        options=["top", "bottom"],
        key="header_pos",
        format_func=lambda x: "上方" if x == "top" else "下方",
    )
    st.number_input(
        "表头上下偏移（px，负=上，正=下）",
        key="header_dy_px",
        value=float(st.session_state.get("header_dy_px", 0.0) or 0.0),
        step=1.0,
    )

    st.caption("样式覆盖（留空/不指定即继承标签默认设置）")
    st.text_input(
        "表头字号（留空=继承 TEXT_FONT_SIZE）",
        key="header_font_size",
        value=str(st.session_state.get("header_font_size", "") or ""),
    )
    st.text_input(
        "表头文字颜色（hex，留空=继承默认文字颜色）",
        key="header_text_color",
        value=str(st.session_state.get("header_text_color", "") or ""),
    )

    st.selectbox(
        "表头通用字体（可选）",
        options=font_opts_optional,
        key="header_font_priority_select",
        format_func=_fmt_font_opt,
    )
    st.selectbox(
        "表头中文字体（可选）",
        options=font_opts_optional,
        key="header_font_zh_priority_select",
        format_func=_fmt_font_opt,
    )
    st.selectbox(
        "表头英文字体（可选）",
        options=font_opts_optional,
        key="header_font_en_priority_select",
        format_func=_fmt_font_opt,
    )
    st.selectbox("表头加粗", options=["继承默认", "是", "否"], key="header_bold_mode")
    st.selectbox("表头斜体", options=["继承默认", "是", "否"], key="header_italic_mode")
    st.selectbox("表头下划线", options=["继承默认", "是", "否"], key="header_underline_mode")

    header_font_priority = () if st.session_state.header_font_priority_select == "（不指定）" else (st.session_state.header_font_priority_select,)
    header_font_zh_priority = () if st.session_state.header_font_zh_priority_select == "（不指定）" else (st.session_state.header_font_zh_priority_select,)
    header_font_en_priority = () if st.session_state.header_font_en_priority_select == "（不指定）" else (st.session_state.header_font_en_priority_select,)

    _hfs = str(st.session_state.get("header_font_size", "") or "").strip()
    try:
        header_font_size = float(_hfs) if _hfs != "" else None
    except Exception:
        header_font_size = None
    header_text_color = str(st.session_state.get("header_text_color", "") or "").strip() or None

    _mode_map = {"继承默认": None, "是": True, "否": False}
    header_bold = _mode_map.get(str(st.session_state.get("header_bold_mode", "继承默认")), None)
    header_italic = _mode_map.get(str(st.session_state.get("header_italic_mode", "继承默认")), None)
    header_underline = _mode_map.get(str(st.session_state.get("header_underline_mode", "继承默认")), None)

with st.sidebar.expander("标题", expanded=False):
    if "title_font_priority_select" not in st.session_state:
        st.session_state.title_font_priority_select = "（不指定）"
    elif st.session_state.title_font_priority_select not in font_opts_optional:
        st.session_state.title_font_priority_select = "（不指定）"
    if "title_font_zh_priority_select" not in st.session_state:
        st.session_state.title_font_zh_priority_select = "（不指定）"
    elif st.session_state.title_font_zh_priority_select not in font_opts_optional:
        st.session_state.title_font_zh_priority_select = "（不指定）"
    if "title_font_en_priority_select" not in st.session_state:
        st.session_state.title_font_en_priority_select = "（不指定）"
    elif st.session_state.title_font_en_priority_select not in font_opts_optional:
        st.session_state.title_font_en_priority_select = "（不指定）"
    if "title_bold_mode" not in st.session_state:
        st.session_state.title_bold_mode = "继承默认"
    if "title_italic_mode" not in st.session_state:
        st.session_state.title_italic_mode = "继承默认"
    if "title_underline_mode" not in st.session_state:
        st.session_state.title_underline_mode = "继承默认"

    st.checkbox("启用标题", key="show_title", value=bool(st.session_state.get("show_title", False)))
    st.text_input("标题文本", key="title_text", value=str(st.session_state.get("title_text", "") or ""))
    st.number_input(
        "标题水平偏移（px，负=左，正=右）",
        key="title_dx_px",
        value=float(st.session_state.get("title_dx_px", 0.0) or 0.0),
        step=1.0,
    )
    st.number_input(
        "标题垂直偏移（px，负=上，正=下）",
        key="title_dy_px",
        value=float(st.session_state.get("title_dy_px", 0.0) or 0.0),
        step=1.0,
    )

    st.caption("样式覆盖（留空/不指定即继承标签默认设置）")
    st.text_input(
        "标题字号（留空=继承 TEXT_FONT_SIZE）",
        key="title_font_size",
        value=str(st.session_state.get("title_font_size", "") or ""),
    )
    st.text_input(
        "标题文字颜色（hex，留空=继承默认文字颜色）",
        key="title_text_color",
        value=str(st.session_state.get("title_text_color", "") or ""),
    )
    st.selectbox(
        "标题通用字体（可选）",
        options=font_opts_optional,
        key="title_font_priority_select",
        format_func=_fmt_font_opt,
    )
    st.selectbox(
        "标题中文字体（可选）",
        options=font_opts_optional,
        key="title_font_zh_priority_select",
        format_func=_fmt_font_opt,
    )
    st.selectbox(
        "标题英文字体（可选）",
        options=font_opts_optional,
        key="title_font_en_priority_select",
        format_func=_fmt_font_opt,
    )
    st.selectbox("标题加粗", options=["继承默认", "是", "否"], key="title_bold_mode")
    st.selectbox("标题斜体", options=["继承默认", "是", "否"], key="title_italic_mode")
    st.selectbox("标题下划线", options=["继承默认", "是", "否"], key="title_underline_mode")

    title_font_priority = () if st.session_state.title_font_priority_select == "（不指定）" else (st.session_state.title_font_priority_select,)
    title_font_zh_priority = () if st.session_state.title_font_zh_priority_select == "（不指定）" else (st.session_state.title_font_zh_priority_select,)
    title_font_en_priority = () if st.session_state.title_font_en_priority_select == "（不指定）" else (st.session_state.title_font_en_priority_select,)

    _tfs = str(st.session_state.get("title_font_size", "") or "").strip()
    try:
        title_font_size = float(_tfs) if _tfs != "" else None
    except Exception:
        title_font_size = None
    title_text_color = str(st.session_state.get("title_text_color", "") or "").strip() or None
    title_bold = _mode_map.get(str(st.session_state.get("title_bold_mode", "继承默认")), None)
    title_italic = _mode_map.get(str(st.session_state.get("title_italic_mode", "继承默认")), None)
    title_underline = _mode_map.get(str(st.session_state.get("title_underline_mode", "继承默认")), None)

with st.sidebar.expander("比例 / 线宽", expanded=False):
    value_to_px = st.number_input("VALUE_TO_PX（数值→像素）", key="value_to_px", value=float(st.session_state.get("value_to_px", 1.0)), min_value=0.0, step=0.1)
    use_min_link = st.checkbox("启用最小线宽（USE_MIN_LINK_THICKNESS）", key="use_min_link", value=bool(st.session_state.get("use_min_link", True)))
    min_link_px = st.number_input("MIN_LINK_PX", key="min_link_px", value=float(st.session_state.get("min_link_px", 5.0)), min_value=0.0, step=1.0)
    min_node_h_px = st.number_input("MIN_NODE_H_PX", key="min_node_h_px", value=float(st.session_state.get("min_node_h_px", 1.0)), min_value=0.0, step=1.0)

with st.sidebar.expander("布局", expanded=False):
    force_align = st.checkbox(
        "FORCE_ALIGN_TOP_BOTTOM（同顶同底）",
        key="force_align",
        value=bool(st.session_state.get("force_align", False)),
        help="覆盖自定义对齐设置。可通过“同顶同底豁免列”让少数列改回列级对齐与间距。",
    )

    _fa_prev = [int(c) for c in st.session_state.get("force_align_exempt_cols", []) if str(c).strip() != ""]
    _fa_prev = [c for c in _fa_prev if 0 <= c < int(n_cols)]
    if _fa_prev != st.session_state.get("force_align_exempt_cols", []):
        st.session_state.force_align_exempt_cols = _fa_prev

    force_align_exempt_cols = st.multiselect(
        "同顶同底豁免列（这些列改走列级 align/gap）",
        options=list(range(int(n_cols))),
        key="force_align_exempt_cols",
        default=_fa_prev,
        disabled=not bool(force_align),
        format_func=lambda i: f"第{i+1}列 (stage {i})",
        help="仅在开启同顶同底时生效。被豁免列不再做同顶同底 gap 拉伸，改按 COL_CFG 的 align/gap_px/group_gap_px。",
    )

    layout_ref_col_index = st.number_input(
        "布局基准列索引 (-1=自动最高列)",
        key="layout_ref_col_index",
        min_value=-1,
        max_value=n_cols - 1,
        value=int(st.session_state.get("layout_ref_col_index", -1)),
        step=1
    )

    # --- 分组间隙全局开关 ---
    st.markdown("---")
    enable_group_gap = st.checkbox(
        "启用同源节点分组 (Global Enable)",
        key="enable_group_gap",
        value=bool(st.session_state.get("enable_group_gap", False)),
        help="开启后，允许根据来源对节点进行分组。具体每列的间隙大小请在右侧表格中【group_gap_px】列设置。"
    )
    # 注意：group_gap_px 全局设置已移除，改由表格控制
    st.markdown("---")


    enable_node_placeholders = st.checkbox(
        "启用节点占位（空白列插入不可见节点）",
        value=bool(st.session_state.get("enable_node_placeholders", False)),
        key="enable_node_placeholders",
        help=("开启：跨越空白列的连线会被拆成相邻连线，并在空白列插入不可见占位节点，让布局把空白列当作真实分隔。\n关闭：保持原逻辑，连线可跨越空白列，不插入占位节点。")
    )

    stack_mode = st.selectbox("STACK_MODE", ["top", "center", "bottom"], key="stack_mode", index=["top", "center", "bottom"].index(str(st.session_state.get("stack_mode", "center")) if str(st.session_state.get("stack_mode", "center")) in ["top", "center", "bottom"] else "center"))
    y_min = st.number_input("Y_MIN", key="y_min", value=float(st.session_state.get("y_min", 0.03)), min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
    y_max = st.number_input("Y_MAX", key="y_max", value=float(st.session_state.get("y_max", 0.97)), min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
    x_min = st.number_input("X_MIN", key="x_min", value=float(st.session_state.get("x_min", 0.10)), min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
    x_max = st.number_input("X_MAX", key="x_max", value=float(st.session_state.get("x_max", 0.90)), min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
    link_node_gap_px = st.number_input("LINK_NODE_GAP_PX", key="link_node_gap_px", value=float(st.session_state.get("link_node_gap_px", 20.0)), min_value=0.0, step=5.0)
    curve_ctrl_rel = st.slider("CURVE_CTRL_REL（弯曲度）", min_value=0.0, max_value=0.5, key="curve_ctrl_rel", value=float(st.session_state.get("curve_ctrl_rel", 0.28)), step=0.01)

# ==============================
# 线条样式 (Color Mode)
# ==============================
with st.sidebar.expander("线条样式 (Links)", expanded=False):
    link_color_mode_ui = st.selectbox(
        "线条颜色模式",
        ["跟随源节点 (Source)", "跟随目标节点 (Target)", "渐变 (Gradient)", "跟随指定列节点 (Follow Column)"],
        key="link_color_mode_ui",
        index=0 if str(st.session_state.get("link_color_mode", "source")) == "source" else (1 if str(st.session_state.get("link_color_mode", "source")) == "target" else (2 if str(st.session_state.get("link_color_mode", "source")) == "gradient" else 3)),
        help="Gradient 模式会从源节点颜色渐变过渡到目标节点颜色；Follow Column 会让所有线条统一跟随你选定列的节点颜色。"
    )
    link_color_mode_map = {
        "跟随源节点 (Source)": "source",
        "跟随目标节点 (Target)": "target",
        "渐变 (Gradient)": "gradient",
        "跟随指定列节点 (Follow Column)": "follow_col"
    }
    link_color_mode = link_color_mode_map[link_color_mode_ui]
    st.session_state.link_color_mode = link_color_mode

    # Follow Column: choose which column's node color to apply to ALL links
    if "link_color_follow_col" not in st.session_state:
        st.session_state.link_color_follow_col = 0
    if link_color_mode == "follow_col":
        st.selectbox(
            "跟随哪一列的节点颜色",
            options=list(range(int(n_cols))),
            format_func=lambda i: f"第{i+1}列 (stage {i})",
            index=int(min(max(st.session_state.link_color_follow_col, 0), int(n_cols) - 1)) if int(n_cols) > 0 else 0,
            key="link_color_follow_col",
            help="开启后，所有线条颜色都将使用你选定列的节点颜色（整条线同色）。"
        )

    # Visual taper mode: keep node heights unchanged; links taper between ends.
    st.checkbox(
        "固定节点高度 + 连线渐变",
        value=bool(st.session_state.get("auto_balance_flow", False)),
        key="auto_balance_flow",
        help="启用后仅改变连线两端厚度与过渡形态，不会改动节点高度或边的基础权重。",
    )

    st.markdown("---")
    st.checkbox(
        "启用连线两侧描边",
        value=bool(st.session_state.get("enable_link_side_outline", False)),
        key="enable_link_side_outline",
        help="仅绘制连线厚度方向的两侧边线（不绘制两端封口）。",
    )
    st.color_picker(
        "描边颜色",
        value=_safe_hex_color(st.session_state.get("link_side_outline_color", "#000000"), "#000000"),
        key="link_side_outline_color",
    )
    st.slider(
        "描边透明度",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("link_side_outline_alpha", 0.35)),
        step=0.01,
        key="link_side_outline_alpha",
    )
    st.number_input(
        "描边粗细（px）",
        min_value=0.0,
        value=float(st.session_state.get("link_side_outline_width_px", 1.0)),
        step=0.1,
        key="link_side_outline_width_px",
    )

link_alpha = st.slider("LINK_ALPHA (透明度)", 0.0, 1.0, float(st.session_state.get("link_alpha", 0.55)), 0.01, key="link_alpha")

# ==============================
# 线段百分比（每段内占比）
# ==============================
with st.sidebar.expander("线段百分比（每段内占比）", expanded=False):
    st.caption("按每一段（同一对列 src→tgt）归一：每段内所有 link 总和=100%。")

    # Defaults (in case JSON import didn't set them)
    if "show_link_pct" not in st.session_state:
        st.session_state.show_link_pct = False
    if "link_pct_position" not in st.session_state:
        st.session_state.link_pct_position = "middle"
    if "link_pct_basis" not in st.session_state:
        st.session_state.link_pct_basis = "raw"
    if "link_pct_format" not in st.session_state:
        st.session_state.link_pct_format = "{pct100:.1f}%"
    if "link_pct_font_size" not in st.session_state:
        st.session_state.link_pct_font_size = 16.0
    if "link_pct_color" not in st.session_state:
        st.session_state.link_pct_color = "#000000"
    if "link_pct_bold" not in st.session_state:
        st.session_state.link_pct_bold = False
    if "link_pct_italic" not in st.session_state:
        st.session_state.link_pct_italic = False
    if "link_pct_dx_px" not in st.session_state:
        st.session_state.link_pct_dx_px = 0.0
    if "link_pct_dy_px" not in st.session_state:
        st.session_state.link_pct_dy_px = 0.0
    if "link_pct_skip_internal_placeholder" not in st.session_state:
        st.session_state.link_pct_skip_internal_placeholder = True

    if "link_pct_aggregate_same_pair" not in st.session_state:
        st.session_state.link_pct_aggregate_same_pair = True
    if "link_pct_enable_density_detection" not in st.session_state:
        st.session_state.link_pct_enable_density_detection = True
    if "link_pct_min_sep_px" not in st.session_state:
        st.session_state.link_pct_min_sep_px = 0.0  # 0=auto

    st.checkbox(
        "显示占比（每段内）",
        key="show_link_pct",
        value=bool(st.session_state.get("show_link_pct", False)),
        help="每条线显示它在【同一段（同一对列）】内的比例。",
    )

    pos_opts = ["source_right", "middle", "target_left"]
    pos_labels = {
        "source_right": "放在 source 节点右边",
        "middle": "放在线段中间",
        "target_left": "放在 target 节点左边",
    }
    cur_pos = str(st.session_state.get("link_pct_position", "middle") or "middle")
    if cur_pos not in pos_opts:
        cur_pos = "middle"
    st.selectbox(
        "显示位置",
        options=pos_opts,
        index=pos_opts.index(cur_pos),
        key="link_pct_position",
        format_func=lambda k: pos_labels.get(k, k),
    )

    basis_opts = ["raw", "px"]
    basis_labels = {
        "raw": "按原始权重（raw）",
        "px": "按绘制线宽（px，受最小线宽/平衡影响）",
    }
    cur_basis = str(st.session_state.get("link_pct_basis", "raw") or "raw")
    if cur_basis not in basis_opts:
        cur_basis = "raw"
    st.selectbox(
        "占比计算口径",
        options=basis_opts,
        index=basis_opts.index(cur_basis),
        key="link_pct_basis",
        format_func=lambda k: basis_labels.get(k, k),
    )

    st.text_input(
        "显示格式（Python format）",
        key="link_pct_format",
        value=str(st.session_state.get("link_pct_format", "{pct100:.1f}%")),
        help="可用变量：{pct} (0..1), {pct100} (0..100), {w}, {total}, {src_col}, {tgt_col}。",
    )

    st.number_input(
        "字号",
        key="link_pct_font_size",
        value=float(st.session_state.get("link_pct_font_size", 16.0) or 16.0),
        min_value=1.0,
        step=1.0,
    )

    st.color_picker(
        "颜色",
        key="link_pct_color",
        value=str(st.session_state.get("link_pct_color", "#000000") or "#000000"),
    )

    b1, b2 = st.columns(2)
    with b1:
        st.checkbox("加粗", key="link_pct_bold", value=bool(st.session_state.get("link_pct_bold", False)))
    with b2:
        st.checkbox("斜体", key="link_pct_italic", value=bool(st.session_state.get("link_pct_italic", False)))

    d1, d2 = st.columns(2)
    with d1:
        st.number_input(
            "dx(px)",
            key="link_pct_dx_px",
            value=float(st.session_state.get("link_pct_dx_px", 0.0) or 0.0),
            step=1.0,
            help="正数向右移动",
        )
    with d2:
        st.number_input(
            "dy(px)",
            key="link_pct_dy_px",
            value=float(st.session_state.get("link_pct_dy_px", 0.0) or 0.0),
            step=1.0,
            help="正数向上移动",
        )

    st.checkbox(
        "隐藏占位↔占位段的占比（避免空白列拆分后太密）",
        key="link_pct_skip_internal_placeholder",
        value=bool(st.session_state.get("link_pct_skip_internal_placeholder", True)),
    )

    st.checkbox(
        "合并相同 source→target（同一段内）只显示一个占比（推荐）",
        key="link_pct_aggregate_same_pair",
        value=bool(st.session_state.get("link_pct_aggregate_same_pair", True)),
        help="很多数据是按行拆开的，视觉上会叠成一条粗线；开启后占比也按同一对节点汇总，只画一次，避免黑条。"
    )

    st.checkbox(
        "自动避让/隐藏过密占比文本（推荐）",
        key="link_pct_enable_density_detection",
        value=bool(st.session_state.get("link_pct_enable_density_detection", True)),
        help="当同一段内占比标签过多导致重叠时，按线条权重优先显示，自动隐藏部分。"
    )

    st.number_input(
        "占比标签最小间距（px，0=自动）",
        key="link_pct_min_sep_px",
        value=float(st.session_state.get("link_pct_min_sep_px", 0.0) or 0.0),
        min_value=0.0,
        step=1.0,
        help="用于“自动避让”。0 表示按字号自动估计。"
    )


with st.sidebar.expander("节点样式", expanded=False):
    node_alpha = st.slider("NODE_ALPHA (透明度)", 0.0, 1.0, float(st.session_state.get("node_alpha", 0.70)), 0.01, key="node_alpha")
    st.checkbox(
        "启用节点描边",
        value=bool(st.session_state.get("enable_node_outline", False)),
        key="enable_node_outline",
    )
    st.color_picker(
        "节点描边颜色",
        value=_safe_hex_color(st.session_state.get("node_outline_color", "#000000"), "#000000"),
        key="node_outline_color",
    )
    st.slider(
        "节点描边透明度",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("node_outline_alpha", 0.35)),
        step=0.01,
        key="node_outline_alpha",
    )
    st.number_input(
        "节点描边粗细（px）",
        min_value=0.0,
        value=float(st.session_state.get("node_outline_width_px", 1.0)),
        step=0.1,
        key="node_outline_width_px",
    )

with st.sidebar.expander("标签全局默认", expanded=False):
    show_labels = st.checkbox("显示所有标签（全局开关）", key="show_labels", value=bool(st.session_state.get("show_labels", True)))
    text_font_size = st.number_input("TEXT_FONT_SIZE（默认字号）", key="text_font_size", value=float(st.session_state.get("text_font_size", 20.0)), min_value=1.0, step=1.0)
    label_below_middle = st.checkbox("LABEL_BELOW_MIDDLE（auto 模式：中间列默认放下方）", key="label_below_middle", value=bool(st.session_state.get("label_below_middle", True)))
    label_offset_px = st.number_input("LABEL_OFFSET_PX（below/above 的距离）", key="label_offset_px", value=float(st.session_state.get("label_offset_px", 20.0)), min_value=0.0, step=1.0)
    enable_faux_bold = st.checkbox(
        "启用伪加粗（无粗体字重字体增强）",
        key="enable_faux_bold",
        value=bool(st.session_state.get("enable_faux_bold", True)),
    )
    faux_bold_width_px = st.number_input(
        "伪加粗强度（px）",
        key="faux_bold_width_px",
        value=float(st.session_state.get("faux_bold_width_px", 0.6)),
        min_value=0.0,
        step=0.1,
    )
    label_text_color_default = st.text_input("默认文字颜色（hex）", key="label_text_color_default", value=str(st.session_state.get("label_text_color_default", "#000000")))

with st.sidebar.expander("画布尺寸（最终导出）", expanded=False):
    if "fig_w" not in st.session_state:
        st.session_state.fig_w = 20.0
    if "fig_h" not in st.session_state:
        st.session_state.fig_h = 15.0
    if "dpi" not in st.session_state:
        st.session_state.dpi = 300
    fig_w = st.number_input("FIG_WIDTH_IN", key="fig_w", value=float(st.session_state.get("fig_w", 20.0)), min_value=1.0, step=1.0)
    fig_h = st.number_input("FIG_HEIGHT_IN", key="fig_h", value=float(st.session_state.get("fig_h", 15.0)), min_value=1.0, step=1.0)
    dpi = st.number_input("DPI（PDF 主要无所谓；PNG 会用到）", key="dpi", value=int(st.session_state.get("dpi", 300)), min_value=50, step=50)

with st.sidebar.expander("自动适配画布", expanded=False):
    if "enable_auto_fit_canvas" not in st.session_state:
        st.session_state.enable_auto_fit_canvas = False
    if "auto_fit_trigger_mode" not in st.session_state:
        st.session_state.auto_fit_trigger_mode = "manual"
    if "auto_fit_max_iter" not in st.session_state:
        st.session_state.auto_fit_max_iter = 8
    if "auto_fit_prefer_expand_canvas" not in st.session_state:
        st.session_state.auto_fit_prefer_expand_canvas = True
    if "auto_fit_consider_legend" not in st.session_state:
        st.session_state.auto_fit_consider_legend = True
    if "auto_fit_consider_link_pct" not in st.session_state:
        st.session_state.auto_fit_consider_link_pct = True

    enable_auto_fit_canvas = st.checkbox(
        "启用自动适配画布",
        key="enable_auto_fit_canvas",
        value=bool(st.session_state.get("enable_auto_fit_canvas", False)),
    )
    auto_fit_trigger_mode = st.selectbox(
        "触发模式",
        options=["manual", "before_render"],
        key="auto_fit_trigger_mode",
        index=0 if str(st.session_state.get("auto_fit_trigger_mode", "manual")) == "manual" else 1,
        format_func=lambda x: {"manual": "手动触发", "before_render": "渲染前自动触发"}.get(x, x),
    )
    auto_fit_max_iter = int(st.number_input(
        "最大迭代次数",
        min_value=1,
        max_value=50,
        step=1,
        key="auto_fit_max_iter",
        value=int(st.session_state.get("auto_fit_max_iter", 8) or 8),
    ))
    auto_fit_prefer_expand_canvas = st.checkbox(
        "优先放大画布",
        key="auto_fit_prefer_expand_canvas",
        value=bool(st.session_state.get("auto_fit_prefer_expand_canvas", True)),
    )
    auto_fit_consider_legend = st.checkbox(
        "纳入图例碰撞检测",
        key="auto_fit_consider_legend",
        value=bool(st.session_state.get("auto_fit_consider_legend", True)),
    )
    auto_fit_consider_link_pct = st.checkbox(
        "纳入百分比标签碰撞检测",
        key="auto_fit_consider_link_pct",
        value=bool(st.session_state.get("auto_fit_consider_link_pct", True)),
    )


# ==============================
# 节点标签密度控制
# ==============================
with st.sidebar.expander("节点标签显示优化", expanded=False):
    enable_alternate_label_sides = st.checkbox(
        "启用按列左右交错标签",
        key="enable_alternate_label_sides",
        value=bool(st.session_state.get("enable_alternate_label_sides", False)),
        help="启用后，对选中列按“右-左-右-左”交错放置标签；优先级高于列标签 pos 设置。",
    )
    _als_prev = [int(c) for c in st.session_state.get("alternate_label_side_cols", []) if str(c).strip() != ""]
    _als_prev = [c for c in _als_prev if 0 <= c < int(n_cols)]
    if _als_prev != st.session_state.get("alternate_label_side_cols", []):
        st.session_state.alternate_label_side_cols = _als_prev
    alternate_label_side_cols = st.multiselect(
        "交错标签列（右-左-右-左）",
        options=list(range(int(n_cols))),
        key="alternate_label_side_cols",
        default=_als_prev,
        disabled=not bool(enable_alternate_label_sides),
        format_func=lambda i: f"第{i+1}列 (stage {i})",
        help="第一行在右侧、第二行在左侧、第三行在右侧，依次交替。",
    )

    enable_vertical_node_labels = st.checkbox(
        "启用指定列节点名竖排",
        key="enable_vertical_node_labels",
        value=bool(st.session_state.get("enable_vertical_node_labels", False)),
        help="选中列的节点名逐字竖排；横向图从上到下，竖向图从下到上。",
    )
    _vnl_prev = [int(c) for c in st.session_state.get("vertical_node_label_cols", []) if str(c).strip() != ""]
    _vnl_prev = [c for c in _vnl_prev if 0 <= c < int(n_cols)]
    if _vnl_prev != st.session_state.get("vertical_node_label_cols", []):
        st.session_state.vertical_node_label_cols = _vnl_prev
    vertical_node_label_cols = st.multiselect(
        "节点名竖排列",
        options=list(range(int(n_cols))),
        key="vertical_node_label_cols",
        default=_vnl_prev,
        disabled=not bool(enable_vertical_node_labels),
        format_func=lambda i: f"第{i+1}列 (stage {i})",
    )

    if "enable_cjk_auto_wrap" not in st.session_state:
        st.session_state.enable_cjk_auto_wrap = False
    if "cjk_wrap_chars_per_line" not in st.session_state:
        st.session_state.cjk_wrap_chars_per_line = 8
    if "wrap_targets" not in st.session_state:
        st.session_state.wrap_targets = ["node_label"]
    if "wrap_line_spacing_mult" not in st.session_state:
        st.session_state.wrap_line_spacing_mult = 1.20
    if "wrap_max_lines" not in st.session_state:
        st.session_state.wrap_max_lines = 0

    st.markdown("---")
    st.caption("中文标签自动换行")
    enable_cjk_auto_wrap = st.checkbox(
        "启用中文自动换行",
        key="enable_cjk_auto_wrap",
        value=bool(st.session_state.get("enable_cjk_auto_wrap", False)),
    )
    cjk_wrap_chars_per_line = int(st.number_input(
        "每行中文字符数",
        min_value=1,
        max_value=200,
        step=1,
        key="cjk_wrap_chars_per_line",
        value=int(st.session_state.get("cjk_wrap_chars_per_line", 8)),
    ))
    wrap_targets = st.multiselect(
        "换行作用对象",
        options=["node_label", "legend_label"],
        default=[x for x in st.session_state.get("wrap_targets", ["node_label"]) if x in ("node_label", "legend_label")],
        key="wrap_targets",
        format_func=lambda x: {"node_label": "节点标签", "legend_label": "图例标签"}.get(x, x),
    )
    wrap_line_spacing_mult = float(st.number_input(
        "多行行距倍率",
        min_value=1.0,
        max_value=3.0,
        step=0.05,
        key="wrap_line_spacing_mult",
        value=float(st.session_state.get("wrap_line_spacing_mult", 1.2) or 1.2),
    ))
    wrap_max_lines = int(st.number_input(
        "最大行数（0=不限制）",
        min_value=0,
        max_value=100,
        step=1,
        key="wrap_max_lines",
        value=int(st.session_state.get("wrap_max_lines", 0) or 0),
    ))

    st.markdown("---")
    # ---- 节点图例（颜色块） ----
    enable_long_label_legend = st.checkbox(
        "启用节点图例（颜色块）",
        key="enable_long_label_legend",
        value=bool(st.session_state.get("enable_long_label_legend", False)),
    )
    long_label_legend_threshold = st.number_input(
        "长节点名进图例阈值（字符数，严格大于才触发）",
        key="long_label_legend_threshold",
        value=int(st.session_state.get("long_label_legend_threshold", 30)),
        min_value=1,
        step=1,
    )
    legend_force_cols = st.multiselect(
        "整列使用图例展示",
        options=list(range(int(n_cols))),
        default=[int(x) for x in st.session_state.get("legend_force_cols", []) if 0 <= int(x) < int(n_cols)],
        key="legend_force_cols",
        format_func=lambda i: f"第{i+1}列",
        help="未开启下方自动避让时，选中列全部进入图例；开启后，仅重叠且权重较小的节点进入图例。",
    )
    legend_include_auto_hidden = st.checkbox(
        "选中列启用自动避让进图例",
        key="legend_include_auto_hidden",
        value=bool(st.session_state.get("legend_include_auto_hidden", False)),
        help="开启后，仅对上方选中的列做节点标签避让：保留权重较大的节点标签，隐藏的小节点进入图例。",
    )
    legend_position = st.selectbox(
        "图例位置",
        options=["right", "left", "bottom"],
        key="legend_position",
        index={"right": 0, "left": 1, "bottom": 2}.get(str(st.session_state.get("legend_position", "right")), 0),
        format_func=lambda x: {"right": "右侧", "left": "左侧", "bottom": "下侧"}.get(x, x),
    )
    legend_font_size = st.number_input(
        "图例字号（独立）",
        key="legend_font_size",
        value=float(st.session_state.get("legend_font_size", 16.0)),
        min_value=1.0,
        step=1.0,
    )
    legend_dx_px = st.number_input(
        "图例横向偏移（px，正=右，负=左）",
        key="legend_dx_px",
        value=float(st.session_state.get("legend_dx_px", 0.0)),
        step=1.0,
    )
    legend_dy_px = st.number_input(
        "图例纵向偏移（px，正=下，负=上）",
        key="legend_dy_px",
        value=float(st.session_state.get("legend_dy_px", 0.0)),
        step=1.0,
    )
    legend_layout_mode = str(st.session_state.get("legend_layout_mode", "packed"))
    legend_column_title_mode = str(st.session_state.get("legend_column_title_mode", "letter"))
    index_label_color = st.color_picker(
        "索引编号颜色（A1/B2…，用于区分正常节点名如 E32/C32）",
        key="index_label_color",
        value=str(st.session_state.get("index_label_color", "#4A4A4A")),
    )
    index_label_font = st.text_input(
        "索引编号字体（留空=跟随全局字体）",
        key="index_label_font",
        value=str(st.session_state.get("index_label_font", "")),
    )
    index_label_bold = st.checkbox(
        "索引编号加粗（图上）",
        key="index_label_bold",
        value=bool(st.session_state.get("index_label_bold", False)),
    )
    index_label_italic = st.checkbox(
        "索引编号斜体（图上）",
        key="index_label_italic",
        value=bool(st.session_state.get("index_label_italic", False)),
    )

    st.markdown("---")
    label_enable_density_detection = st.checkbox(
        "启用节点标签自动避让",
        key="label_enable_density_detection",
        value=bool(st.session_state.get("label_enable_density_detection", False)),
        help="当同一列节点过多、标签可能重叠时，自动隐藏部分节点标签"
    )
    label_density_cols = st.multiselect(
        "节点标签自动避让生效列（不选=全部列）",
        options=list(range(int(n_cols))),
        default=[int(x) for x in st.session_state.get("label_density_cols", []) if 0 <= int(x) < int(n_cols)],
        format_func=lambda i: f"第{int(i)+1}列",
        key="label_density_cols",
        help="仅在启用节点标签自动避让时生效；不选择任何列则保持旧行为：全部列自动避让",
    )
    label_density_priority = st.selectbox(
        "标签保留优先级",
        options=["height", "weight"],
        key="label_density_priority",
        index=0 if str(st.session_state.get("label_density_priority", "height")) == "height" else 1,
        help="height：优先显示视觉上更高的节点（推荐）"
    )
    min_gap_px = st.number_input(
        "标签最小垂直间距（px）",
        key="min_gap_px",
        min_value=0.0,
        value=float(st.session_state.get("min_gap_px", 0.0)),
        step=1.0,
        help="0 表示自动"
    )
    label_min_vsep_px = None if min_gap_px <= 0 else float(min_gap_px)

with st.sidebar.expander("标签颜色映射（按列）", expanded=False):
    st.caption("把某列的节点 label 颜色设为该节点的填充色（按列控制）。")
    col_options = list(range(n_cols))
    selected_cols_for_node_color = st.multiselect(
        "选择要使用节点颜色的列（多选）",
        options=col_options,
        key="ui_selected_cols_for_node_color",
        default=[int(x) for x in st.session_state.get("ui_selected_cols_for_node_color", []) if int(x) in col_options]
    )
    apply_btn, clear_btn = st.columns([1, 1])
    with apply_btn:
        if st.button("应用到选中列"):
            t = st.session_state.label_table.copy()
            for i in range(len(t)):
                if int(t.at[i, "col_index"]) in selected_cols_for_node_color:
                    t.at[i, "use_node_color"] = True
            st.session_state.label_table = t
            st.success(f"已对列 {selected_cols_for_node_color} 设置 use_node_color=True")
    with clear_btn:
        if st.button("清除选中列设置"):
            t = st.session_state.label_table.copy()
            for i in range(len(t)):
                if int(t.at[i, "col_index"]) in selected_cols_for_node_color:
                    t.at[i, "use_node_color"] = False
            st.session_state.label_table = t
            st.success(f"已对列 {selected_cols_for_node_color} 清除 use_node_color")

with st.sidebar.expander("节点顺序优化（只移动节点，不改线段）", expanded=False):
    order_mode_ui = st.selectbox(
        "节点顺序模式",
        ["按 Excel（不优化）", "减少交叉（自动）", "增加交叉（自动）"],
        key="order_mode_ui",
        index=({"excel": 0, "min_cross": 1, "max_cross": 2}.get(str(st.session_state.get("order_mode", "excel")), 0)),
    )
    order_mode = {
        "按 Excel（不优化）": "excel",
        "减少交叉（自动）": "min_cross",
        "增加交叉（自动）": "max_cross",
    }[order_mode_ui]
    st.session_state.order_mode = order_mode

    stage_opts = list(range(max(0, n_cols - 1)))
    default_stages = stage_opts if (order_mode != "excel" and stage_opts) else []
    order_target_stages = st.multiselect(
        "选择要优化的段 stage（0 表示 col0→col1）",
        options=stage_opts,
        key="order_target_stages",
        default=[int(x) for x in st.session_state.get("order_target_stages", default_stages) if int(x) in stage_opts],
    )
    order_keep_ratio = st.slider("保留 Excel 顺序比例", 0.0, 1.0, float(st.session_state.get("order_keep_ratio", 0.35)), 0.05, key="order_keep_ratio")

with st.sidebar.expander("高级：手动翻转列（不推荐）", expanded=False):
    flip_cols = st.multiselect("选择要翻转的列（0..N_COLS-1）", options=list(range(n_cols)), key="flip_cols", default=[int(x) for x in st.session_state.get("flip_cols", []) if int(x) in list(range(n_cols))])

# ---------- Main layout ----------
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader("列布局 COL_CFG")

    btns = st.columns(3)
    if btns[0].button("重置列布局"):
        st.session_state.col_table = init_col_layout_table(n_cols, x_min=x_min, x_max=x_max)

    if btns[1].button("x 均匀分布"):
        t = st.session_state.col_table.copy()
        xs = [x_min + (x_max - x_min) * j / (n_cols - 1) for j in range(n_cols)] if n_cols > 1 else [
            (x_min + x_max) / 2]
        t["x"] = xs
        st.session_state.col_table = t

    if btns[2].button("x 全部置空（走自动插值）"):
        t = st.session_state.col_table.copy()
        t["x"] = [None] * n_cols
        st.session_state.col_table = t

    edited_col = st.data_editor(
        st.session_state.col_table,
        use_container_width=True,
        num_rows="fixed",
        key="col_table_editor",
        column_config={
            "align": st.column_config.SelectboxColumn(
                "align (对齐)",
                options=["top", "center", "bottom"],
                help="相对基准列的对齐方式"
            ),
            # --- 新增列配置 ---
            "group_gap_on": st.column_config.CheckboxColumn(
                "group_gap_on",
                help="该列是否启用同源分组间隙？需配合侧边栏全局开关使用。"
            ),
            "group_gap_px": st.column_config.NumberColumn(
                "group_gap_px",
                help="该列的分组间隙大小 (px)"
            ),
            # -----------------
            "pos": st.column_config.SelectboxColumn("pos", options=["auto", "inside", "below", "above", "left", "right",
                                                                    "none"]),
            "text_color": st.column_config.TextColumn("text_color"),
            "use_node_color": st.column_config.CheckboxColumn("use_node_color"),
        },
    )
    st.session_state.col_table = edited_col

    st.subheader("列标签配置（按列分组控制）")

    btnl = st.columns(2)
    if btnl[0].button("重置列标签配置"):
        st.session_state.label_table = init_col_label_table(n_cols)
    if btnl[1].button("全部设为 auto + 显示"):
        t = st.session_state.label_table.copy()
        t["show"] = [True] * n_cols
        t["pos"] = ["auto"] * n_cols
        st.session_state.label_table = t

    edited_label = st.data_editor(
        st.session_state.label_table,
        use_container_width=True,
        num_rows="fixed",
        key="label_table_editor",
        column_config={
            "pos": st.column_config.SelectboxColumn("pos", options=["auto", "inside", "below", "above", "left", "right",
                                                                    "none"]),
            "text_color": st.column_config.TextColumn("text_color"),
            "use_node_color": st.column_config.CheckboxColumn("use_node_color"),
            "bold": st.column_config.CheckboxColumn("bold"),
            "italic": st.column_config.CheckboxColumn("italic"),
            "underline": st.column_config.CheckboxColumn("underline"),
        },
    )
    st.subheader("按段固定线条颜色（可选）")
    st.caption("你可以指定某一段（stage）的线条颜色为固定色；未开启的段仍按“线条颜色模式”走原逻辑。"
               "跳列连接（例如第2列直接连第4列）会按【目标列-1】计入段：也就是算到第3段（第3列→第4列）。")

    if int(n_cols) <= 1:
        edited_stage = st.session_state.link_stage_table
        st.info("当前只有 1 列，没有线段可以配置。")
    else:
        edited_stage = st.data_editor(
            st.session_state.link_stage_table,
            use_container_width=True,
            num_rows="fixed",
            key="link_stage_table_editor",
            disabled=["stage_index", "segment"],
            column_config={
                "enable": st.column_config.CheckboxColumn("enable", help="是否启用该段的固定颜色覆盖"),
                "color": st.column_config.TextColumn("color", help="固定颜色（hex，例如 #FF0000）"),
                "segment": st.column_config.TextColumn("segment"),
            },
        )
        st.session_state.link_stage_table = edited_stage

    st.session_state.label_table = edited_label

    with st.expander("导出/导入配置（JSON）", expanded=False):
        template_sig = build_data_signature(df=df, sheet_name=str(sheet), n_cols=int(n_cols))
        cfg_export = {
            "template_schema_version": 1,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "data_signature": template_sig,
            "n_cols": int(n_cols),
            "global": {
                "orientation": str(st.session_state.get("orientation", "horizontal")),
                "fig_width_in": float(fig_w),
                "fig_height_in": float(fig_h),
                "dpi": int(dpi),
                "preview_fig_width_in": float(st.session_state.get("preview_fig_w", 16.0)),
                "preview_fig_height_in": float(st.session_state.get("preview_fig_h", 10.0)),
                "preview_dpi": int(st.session_state.get("preview_dpi", 150)),
                "value_to_px": float(value_to_px),
                "use_min_link_thickness": bool(use_min_link),
                "min_link_px": float(min_link_px),
                "min_node_h_px": float(min_node_h_px),
                "force_align_top_bottom": bool(force_align),
                "force_align_exempt_cols": list(map(int, force_align_exempt_cols)),
                "layout_ref_col_index": int(layout_ref_col_index),
                "enable_group_gap": bool(enable_group_gap),
                "enable_node_placeholders": bool(st.session_state.get("enable_node_placeholders", False)),
                "stack_mode": str(stack_mode),
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "link_node_gap_px": float(link_node_gap_px),
                "curve_ctrl_rel": float(curve_ctrl_rel),
                "link_alpha": float(link_alpha),
                "enable_link_side_outline": bool(st.session_state.get("enable_link_side_outline", False)),
                "link_side_outline_color": _safe_hex_color(st.session_state.get("link_side_outline_color", "#000000"), "#000000"),
                "link_side_outline_alpha": float(st.session_state.get("link_side_outline_alpha", 0.35) or 0.35),
                "link_side_outline_width_px": float(st.session_state.get("link_side_outline_width_px", 1.0) or 1.0),
                "link_color_mode": str(link_color_mode),
                "link_color_follow_col": int(st.session_state.get("link_color_follow_col", 0)),
                "auto_balance_flow": bool(st.session_state.get("auto_balance_flow", False)),
                "show_link_pct": bool(st.session_state.get("show_link_pct", False)),
                "link_pct_position": str(st.session_state.get("link_pct_position", "middle")),
                "link_pct_basis": str(st.session_state.get("link_pct_basis", "raw")),
                "link_pct_format": str(st.session_state.get("link_pct_format", "{pct100:.1f}%")),
                "link_pct_font_size": float(st.session_state.get("link_pct_font_size", 16.0) or 16.0),
                "link_pct_color": str(st.session_state.get("link_pct_color", "#000000") or "#000000"),
                "link_pct_bold": bool(st.session_state.get("link_pct_bold", False)),
                "link_pct_italic": bool(st.session_state.get("link_pct_italic", False)),
                "link_pct_dx_px": float(st.session_state.get("link_pct_dx_px", 0.0) or 0.0),
                "link_pct_dy_px": float(st.session_state.get("link_pct_dy_px", 0.0) or 0.0),
                "link_pct_skip_internal_placeholder": bool(st.session_state.get("link_pct_skip_internal_placeholder", True)),
                "link_pct_aggregate_same_pair": bool(st.session_state.get("link_pct_aggregate_same_pair", True)),
                "link_pct_enable_density_detection": bool(st.session_state.get("link_pct_enable_density_detection", True)),
                "link_pct_min_sep_px": float(st.session_state.get("link_pct_min_sep_px", 0.0) or 0.0),
                "node_alpha": float(node_alpha),
                "enable_node_outline": bool(st.session_state.get("enable_node_outline", False)),
                "node_outline_color": _safe_hex_color(st.session_state.get("node_outline_color", "#000000"), "#000000"),
                "node_outline_alpha": float(st.session_state.get("node_outline_alpha", 0.35) or 0.35),
                "node_outline_width_px": float(st.session_state.get("node_outline_width_px", 1.0) or 1.0),
                "font_priority": list(font_priority),
                "font_zh_priority": list(font_zh_priority),
                "font_en_priority": list(font_en_priority),
                "enable_header_row": bool(st.session_state.get("enable_header_row", False)),
                "show_headers": bool(st.session_state.get("show_headers", True)),
                "header_pos": str(st.session_state.get("header_pos", "top")),
                "header_dy_px": float(st.session_state.get("header_dy_px", 0.0) or 0.0),
                "header_font_size": (None if header_font_size is None else float(header_font_size)),
                "header_text_color": (header_text_color or ""),
                "header_font_priority": list(header_font_priority),
                "header_font_zh_priority": list(header_font_zh_priority),
                "header_font_en_priority": list(header_font_en_priority),
                "header_bold": header_bold,
                "header_italic": header_italic,
                "header_underline": header_underline,
                "show_title": bool(st.session_state.get("show_title", False)),
                "title_text": str(st.session_state.get("title_text", "") or ""),
                "title_dx_px": float(st.session_state.get("title_dx_px", 0.0) or 0.0),
                "title_dy_px": float(st.session_state.get("title_dy_px", 0.0) or 0.0),
                "title_font_size": (None if title_font_size is None else float(title_font_size)),
                "title_text_color": (title_text_color or ""),
                "title_font_priority": list(title_font_priority),
                "title_font_zh_priority": list(title_font_zh_priority),
                "title_font_en_priority": list(title_font_en_priority),
                "title_bold": title_bold,
                "title_italic": title_italic,
                "title_underline": title_underline,
                "text_font_size": float(text_font_size),
                "label_below_middle": bool(label_below_middle),
                "label_offset_px": float(label_offset_px),
                "enable_faux_bold": bool(enable_faux_bold),
                "faux_bold_width_px": float(faux_bold_width_px),
                "enable_alternate_label_sides": bool(enable_alternate_label_sides),
                "alternate_label_side_cols": list(map(int, alternate_label_side_cols)),
                "enable_vertical_node_labels": bool(enable_vertical_node_labels),
                "vertical_node_label_cols": list(map(int, vertical_node_label_cols)),
                "label_enable_density_detection": bool(label_enable_density_detection),
                "label_density_cols": list(map(int, label_density_cols)),
                "label_density_priority": str(label_density_priority),
                "min_gap_px": float(min_gap_px),
                "show_labels": bool(show_labels),
                "label_text_color_default": str(label_text_color_default),
                "enable_cjk_auto_wrap": bool(enable_cjk_auto_wrap),
                "cjk_wrap_chars_per_line": int(cjk_wrap_chars_per_line),
                "wrap_targets": list(wrap_targets),
                "wrap_line_spacing_mult": float(wrap_line_spacing_mult),
                "wrap_max_lines": (None if int(wrap_max_lines) <= 0 else int(wrap_max_lines)),
                "enable_auto_fit_canvas": bool(enable_auto_fit_canvas),
                "auto_fit_trigger_mode": str(auto_fit_trigger_mode),
                "auto_fit_max_iter": int(auto_fit_max_iter),
                "auto_fit_prefer_expand_canvas": bool(auto_fit_prefer_expand_canvas),
                "auto_fit_consider_legend": bool(auto_fit_consider_legend),
                "auto_fit_consider_link_pct": bool(auto_fit_consider_link_pct),
                "enable_long_label_legend": bool(enable_long_label_legend),
                "long_label_legend_threshold": int(long_label_legend_threshold),
                "legend_force_cols": list(map(int, legend_force_cols)),
                "legend_include_auto_hidden": bool(legend_include_auto_hidden),
                "legend_position": str(legend_position),
                "legend_dx_px": float(legend_dx_px),
                "legend_dy_px": float(legend_dy_px),
                "legend_font_size": float(legend_font_size),
                "legend_layout_mode": str(legend_layout_mode),
                "legend_column_title_mode": str(legend_column_title_mode),
                "index_label_color": str(index_label_color),
                "index_label_font": str(index_label_font),
                "index_label_bold": bool(index_label_bold),
                "index_label_italic": bool(index_label_italic),
                "order_mode": str(order_mode),
                "order_target_stages": list(map(int, order_target_stages)),
                "order_keep_ratio": float(order_keep_ratio),
                "flip_cols": list(map(int, flip_cols)),
                "ui_selected_cols_for_node_color": list(map(int, selected_cols_for_node_color)),
                "disable_merge_toggle": bool(st.session_state.get("disable_merge_toggle", False)),
                "no_merge_cols": list(map(int, st.session_state.get("no_merge_cols", []))),
                "use_last_col_weight_override": bool(st.session_state.get("use_last_col_weight_override", False)),
            },
            "col_cfg_rows": edited_col.to_dict(orient="records"),
            "col_label_cfg_rows": edited_label.to_dict(orient="records"),
            "link_stage_override_rows": (edited_stage.to_dict(orient="records") if hasattr(edited_stage, "to_dict") else []),
        }

        st.download_button(
            "下载固定模板 JSON（强绑定）",
            data=json.dumps(cfg_export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="sankey_template.json",
            mime="application/json",
        )

        if st.session_state.pop("_cfg_loaded_ok", False):
            st.success("配置已载入！")
        if "_cfg_loaded_err" in st.session_state:
            st.error(f"配置载入失败：{st.session_state.pop('_cfg_loaded_err')}")

        tab_patch, tab_full = st.tabs(["继续工作模式（补丁导入）", "模板复刻模式（全量导入）"])

        with tab_patch:
            st.caption("仅覆盖 JSON 中出现的键；缺失键保持当前参数。")
            up_cfg_patch = st.file_uploader("上传配置 JSON（补丁）", type=["json"], key="cfg_uploader_patch")
            if st.button("应用补丁配置", key="btn_apply_patch_cfg"):
                if up_cfg_patch is None:
                    st.warning("请先上传 JSON 文件。")
                else:
                    try:
                        loaded = json.loads(up_cfg_patch.getvalue().decode("utf-8-sig"))
                        st.session_state["_pending_cfg_payload"] = {"mode": "patch", "config": loaded}
                        _safe_rerun()
                    except Exception as e:
                        st.error(f"补丁导入失败：{e}")

        with tab_full:
            st.caption("全量恢复所有参数。")
            up_cfg_full = st.file_uploader("上传模板 JSON（全量）", type=["json"], key="cfg_uploader_full")
            if st.button("应用模板复刻", key="btn_apply_full_cfg"):
                if up_cfg_full is None:
                    st.warning("请先上传模板 JSON 文件。")
                else:
                    try:
                        loaded = json.loads(up_cfg_full.getvalue().decode("utf-8-sig"))
                        st.session_state["_pending_cfg_payload"] = {"mode": "full", "config": loaded}
                        _safe_rerun()
                    except Exception as e:
                        st.error(f"模板导入失败：{e}")

with col2:
    st.subheader("生成与下载")

    with st.expander("预览设置", expanded=False):
        if "preview_fig_w" not in st.session_state:
            st.session_state.preview_fig_w = 16.0
        if "preview_fig_h" not in st.session_state:
            st.session_state.preview_fig_h = 10.0
        if "preview_dpi" not in st.session_state:
            st.session_state.preview_dpi = 150
        prev_w = st.number_input("预览 FIG_WIDTH_IN", key="preview_fig_w", value=float(st.session_state.get("preview_fig_w", 16.0)), min_value=4.0, step=1.0)
        prev_h = st.number_input("预览 FIG_HEIGHT_IN", key="preview_fig_h", value=float(st.session_state.get("preview_fig_h", 10.0)), min_value=3.0, step=1.0)
        prev_dpi = st.number_input("预览 DPI", key="preview_dpi", value=int(st.session_state.get("preview_dpi", 150)), min_value=50, step=50)


    def build_cfg() -> SankeyConfig:
        col_rows: List[Dict[str, Any]] = st.session_state.col_table.to_dict(orient="records")
        label_rows: List[Dict[str, Any]] = st.session_state.label_table.to_dict(orient="records")
        link_stage_rows: List[Dict[str, Any]] = st.session_state.get("link_stage_table", pd.DataFrame()).to_dict(orient="records")

        return SankeyConfig(
            n_cols=int(n_cols),
            fig_width_in=float(fig_w),
            fig_height_in=float(fig_h),
            dpi=int(dpi),
            orientation=str(st.session_state.get('orientation', 'horizontal')),

            value_to_px=float(value_to_px),
            use_min_link_thickness=bool(use_min_link),
            min_link_px=float(min_link_px),
            min_node_h_px=float(min_node_h_px),
            force_align_top_bottom=bool(force_align),
            force_align_exempt_cols=tuple(map(int, force_align_exempt_cols)),
            stack_mode=str(stack_mode),
            layout_ref_col_index=int(layout_ref_col_index) if layout_ref_col_index != -1 else None,

            # --- 传入全局开关 (具体间隙值现在从 col_rows 里读) ---
            enable_group_gap=bool(enable_group_gap),
            enable_node_placeholders=bool(st.session_state.get('enable_node_placeholders', False)),

            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            link_node_gap_px=float(link_node_gap_px),
            link_alpha=float(link_alpha),
            enable_link_side_outline=bool(st.session_state.get("enable_link_side_outline", False)),
            link_side_outline_color=_safe_hex_color(st.session_state.get("link_side_outline_color", "#000000"), "#000000"),
            link_side_outline_alpha=float(st.session_state.get("link_side_outline_alpha", 0.35) or 0.35),
            link_side_outline_width_px=float(st.session_state.get("link_side_outline_width_px", 1.0) or 1.0),
            link_color_mode=str(link_color_mode),
            link_color_follow_col=int(st.session_state.get("link_color_follow_col", 0)),
            link_stage_override_rows=link_stage_rows,
            auto_balance_flow=bool(st.session_state.get("auto_balance_flow", False)),
            show_link_pct=bool(st.session_state.get("show_link_pct", False)),
            link_pct_position=str(st.session_state.get("link_pct_position", "middle")),
            link_pct_basis=str(st.session_state.get("link_pct_basis", "raw")),
            link_pct_format=str(st.session_state.get("link_pct_format", "{pct100:.1f}%")),
            link_pct_font_size=float(st.session_state.get("link_pct_font_size", 16.0) or 16.0),
            link_pct_color=str(st.session_state.get("link_pct_color", "#000000") or "#000000"),
            link_pct_bold=bool(st.session_state.get("link_pct_bold", False)),
            link_pct_italic=bool(st.session_state.get("link_pct_italic", False)),
            link_pct_dx_px=float(st.session_state.get("link_pct_dx_px", 0.0) or 0.0),
            link_pct_dy_px=float(st.session_state.get("link_pct_dy_px", 0.0) or 0.0),
            link_pct_skip_internal_placeholder=bool(st.session_state.get("link_pct_skip_internal_placeholder", True)),

            link_pct_aggregate_same_pair=bool(st.session_state.get("link_pct_aggregate_same_pair", True)),
            link_pct_enable_density_detection=bool(st.session_state.get("link_pct_enable_density_detection", True)),
            link_pct_min_sep_px=(None if float(st.session_state.get("link_pct_min_sep_px", 0.0) or 0.0) <= 0 else float(st.session_state.get("link_pct_min_sep_px", 0.0) or 0.0)),
            node_alpha=float(node_alpha),
            enable_node_outline=bool(st.session_state.get("enable_node_outline", False)),
            node_outline_color=_safe_hex_color(st.session_state.get("node_outline_color", "#000000"), "#000000"),
            node_outline_alpha=float(st.session_state.get("node_outline_alpha", 0.35) or 0.35),
            node_outline_width_px=float(st.session_state.get("node_outline_width_px", 1.0) or 1.0),
            curve_ctrl_rel=float(curve_ctrl_rel),
            font_priority=tuple(font_priority),
            font_zh_priority=tuple(font_zh_priority),
            font_en_priority=tuple(font_en_priority),
            enable_header_row=bool(st.session_state.get("enable_header_row", False)),
            show_headers=bool(st.session_state.get("show_headers", True)),
            header_pos=str(st.session_state.get("header_pos", "top")),
            header_dy_px=float(st.session_state.get("header_dy_px", 0.0) or 0.0),
            header_font_size=header_font_size,
            header_text_color=header_text_color,
            header_font_priority=tuple(header_font_priority),
            header_font_zh_priority=tuple(header_font_zh_priority),
            header_font_en_priority=tuple(header_font_en_priority),
            header_bold=header_bold,
            header_italic=header_italic,
            header_underline=header_underline,
            show_title=bool(st.session_state.get("show_title", False)),
            title_text=str(st.session_state.get("title_text", "") or ""),
            title_dx_px=float(st.session_state.get("title_dx_px", 0.0) or 0.0),
            title_dy_px=float(st.session_state.get("title_dy_px", 0.0) or 0.0),
            title_font_size=title_font_size,
            title_text_color=title_text_color,
            title_font_priority=tuple(title_font_priority),
            title_font_zh_priority=tuple(title_font_zh_priority),
            title_font_en_priority=tuple(title_font_en_priority),
            title_bold=title_bold,
            title_italic=title_italic,
            title_underline=title_underline,
            text_font_size=float(text_font_size),
            label_below_middle=bool(label_below_middle),
            label_offset_px=float(label_offset_px),
            enable_faux_bold=bool(enable_faux_bold),
            faux_bold_width_px=float(faux_bold_width_px),
            enable_alternate_label_sides=bool(enable_alternate_label_sides),
            alternate_label_side_cols=tuple(map(int, alternate_label_side_cols)),
            enable_vertical_node_labels=bool(enable_vertical_node_labels),
            vertical_node_label_cols=tuple(map(int, vertical_node_label_cols)),
            label_enable_density_detection=bool(label_enable_density_detection),
            label_density_cols=tuple(map(int, label_density_cols)),
            label_density_priority=str(label_density_priority),
            label_min_vsep_px=label_min_vsep_px,
            show_labels=bool(show_labels),
            label_text_color_default=str(label_text_color_default),
            enable_cjk_auto_wrap=bool(enable_cjk_auto_wrap),
            cjk_wrap_chars_per_line=int(cjk_wrap_chars_per_line),
            wrap_targets=tuple(wrap_targets),
            wrap_line_spacing_mult=float(wrap_line_spacing_mult),
            wrap_max_lines=(None if int(wrap_max_lines) <= 0 else int(wrap_max_lines)),
            enable_auto_fit_canvas=bool(enable_auto_fit_canvas),
            auto_fit_trigger_mode=str(auto_fit_trigger_mode),
            auto_fit_max_iter=int(auto_fit_max_iter),
            auto_fit_prefer_expand_canvas=bool(auto_fit_prefer_expand_canvas),
            auto_fit_consider_legend=bool(auto_fit_consider_legend),
            auto_fit_consider_link_pct=bool(auto_fit_consider_link_pct),
            enable_long_label_legend=bool(enable_long_label_legend),
            long_label_legend_threshold=int(long_label_legend_threshold),
            legend_force_cols=tuple(map(int, legend_force_cols)),
            legend_include_auto_hidden=bool(legend_include_auto_hidden),
            legend_position=str(legend_position),
            legend_dx_px=float(legend_dx_px),
            legend_dy_px=float(legend_dy_px),
            legend_font_size=float(legend_font_size),
            legend_layout_mode=str(legend_layout_mode),
            legend_column_title_mode=str(legend_column_title_mode),
            index_label_color=str(index_label_color),
            index_label_font=str(index_label_font),
            index_label_bold=bool(index_label_bold),
            index_label_italic=bool(index_label_italic),
            order_mode=str(order_mode),
            order_target_stages=tuple(map(int, order_target_stages)),
            order_keep_ratio=float(order_keep_ratio),
            flip_cols=tuple(map(int, flip_cols)),
            no_merge_cols=tuple(map(int, st.session_state.get("no_merge_cols", []))) if bool(st.session_state.get("disable_merge_toggle", False)) else tuple(),
            use_last_col_weight_override=bool(st.session_state.get("use_last_col_weight_override", False)),
            col_cfg_rows=col_rows,
            col_label_cfg_rows=label_rows,
            default_node_width_px=100.0,
            default_gap_px=50.0,
        )


    if "preview_png" not in st.session_state:
        st.session_state.preview_png = None
        st.session_state.preview_diag = None

    if "final_pdf" not in st.session_state:
        st.session_state.final_pdf = None
        st.session_state.final_png = None
        st.session_state.final_diag = None

    cbtn = st.columns(3)
    if cbtn[0].button("生成预览 PNG", type="primary"):
        try:
            base_cfg = build_cfg()
            prev_cfg = scale_config_for_preview(
                base_cfg,
                preview_w_in=float(prev_w),
                preview_h_in=float(prev_h),
                preview_dpi=int(prev_dpi),
            )
            pdf_bytes, png_bytes, diag = render_sankey_from_df(df, prev_cfg)
            st.session_state.preview_png = png_bytes
            st.session_state.preview_diag = diag
            st.success("预览生成完成。")
        except Exception as e:
            st.error(f"预览生成失败：{e}")

    if cbtn[1].button("自动适配 + 预览 PNG"):
        try:
            base_cfg = build_cfg()
            base_cfg.enable_auto_fit_canvas = True
            pdf_bytes, png_bytes, diag = render_sankey_from_df(df, base_cfg, force_auto_fit=True)
            st.session_state.preview_png = png_bytes
            st.session_state.preview_diag = diag
            st.session_state.final_pdf = pdf_bytes
            st.session_state.final_png = png_bytes
            st.session_state.final_diag = diag
            st.success("自动适配完成，预览已更新。")
            if isinstance(diag, dict) and "auto_fit_final_params" in diag:
                st.info(f"自动适配参数：{diag.get('auto_fit_final_params')}")
        except Exception as e:
            st.error(f"自动适配预览失败：{e}")

    if cbtn[2].button("导出最终 PDF+PNG"):
        try:
            base_cfg = build_cfg()
            pdf_bytes, png_bytes, diag = render_sankey_from_df(df, base_cfg)
            st.session_state.final_pdf = pdf_bytes
            st.session_state.final_png = png_bytes
            st.session_state.final_diag = diag
            st.success("最终导出完成。")
        except Exception as e:
            st.error(f"最终导出失败：{e}")

    if st.session_state.preview_png:
        st.markdown("### 预览")
        st.image(st.session_state.preview_png, use_container_width=True)
        if st.session_state.preview_diag:
            st.json(st.session_state.preview_diag)

    st.markdown("---")

    if st.session_state.final_pdf and st.session_state.final_png:
        st.markdown("### 下载最终文件")
        st.download_button(
            "下载 PDF",
            data=st.session_state.final_pdf,
            file_name="sankey_output.pdf",
            mime="application/pdf",
        )
        st.download_button(
            "下载 PNG",
            data=st.session_state.final_png,
            file_name="sankey_output.png",
            mime="image/png",
        )
        if st.session_state.final_diag:
            st.json(st.session_state.final_diag)
    else:
        st.info("还没有最终文件：点击“导出最终 PDF+PNG”。")
