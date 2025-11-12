# app.py
import streamlit as st
import requests
import pandas as pd
import json
import re
from typing import Any, Dict, List, Optional
from io import BytesIO

# ==== Python 3.8 兼容補丁：忽略 hashlib 的 usedforsecurity 參數 ====
import hashlib
try:
    hashlib.md5(b"test", usedforsecurity=False)
except TypeError:
    _orig_md5 = hashlib.md5
    def _md5_compat(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _orig_md5(*args, **kwargs)
    hashlib.md5 = _md5_compat
try:
    hashlib.sha1(b"test", usedforsecurity=False)
except TypeError:
    _orig_sha1 = hashlib.sha1
    def _sha1_compat(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _orig_sha1(*args, **kwargs)
    hashlib.sha1 = _sha1_compat
# ===================================================================

# ============ PDF（reportlab） ============
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False
# =========================================

# ---- App 基本設定 ----
st.set_page_config(page_title="Topigeon Ace Pigeon List", layout="wide")
API_BASE = "https://www.topigeon.com/api/"

# ★★★ 隱藏帳密：直接在程式內使用這組預設（不在 UI 顯示）★★★
UNAME_DEFAULT = "RafaelES"
UKEY_DEFAULT  = "ae866e78944cabad"

# ---- 欄位候選（兼容不同站的鍵名）----
ID_KEYS_CANDIDATE    = ["pring_no", "PRING_NO", "ringno", "ring_no"]
SPEED_KEYS_CANDIDATE = ["flyspeed", "speed", "avg_speed", "velocity", "v"]
LOFT_KEYS_CANDIDATE  = ["loftname", "loft_name", "loft", "loftnm"]
SEX_KEYS_CANDIDATE   = ["sex", "gender"]

# =========================
# i18n 多語
# =========================
LANGS = {
    "en": {
        "app_title": "Topigeon Ace Pigeon List",
        "debug_toggle": "Show raw API response (debug)",
        "sidebar_header": "Query",
        "clubno": "clubno (Club ID)",
        "raceyear": "raceyear (Year)",
        "load_racelist": "① Load race list (get_racelist)",
        "racelist_empty": "No race list found. Check year/clubno or permission.",
        "racelist_loaded": "Loaded {n} races.",
        "racelist_table": "Race list (raw)",
        "select_races": "② Select races to include (multi-select)",
        "calc_button": "③ Fetch race details and compute average speed",
        "no_rows": "No rows to aggregate.",
        "result_title": "Result: Ace Pigeon Ranking (desc)",
        "download_csv": "Download CSV",
        "download_pdf": "Download PDF",
        "pdf_title": "Topigeon Ace Pigeon List",
        "race_summary": "raceno {rn} summary",
        "sample_rows": "First 3 sample rows:",
        "info_no_detail": "raceno {rn} has no detail or returns empty.",
    },
    "zh": {
        "app_title": "Topigeon Ace Pigeon List",
        "debug_toggle": "顯示原始回傳（除錯用）",
        "sidebar_header": "查詢參數",
        "clubno": "clubno（俱樂部代碼）",
        "raceyear": "raceyear（年份）",
        "load_racelist": "① 取得賽事清單 get_racelist",
        "racelist_empty": "查無賽事清單（可能是 year/clubno 或權限問題）。",
        "racelist_loaded": "載入 {n} 筆賽事。",
        "racelist_table": "賽事清單（原樣）",
        "select_races": "② 勾選要計算的賽事（可多選）",
        "calc_button": "③ 取得賽事明細並計算平均速度",
        "no_rows": "沒有可彙總的明細。",
        "result_title": "結果：Ace Pigeon 名單（由高到低）",
        "download_csv": "下載 CSV",
        "download_pdf": "下載 PDF",
        "pdf_title": "Topigeon Ace Pigeon List",
        "race_summary": "raceno {rn} 處理摘要",
        "sample_rows": "前 3 筆樣本：",
        "info_no_detail": "raceno {rn} 無明細或回傳空資料",
    },
}

def t(key, **kwargs):
    lang = st.session_state.get("lang", "en")
    s = LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"].get(key, key))
    return s.format(**kwargs)

# 側欄語言選擇
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"  # 預設英文
with st.sidebar:
    st.selectbox("Language / 語言", ["en", "zh"], key="lang")

# =========================
# Utilities
# =========================
def normalize_raceno(x) -> str:
    s = str(x).strip()
    return s.zfill(8) if s.isdigit() and len(s) < 8 else s

def _text_is_jsonable(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=300)
def call_api(params: dict):
    try:
        r = requests.get(API_BASE, params=params, timeout=30)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype:
            return r.json()
        parsed = _text_is_jsonable(r.text)
        return parsed if parsed is not None else r.text
    except Exception as e:
        st.error(f"API error: {e}")
        return {}

def _list_of_dicts(obj) -> bool:
    return isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, dict) for x in obj)

def _score_row_list(rows: List[Dict[str, Any]]) -> int:
    keys = set().union(*[set(r.keys()) for r in rows]) if rows else set()
    score = 0
    for k in ID_KEYS_CANDIDATE + SPEED_KEYS_CANDIDATE + ["raceno", "race_no", "raceid", "race_id", "racenum"]:
        if k in keys:
            score += 1
    return score

def find_list_deep(obj: Any) -> List[Dict[str, Any]]:
    best_rows: List[Dict[str, Any]] = []
    best_score = -1
    def dfs(x: Any):
        nonlocal best_rows, best_score
        if _list_of_dicts(x):
            s = _score_row_list(x)
            if s > best_score:
                best_rows = x
                best_score = s
        elif isinstance(x, dict):
            for v in x.values(): dfs(v)
        elif isinstance(x, list):
            for v in x: dfs(v)
    dfs(obj)
    return best_rows

def coalesce_key(d: Dict[str, Any], cands: List[str]) -> Optional[str]:
    for k in cands:
        if k in d:
            return k
    lower_map = {k.lower(): k for k in d.keys()}
    for k in cands:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None

def most_common(series: pd.Series):
    s = series.dropna()
    s = s[s.astype(str).str.len() > 0]
    if s.empty:
        return None
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else s.iloc[0]

def parse_speed(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = s.replace(",", "").replace("\u00A0", " ")
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None

def build_pdf_from_df(df: pd.DataFrame, title: str) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed. Run: pip install reportlab")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                            leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elems = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 10),
        ("FONTSIZE",   (0,1), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elems.append(table)
    doc.build(elems)
    buffer.seek(0)
    return buffer.read()

# =========================
# UI
# =========================
st.title(t("app_title"))
show_debug = st.toggle(t("debug_toggle"), value=False)

# ---- Sidebar（只公開 year/clubno）----
with st.sidebar:
    st.header(t("sidebar_header"))
    clubno   = st.text_input(t("clubno"), value="")
    raceyear = st.text_input(t("raceyear"), value="2025")
    st.caption(" ")

# =========================
# ① 取得賽事清單（不帶 APP=Y）
# =========================
if st.button(t("load_racelist"), use_container_width=True):
    params = dict(
        act="get_racelist",
        raceyear=raceyear,
        uname=UNAME_DEFAULT,
        ukey=UKEY_DEFAULT,
        clubno=clubno,
    )
    raw = call_api(params)
    if show_debug:
        st.subheader("get_racelist raw")
        st.code(json.dumps(raw, ensure_ascii=False, indent=2) if isinstance(raw, (dict, list)) else str(raw))

    racelist = find_list_deep(raw)
    if not racelist:
        st.warning(t("racelist_empty"))
    else:
        st.session_state["racelist"] = racelist
        st.success(t("racelist_loaded", n=len(racelist)))

# =========================
# 顯示賽事清單 + 勾選
# =========================
racelist = st.session_state.get("racelist", [])
if racelist:
    sample = racelist[0]
    k_raceno   = coalesce_key(sample, ["raceno","race_no","raceid","race_id","racenum"])
    k_racename = coalesce_key(sample, ["racename","race_name","name","title"])
    k_racedate = coalesce_key(sample, ["racedate","race_date","date","rdate"])

    st.subheader(t("racelist_table"))
    st.dataframe(pd.DataFrame(racelist), use_container_width=True)

    def label_of(r: Dict[str, Any]) -> str:
        name = str(r.get(k_racename) or "")
        date = str(r.get(k_racedate) or "")
        no   = normalize_raceno(r.get(k_raceno) or "")
        return f"{date} {name} ({no})".strip()

    options = [(label_of(r), normalize_raceno(r.get(k_raceno))) for r in racelist if r.get(k_raceno) is not None]
    labels  = [o[0] for o in options]
    values  = [o[1] for o in options]

    selected = st.multiselect(
        t("select_races"),
        options=values,
        format_func=lambda v: labels[values.index(v)] if v in values else v
    )

    # =========================
    # ③ 取得賽事明細並計算平均速度（缺場算 0）
    # =========================
    if st.button(t("calc_button"), type="primary", use_container_width=True, disabled=not selected):
        progress = st.progress(0)
        all_rows = []
        selected_str = [normalize_raceno(rn) for rn in selected]

        for i, rn in enumerate(selected_str, 1):
            raw = call_api(dict(
                act="get_race",
                raceno=rn,
                raceyear=raceyear,
                uname=UNAME_DEFAULT,
                ukey=UKEY_DEFAULT,
                clubno=clubno,
                APP="Y",
            ))
            rows = find_list_deep(raw)

            if rows:
                first = rows[0]
                k_id   = coalesce_key(first, ID_KEYS_CANDIDATE) or "pring_no"
                k_loft = coalesce_key(first, LOFT_KEYS_CANDIDATE)
                k_sex  = coalesce_key(first,  SEX_KEYS_CANDIDATE)

                for r in rows:
                    pid = r.get(k_id)
                    if not pid:
                        continue
                    sp_raw = next((r.get(sk) for sk in SPEED_KEYS_CANDIDATE if r.get(sk) not in (None, "")), None)
                    sp = parse_speed(sp_raw) if sp_raw is not None else 0.0

                    all_rows.append(dict(
                        raceno=str(rn),
                        pring_no=str(pid).strip(),
                        speed=float(sp) if sp is not None else 0.0,
                        loftname=r.get(k_loft) if k_loft else None,
                        sex=r.get(k_sex) if k_sex else None
                    ))
            else:
                st.info(t("info_no_detail", rn=rn))

            progress.progress(i/len(selected_str))

        if not all_rows:
            st.warning(t("no_rows"))
            st.stop()

        # ========== 聚合 ==========
        df_raw = pd.DataFrame(all_rows)
        df_raw["speed"] = pd.to_numeric(df_raw["speed"], errors="coerce").fillna(0.0)
        df_raw["pring_no"] = (
            df_raw["pring_no"].astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True)
        )
        df_raw = df_raw[df_raw["pring_no"].str.len() > 0]
        df_raw["raceno"] = df_raw["raceno"].map(normalize_raceno)

        per_race = (
            df_raw.groupby(["pring_no", "raceno"], as_index=False)
                  .agg(
                      loftname=("loftname", most_common),
                      sex=("sex", most_common),
                      speed=("speed", "max"),
                  )
        )

        # 轉寬表（避免 index 含 NaN 被 pivot 丟掉）
        index_cols = ["pring_no", "loftname", "sex"]
        for c in ["loftname", "sex"]:
            if per_race[c].notna().sum() == 0:
                index_cols.remove(c)
            else:
                per_race[c] = per_race[c].fillna("")

        wide = per_race.pivot_table(
            index=index_cols, columns="raceno", values="speed", aggfunc="max", fill_value=0.0
        ).reset_index()

        for rn in selected_str:
            if rn not in wide.columns:
                wide[rn] = 0.0

        total_speed = wide[selected_str].sum(axis=1)
        avg_speed = total_speed / float(len(selected_str))

        keep_cols = ["pring_no"]
        if "loftname" in wide.columns: keep_cols.append("loftname")
        if "sex" in wide.columns: keep_cols.append("sex")

        agg = wide[keep_cols].copy()
        agg["total_speed"] = total_speed
        agg["races_count"] = len(selected_str)
        agg["avg_speed"] = avg_speed
        if "loftname" in agg.columns: agg["loftname"] = agg["loftname"].fillna("")
        if "sex" in agg.columns: agg["sex"] = agg["sex"].fillna("")

        # 排序 + 加上排名欄位（由 1 開始）
        agg = agg.sort_values("avg_speed", ascending=False).reset_index(drop=True)
        agg.insert(0, "Rank", range(1, len(agg) + 1))

        st.subheader(t("result_title"))
        st.dataframe(agg, use_container_width=True)

        # ===== 匯出（欄名跟著語言；包含 Rank/名次）=====
        cols_map_en = {
            "Rank": "Rank",
            "pring_no": "Pigeon", "loftname": "Loft", "sex": "Sex",
            "total_speed": "Total Speed", "races_count": "Races", "avg_speed": "Avg Speed"
        }
        cols_map_zh = {
            "Rank": "名次",
            "pring_no": "腳環號", "loftname": "鴿舍", "sex": "性別",
            "total_speed": "總速度", "races_count": "場數", "avg_speed": "平均速度"
        }
        lang = st.session_state.get("lang", "en")
        export = agg.rename(columns=cols_map_en if lang == "en" else cols_map_zh)

        # CSV
        csv = export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(t("download_csv"), data=csv,
                           file_name="Topigeon_Ace_Pigeon_List.csv", mime="text/csv")

        # PDF
        if REPORTLAB_OK:
            try:
                pdf_bytes = build_pdf_from_df(export, title=t("pdf_title"))
                st.download_button(t("download_pdf"), data=pdf_bytes,
                                   file_name="Topigeon_Ace_Pigeon_List.pdf",
                                   mime="application/pdf")
            except Exception as e:
                st.warning(f"PDF export failed: {e}")
        else:
            st.info("To enable PDF export, install reportlab: pip install reportlab")
