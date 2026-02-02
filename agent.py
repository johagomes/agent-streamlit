import os
import io
import re
import hashlib
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# =========================
# CONTRATO v1 (base nova)
# =========================
METRICS_POLICY = {
    "volume_col": "PACOTES_REAL",
    "route_col": "ROUTE_ID_REAL",
    "date_col": "DATA",
    "week_col": "SEMANA",

    "guardrails": {
        "drop_null_routes": True,
        "warn_if_negative_volume": True,
        "warn_if_routes_zero": True,
    },

    # 6 dimensões fixas
    "dims6": [
        "CLUSTER_ID",
        "HUB",
        "TIPO_ROTA",
        "TRANSPORTADORA",
        "MODAL",
        "DIA_DA_SEMANA",
    ],

    # ✅ tabela canônica obrigatória
    "canonical_dim": "HUB",
}

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0  # fixo (não aparece na UI)

# Quantas linhas injetar no prompt (Top negativos)
PROMPT_TOP_N = 100

# Arredondamento de IPR nos cards
IPR_DECIMALS = 0  # 0 -> sem casas; se quiser 2, coloque 2


# =========================
# App Setup
# =========================
load_dotenv()
st.set_page_config(page_title="IPR Agent — Impacto (Perf/Mix/Total) — tabelas únicas", layout="wide")
st.title("💬 IPR Agent — Decomposição do ΔIPR (Performance + Mix) — tabelas fixas sem duplicidade")
st.caption(
    "IPR = sum(PACOTES_REAL) / COUNTD(ROUTE_ID_REAL). "
    "Decomposição midpoint/Shapley: Impacto_Performance + Impacto_Mix = Impacto_Total (fecha no ΔIPR)."
)


# =========================
# Helpers
# =========================
def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Formato não suportado. Envie .csv, .xlsx ou .xls")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def ensure_week_iso(df: pd.DataFrame, date_col: str, week_col: str) -> pd.DataFrame:
    out = df.copy()
    if week_col in out.columns and out[week_col].notna().any():
        return out
    if date_col not in out.columns:
        return out

    isocal = out[date_col].dt.isocalendar()
    out[week_col] = isocal["year"].astype(str) + "/" + isocal["week"].astype(str).str.zfill(2)
    return out


def safe_sum(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.fillna(0).sum())


def make_range_slice(df: pd.DataFrame, date_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e < s:
        s, e = e, s
    mask = (df[date_col].dt.normalize() >= s) & (df[date_col].dt.normalize() <= e)
    return df[mask].copy()


def infer_extra_numeric_metrics(df: pd.DataFrame, forbidden_cols: List[str]) -> List[str]:
    extras: List[str] = []
    for c in df.columns:
        if c in forbidden_cols:
            continue

        uc = str(c).upper()
        if "ID" in uc:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            extras.append(c)

    return sorted(extras)


def compute_aggregates(
    df: pd.DataFrame,
    group_cols: List[str],
    volume_col: str,
    route_col: str,
    extra_numeric_cols: List[str],
    drop_null_routes: bool,
) -> pd.DataFrame:
    dfx = df.copy()

    if drop_null_routes and route_col in dfx.columns:
        dfx = dfx[dfx[route_col].notna()].copy()

    fake_group = False
    if not group_cols:
        dfx["_ALL_"] = "ALL"
        group_cols = ["_ALL_"]
        fake_group = True

    gb = dfx.groupby(group_cols, dropna=False)
    out = pd.DataFrame(index=gb.size().index).reset_index()

    out["volume"] = gb[volume_col].apply(safe_sum).values if volume_col in dfx.columns else 0.0
    out["rotas"] = gb[route_col].nunique(dropna=True).values if route_col in dfx.columns else 0
    out["ipr"] = out.apply(lambda r: (r["volume"] / r["rotas"]) if r["rotas"] else float("nan"), axis=1)

    for col in extra_numeric_cols:
        out[col] = gb[col].apply(safe_sum).values if col in dfx.columns else 0.0

    if fake_group:
        out = out.drop(columns=["_ALL_"], errors="ignore")

    return out


def merge_A_B(agg_A: pd.DataFrame, agg_B: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if not group_cols:
        agg_A["_k"] = 1
        agg_B["_k"] = 1
        group_cols = ["_k"]

    m = pd.merge(agg_A, agg_B, on=group_cols, how="outer", suffixes=("_A", "_B"))

    if "_k" in m.columns:
        m = m.drop(columns=["_k"], errors="ignore")

    return m


def add_deltas(m: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    out = m.copy()
    for col in metric_cols:
        a = f"{col}_A"
        b = f"{col}_B"

        if a not in out.columns:
            out[a] = 0.0
        if b not in out.columns:
            out[b] = 0.0

        out[a] = pd.to_numeric(out[a], errors="coerce")
        out[b] = pd.to_numeric(out[b], errors="coerce")

        out[f"{col}_delta"] = out[a] - out[b]
        out[f"{col}_delta_pct"] = out.apply(
            lambda r: (r[a] / r[b] - 1.0) if pd.notna(r[b]) and r[b] not in (0, 0.0) else float("nan"),
            axis=1,
        )
    return out


def totals_ipr(df: pd.DataFrame, volume_col: str, route_col: str, drop_null_routes: bool) -> Dict[str, float]:
    dfx = df.copy()
    if drop_null_routes and route_col in dfx.columns:
        dfx = dfx[dfx[route_col].notna()].copy()

    vol = safe_sum(dfx[volume_col]) if volume_col in dfx.columns else 0.0
    rotas = float(dfx[route_col].nunique(dropna=True)) if route_col in dfx.columns else 0.0
    ipr = (vol / rotas) if rotas else float("nan")
    return {"volume": float(vol), "rotas": float(rotas), "ipr": float(ipr)}


def add_shares_and_impacts_midpoint(table: pd.DataFrame, total_rotas_A: float, total_rotas_B: float) -> pd.DataFrame:
    out = table.copy()

    for c in ["rotas_A", "rotas_B", "ipr_A", "ipr_B"]:
        if c not in out.columns:
            out[c] = 0.0

    out["rotas_A"] = pd.to_numeric(out["rotas_A"], errors="coerce").fillna(0.0)
    out["rotas_B"] = pd.to_numeric(out["rotas_B"], errors="coerce").fillna(0.0)
    out["ipr_A"] = pd.to_numeric(out["ipr_A"], errors="coerce")
    out["ipr_B"] = pd.to_numeric(out["ipr_B"], errors="coerce")

    out["SHARE_ROTAS_A"] = (out["rotas_A"] / total_rotas_A) if total_rotas_A else 0.0
    out["SHARE_ROTAS_B"] = (out["rotas_B"] / total_rotas_B) if total_rotas_B else 0.0

    avg_share = (out["SHARE_ROTAS_A"] + out["SHARE_ROTAS_B"]) / 2.0
    avg_ipr = (out["ipr_A"].fillna(0.0) + out["ipr_B"].fillna(0.0)) / 2.0

    out["IMPACTO_PERFORMANCE"] = avg_share * (out["ipr_A"].fillna(0.0) - out["ipr_B"].fillna(0.0))
    out["IMPACTO_MIX"] = avg_ipr * (out["SHARE_ROTAS_A"] - out["SHARE_ROTAS_B"])
    out["IMPACTO_TOTAL"] = out["IMPACTO_PERFORMANCE"] + out["IMPACTO_MIX"]

    return out


def detect_table_name_from_text(text: str, available_names: List[str]) -> Optional[str]:
    t = (text or "").strip().lower().replace("×", "x")
    t = re.sub(r"\s+", " ", t)

    for name in available_names:
        n = name.lower().replace("×", "x")
        n = re.sub(r"\s+", " ", n)
        if n in t:
            return name

    return None


def format_history(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    trimmed = messages[-max_turns * 2 :]
    lines = []
    for m in trimmed:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines).strip()


def reduce_cols_for_prompt(df_in: pd.DataFrame, dims_present: List[str], include_table_col: bool) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in

    cols = []
    if include_table_col and "TABELA" in df_in.columns:
        cols.append("TABELA")

    for d in dims_present:
        if d in df_in.columns:
            cols.append(d)

    essentials = [
        "IMPACTO_TOTAL", "IMPACTO_PERFORMANCE", "IMPACTO_MIX",
        "SHARE_ROTAS_A", "SHARE_ROTAS_B",
        "ipr_A", "ipr_B", "ipr_delta",
        "volume_A", "volume_B", "volume_delta",
        "rotas_A", "rotas_B", "rotas_delta",
    ]
    for c in essentials:
        if c in df_in.columns:
            cols.append(c)

    cols = list(dict.fromkeys(cols))
    return df_in[cols].copy()


def build_agent_prompt(
    user_question: str,
    label_A: str,
    label_B: str,
    totals_A: Dict[str, float],
    totals_B: Dict[str, float],
    delta_ipr_total: float,
    canonical_table_name: str,
    closure_summary: Dict[str, float],
    inj_global_top: pd.DataFrame,
    inj_table_name: str,
    inj_table_top: pd.DataFrame,
    messages: List[Dict[str, str]],
) -> str:
    def fnum(x: float) -> str:
        if x != x:
            return "NaN"
        return f"{x:,.6f}"

    base = f"""
Você é um analista de dados especializado nesta base.
Métrica OFICIAL:
- Volume = sum({METRICS_POLICY["volume_col"]})
- Rotas  = COUNTD({METRICS_POLICY["route_col"]}) = nunique({METRICS_POLICY["route_col"]})
- IPR    = Volume / Rotas

Comparação por período:
- A = {label_A}
- B = {label_B}

TOTAIS:
A: volume={totals_A["volume"]:.0f}, rotas={totals_A["rotas"]:.0f}, ipr={fnum(totals_A["ipr"])}
B: volume={totals_B["volume"]:.0f}, rotas={totals_B["rotas"]:.0f}, ipr={fnum(totals_B["ipr"])}
ΔIPR_TOTAL = {fnum(delta_ipr_total)}

Tabela canônica de fechamento: {canonical_table_name}

Decomposição midpoint/Shapley por grupo (FECHA no ΔIPR):
SHARE_ROTAS_A = rotas_grupo_A / rotas_total_A
SHARE_ROTAS_B = rotas_grupo_B / rotas_total_B

IMPACTO_PERFORMANCE = ((SHARE_ROTAS_A + SHARE_ROTAS_B)/2) * (IPR_A - IPR_B)
IMPACTO_MIX         = ((IPR_A + IPR_B)/2) * (SHARE_ROTAS_A - SHARE_ROTAS_B)
IMPACTO_TOTAL       = IMPACTO_PERFORMANCE + IMPACTO_MIX

RESUMO DE FECHAMENTO (canônico):
ΣPERFORMANCE = {fnum(closure_summary.get("sum_perf", float("nan")))}
ΣMIX         = {fnum(closure_summary.get("sum_mix", float("nan")))}
ΣTOTAL       = {fnum(closure_summary.get("sum_total", float("nan")))}
RESIDUAL (ΔIPR - ΣTOTAL) = {fnum(closure_summary.get("residual", float("nan")))}

IMPORTANTE:
- Você TEM acesso ao dataframe df (granular) e pode recalcular/validar qualquer coisa.
- Abaixo estão rankings (Top {PROMPT_TOP_N}) com os MAIORES impactos negativos por IMPACTO_TOTAL.

Formato obrigatório:
(1) Resumo executivo
(2) Comparação e filtros
(3) Fechamento do ΔIPR (Perf vs Mix)
(4) Ranking por impacto (Top negativos) + recomendações
(5) Diagnóstico
(6) Próximos passos
""".strip()

    global_csv = inj_global_top.to_csv(index=False) if inj_global_top is not None else ""
    table_csv = inj_table_top.to_csv(index=False) if (inj_table_top is not None and not inj_table_top.empty) else ""

    hist = format_history(messages)

    parts = [base, f"TOP {PROMPT_TOP_N} GLOBAL (pior IMPACTO_TOTAL):", global_csv]

    if inj_table_name:
        parts += [f"TOP {PROMPT_TOP_N} — {inj_table_name} (pior IMPACTO_TOTAL):", table_csv]

    if hist:
        parts.append("HISTÓRICO RECENTE:\n" + hist)

    parts.append("PERGUNTA DO USUÁRIO:\n" + user_question)

    return "\n\n".join([p for p in parts if p]).strip()


# =========================
# API Key e Modelo (somente via Secrets/env) — NÃO aparece na UI
# =========================
api_key = ""
try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    api_key = ""
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY", "")

model_name = DEFAULT_MODEL
try:
    model_name = st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL)
except Exception:
    model_name = DEFAULT_MODEL

if not api_key:
    st.error(
        "OPENAI_API_KEY não encontrada. Defina no Streamlit Secrets (TOML) ou como variável de ambiente.\n\n"
        'Ex.: OPENAI_API_KEY = "sua-chave"'
    )
    st.stop()


# =========================
# Sidebar — Configurações
# =========================
st.sidebar.header("Configurações")
uploaded = st.sidebar.file_uploader("Envie sua base (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])


# =========================
# Session State
# =========================
if "df" not in st.session_state:
    st.session_state.df = None
if "df_id" not in st.session_state:
    st.session_state.df_id = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tables_cache" not in st.session_state:
    st.session_state.tables_cache = {}


# =========================
# Load Data
# =========================
if uploaded is None:
    st.info("Envie o arquivo na barra lateral para começar.")
    st.stop()

raw = uploaded.getvalue()
df_id = file_sha256(raw)

if st.session_state.df is None or st.session_state.df_id != df_id:
    df = read_uploaded_dataframe(uploaded)
    st.session_state.df = df
    st.session_state.df_id = df_id
    st.session_state.agent = None
    st.session_state.tables_cache = {}

df = st.session_state.df
df = ensure_datetime(df, METRICS_POLICY["date_col"])
df = ensure_week_iso(df, METRICS_POLICY["date_col"], METRICS_POLICY["week_col"])
st.session_state.df = df

date_col = METRICS_POLICY["date_col"]
if date_col not in df.columns or not df[date_col].notna().any():
    st.error(f"Coluna de data '{date_col}' não encontrada ou inválida.")
    st.stop()

min_date = df[date_col].min().date()
max_date = df[date_col].max().date()

volume_col = METRICS_POLICY["volume_col"]
route_col = METRICS_POLICY["route_col"]
drop_null_routes = METRICS_POLICY["guardrails"]["drop_null_routes"]

# =========================
# Sidebar — Datas de comparação (ranges)
# =========================
st.sidebar.subheader("Datas de comparação (ranges)")

default_A_end = max_date
default_A_start = (pd.Timestamp(max_date) - pd.Timedelta(days=6)).date()
if default_A_start < min_date:
    default_A_start = min_date

default_B_end = (pd.Timestamp(default_A_start) - pd.Timedelta(days=1)).date()
default_B_start = (pd.Timestamp(default_B_end) - pd.Timedelta(days=6)).date()
if default_B_end < min_date:
    default_B_end = min_date
if default_B_start < min_date:
    default_B_start = min_date

st.sidebar.markdown("**Período A**")
A_start = st.sidebar.date_input("Início A", value=default_A_start, min_value=min_date, max_value=max_date, key="A_start")
A_end = st.sidebar.date_input("Fim A", value=default_A_end, min_value=min_date, max_value=max_date, key="A_end")

st.sidebar.markdown("**Período B**")
B_start = st.sidebar.date_input("Início B", value=default_B_start, min_value=min_date, max_value=max_date, key="B_start")
B_end = st.sidebar.date_input("Fim B", value=default_B_end, min_value=min_date, max_value=max_date, key="B_end")

label_A = f"{pd.Timestamp(A_start).strftime('%Y-%m-%d')} → {pd.Timestamp(A_end).strftime('%Y-%m-%d')}"
label_B = f"{pd.Timestamp(B_start).strftime('%Y-%m-%d')} → {pd.Timestamp(B_end).strftime('%Y-%m-%d')}"

# =========================
# Recortes A e B
# =========================
A_df = make_range_slice(df, date_col, pd.Timestamp(A_start), pd.Timestamp(A_end))
B_df = make_range_slice(df, date_col, pd.Timestamp(B_start), pd.Timestamp(B_end))

# =========================
# Dimensões e extras numéricos
# =========================
dims6_present = [d for d in METRICS_POLICY["dims6"] if d in df.columns]
if len(dims6_present) < 2:
    st.error("Não encontrei as dimensões esperadas na base. Verifique os nomes das colunas.")
    st.stop()

# ✅ remove duplicidade: só pares NÃO-ordenados (combinações) + diagonais
# ex.: (HUB, CLUSTER_ID) existe, mas (CLUSTER_ID, HUB) não.
unique_pairs: List[Tuple[str, str]] = []
for i, d1 in enumerate(dims6_present):
    for j, d2 in enumerate(dims6_present):
        if j < i:
            continue
        unique_pairs.append((d1, d2))

# Ex.: com 6 dims -> 6 diagonais + C(6,2)=15 pares = 21 tabelas
# (reduz, evita duplicidade no global)

forbidden_for_extras = dims6_present + [METRICS_POLICY["date_col"], METRICS_POLICY["week_col"], route_col, volume_col]
extra_metrics = infer_extra_numeric_metrics(df, forbidden_cols=forbidden_for_extras)
metric_cols_for_delta = ["volume", "rotas", "ipr"] + extra_metrics

# =========================
# Cache key
# =========================
cache_key = (
    st.session_state.df_id,
    str(A_start), str(A_end),
    str(B_start), str(B_end),
    "|".join(dims6_present),
    "|".join(metric_cols_for_delta),
    "unique_pairs_only",
    f"canonical={METRICS_POLICY['canonical_dim']}",
)

if st.session_state.tables_cache.get("_cache_key") != cache_key:
    totals_A = totals_ipr(A_df, volume_col, route_col, drop_null_routes)
    totals_B = totals_ipr(B_df, volume_col, route_col, drop_null_routes)
    delta_ipr_total = float(totals_A["ipr"] - totals_B["ipr"])

    total_rotas_A = float(totals_A["rotas"])
    total_rotas_B = float(totals_B["rotas"])

    tables: Dict[str, pd.DataFrame] = {}

    for d1, d2 in unique_pairs:
        group_cols = [d1] if d1 == d2 else [d1, d2]
        name = f"{d1} × {d2}"

        agg_A = compute_aggregates(
            df=A_df,
            group_cols=group_cols,
            volume_col=volume_col,
            route_col=route_col,
            extra_numeric_cols=extra_metrics,
            drop_null_routes=drop_null_routes,
        )
        agg_B = compute_aggregates(
            df=B_df,
            group_cols=group_cols,
            volume_col=volume_col,
            route_col=route_col,
            extra_numeric_cols=extra_metrics,
            drop_null_routes=drop_null_routes,
        )

        merged = merge_A_B(agg_A, agg_B, group_cols=group_cols)
        merged = add_deltas(merged, metric_cols=metric_cols_for_delta)
        merged = add_shares_and_impacts_midpoint(merged, total_rotas_A=total_rotas_A, total_rotas_B=total_rotas_B)
        merged = merged.sort_values("IMPACTO_TOTAL", ascending=True)

        tables[name] = merged

    # Global sem duplicidade (concatena as tabelas únicas)
    global_rows = []
    for name, t in tables.items():
        if t is None or t.empty:
            continue
        tmp = t.copy()
        tmp.insert(0, "TABELA", name)
        global_rows.append(tmp)

    global_df = pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame()

    # ✅ Canonical HUB×HUB obrigatória
    canonical_dim = METRICS_POLICY["canonical_dim"]
    canonical_name = f"{canonical_dim} × {canonical_dim}"
    if canonical_name not in tables:
        st.error(f"Não encontrei a tabela canônica obrigatória: {canonical_name}.")
        st.stop()

    canon = tables[canonical_name]

    sum_perf = float(pd.to_numeric(canon["IMPACTO_PERFORMANCE"], errors="coerce").fillna(0.0).sum())
    sum_mix = float(pd.to_numeric(canon["IMPACTO_MIX"], errors="coerce").fillna(0.0).sum())
    sum_total = float(pd.to_numeric(canon["IMPACTO_TOTAL"], errors="coerce").fillna(0.0).sum())
    residual = float(delta_ipr_total - sum_total)

    closure = {
        "canonical_table": canonical_name,
        "sum_perf": sum_perf,
        "sum_mix": sum_mix,
        "sum_total": sum_total,
        "residual": residual,
    }

    st.session_state.tables_cache = {
        "_cache_key": cache_key,
        "tables": tables,
        "global": global_df,
        "totals_A": totals_A,
        "totals_B": totals_B,
        "delta_ipr_total": delta_ipr_total,
        "closure": closure,
        "canonical_table": canonical_name,
    }
else:
    tables = st.session_state.tables_cache["tables"]
    global_df = st.session_state.tables_cache["global"]
    totals_A = st.session_state.tables_cache["totals_A"]
    totals_B = st.session_state.tables_cache["totals_B"]
    delta_ipr_total = st.session_state.tables_cache["delta_ipr_total"]
    closure = st.session_state.tables_cache["closure"]
    canonical_name = st.session_state.tables_cache["canonical_table"]

# =========================
# Big numbers (A e B) + arredondamento de IPR
# =========================
def fmt_ipr(x: float) -> str:
    if x != x:
        return "NaN"
    if IPR_DECIMALS == 0:
        return f"{x:,.0f}"
    return f"{x:,.{IPR_DECIMALS}f}"


delta_vol = totals_A["volume"] - totals_B["volume"]
delta_rot = totals_A["rotas"] - totals_B["rotas"]

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("IPR (A)", fmt_ipr(totals_A["ipr"]), delta=fmt_ipr(delta_ipr_total))
col2.metric("IPR (B)", fmt_ipr(totals_B["ipr"]))
col3.metric("Volume (A)", f"{totals_A['volume']:.0f}", delta=f"{delta_vol:+.0f}")
col4.metric("Volume (B)", f"{totals_B['volume']:.0f}")
col5.metric("Rotas (A)", f"{totals_A['rotas']:.0f}", delta=f"{delta_rot:+.0f}")
col6.metric("Rotas (B)", f"{totals_B['rotas']:.0f}")

st.caption(f"A = **{label_A}** | B = **{label_B}**")

# =========================
# Fechamento (HUB×HUB)
# =========================
with st.expander("✅ Fechamento do ΔIPR (tabela canônica HUB×HUB)", expanded=True):
    st.markdown(
        f"""
**Tabela canônica usada para fechar o ΔIPR:** `{canonical_name}`

- **ΔIPR_TOTAL** = `{delta_ipr_total:.6f}`
- **Σ Impacto Performance** = `{closure.get('sum_perf', float('nan')):.6f}`
- **Σ Impacto Mix** = `{closure.get('sum_mix', float('nan')):.6f}`
- **Σ Impacto Total** = `{closure.get('sum_total', float('nan')):.6f}`
- **Residual (ΔIPR - ΣTotal)** = `{closure.get('residual', float('nan')):.6f}`
"""
    )

# =========================
# Global Top 100 negativos (IMPACTO_TOTAL) — sem duplicidade
# =========================
st.subheader(f"🔥 Global Top {PROMPT_TOP_N} impactos negativos (IMPACTO_TOTAL) — sem pares duplicados")

if global_df is None or global_df.empty:
    st.info("Sem dados para ranking global.")
else:
    global_top = global_df.sort_values("IMPACTO_TOTAL", ascending=True).head(PROMPT_TOP_N).copy()
    keep = ["TABELA"] + [d for d in dims6_present if d in global_top.columns] + [
        "IMPACTO_TOTAL", "IMPACTO_PERFORMANCE", "IMPACTO_MIX",
        "SHARE_ROTAS_A", "SHARE_ROTAS_B",
        "ipr_A", "ipr_B", "ipr_delta",
        "volume_A", "volume_B", "volume_delta",
        "rotas_A", "rotas_B", "rotas_delta",
    ]
    keep = list(dict.fromkeys([c for c in keep if c in global_top.columns]))
    st.dataframe(global_top[keep], use_container_width=True)

# =========================
# Tabelas fixas (agora são 21 se tiver 6 dims)
# =========================
st.subheader("📌 Tabelas fixas únicas — ordenadas por pior IMPACTO_TOTAL (mais negativo)")

st.caption(
    "Removemos duplicidades do tipo (A×B) e (B×A). Mantemos apenas combinações não-ordenadas + diagonais. "
    "Isso reduz tabelas (ex.: 6 dims → 21 tabelas) e evita duplicidade no ranking global."
)

for name in sorted(tables.keys()):
    with st.expander(f"🧾 {name}", expanded=False):
        t = tables[name]
        st.dataframe(t, use_container_width=True)

        csv = t.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar CSV desta tabela",
            data=csv,
            file_name=f"tabela_{name.replace(' × ', '_x_')}.csv",
            mime="text/csv",
        )

# =========================
# Injeção pro chat
# =========================
def make_injection_tables_for_prompt(user_text: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    available_names = list(tables.keys())
    cited = detect_table_name_from_text(user_text, available_names)

    if global_df is None or global_df.empty:
        global_top = pd.DataFrame()
    else:
        global_top = global_df.sort_values("IMPACTO_TOTAL", ascending=True).head(PROMPT_TOP_N).copy()
        global_top = reduce_cols_for_prompt(global_top, dims6_present, include_table_col=True)

    cited_name = ""
    table_top = pd.DataFrame()
    if cited and cited in tables:
        cited_name = cited
        t = tables[cited].sort_values("IMPACTO_TOTAL", ascending=True).head(PROMPT_TOP_N).copy()
        table_top = reduce_cols_for_prompt(t, dims6_present, include_table_col=False)

    return cited_name, table_top, global_top


def build_prompt_wrapper(user_question: str, cited_name: str, cited_top: pd.DataFrame, global_top: pd.DataFrame) -> str:
    closure_summary = {
        "sum_perf": closure.get("sum_perf", float("nan")),
        "sum_mix": closure.get("sum_mix", float("nan")),
        "sum_total": closure.get("sum_total", float("nan")),
        "residual": closure.get("residual", float("nan")),
    }

    return build_agent_prompt(
        user_question=user_question,
        label_A=label_A,
        label_B=label_B,
        totals_A=totals_A,
        totals_B=totals_B,
        delta_ipr_total=float(delta_ipr_total),
        canonical_table_name=canonical_name,
        closure_summary=closure_summary,
        inj_global_top=global_top,
        inj_table_name=cited_name,
        inj_table_top=cited_top,
        messages=st.session_state.messages,
    )


# =========================
# Chat
# =========================
st.divider()
st.subheader("💬 Chat com o agente (Modo Forte: Top 100 por IMPACTO_TOTAL + fechamento HUB×HUB)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_question = st.chat_input(
    "Pergunte (ex.: 'Explique o delta usando HUB × HUB' ou 'Top 10 causas do delta global')"
)

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if st.session_state.agent is None:
        llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=DEFAULT_TEMPERATURE)
        st.session_state.agent = create_pandas_dataframe_agent(
            llm,
            st.session_state.df,
            agent_type="openai-functions",
            verbose=False,
            allow_dangerous_code=True,
        )

    cited_name, cited_top, global_top = make_injection_tables_for_prompt(user_question)
    prompt = build_prompt_wrapper(user_question, cited_name, cited_top, global_top)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            try:
                result = st.session_state.agent.invoke(prompt)
                if isinstance(result, dict):
                    answer = result.get("output") or result.get("final") or str(result)
                else:
                    answer = str(result)
            except Exception as e:
                answer = (
                    "Deu erro ao executar a análise.\n\n"
                    f"**Detalhes:** {e}\n\n"
                    "Dicas:\n"
                    "- cite uma tabela (ex.: 'HUB × HUB' ou 'HUB × CLUSTER_ID')\n"
                    "- valide se PACOTES_REAL e ROUTE_ID_REAL existem\n"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Agora: (1) tabela canônica fixa HUB×HUB; (2) cards trazem A e B; (3) removemos tabelas duplicadas (A×B e B×A)."
)
