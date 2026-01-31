import os
import io
import hashlib
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# =========================
# Config / Policy (FIXO)
# =========================
METRICS_POLICY = {
    "volume_col": "PACOTES_COLETADOS_TOTAL",
    "route_col": "FIRST_ROUTE_ID",
    "date_col": "DATES_DT",
    "week_col": "WEEK",  # se não existir, será derivado do DATES_DT como YYYY/WW (ISO)
    "default_driver_dimensions": ["FIRST_COMPANY_NAME", "WAREHOUSE_ID", "VEHICLE_TYPE_BUDGET"],
    "guardrails": {
        "drop_null_routes": True,
        "warn_if_negative_volume": True,
        "warn_if_routes_zero": True,
    },
}

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0  # fixo (não aparece na UI)

# =========================
# App Setup
# =========================
load_dotenv()
st.set_page_config(page_title="IPR Agent (Pacotes por rota)", layout="wide")
st.title("💬 IPR Agent — Pacotes por rota (IPR = Volume / Rotas)")

st.caption(
    "IPR (oficial) = sum(PACOTES_COLETADOS_TOTAL) / COUNTD(FIRST_ROUTE_ID). "
    "Escolha dois períodos (A vs B) nas configurações."
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
    """
    Gera WEEK no padrão YYYY/WW (ISO) se não existir ou se estiver vazio.
    """
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


def safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def calc_ipr(df: pd.DataFrame, volume_col: str, route_col: str, drop_null_routes: bool = True) -> Dict[str, float]:
    """
    IPR = sum(volume_col) / nunique(route_col)
    """
    dfx = df
    if drop_null_routes and route_col in dfx.columns:
        dfx = dfx[dfx[route_col].notna()].copy()

    volume = safe_sum(dfx[volume_col]) if volume_col in dfx.columns else 0.0
    rotas = safe_nunique(dfx[route_col]) if route_col in dfx.columns else 0

    ipr = (volume / rotas) if rotas > 0 else float("nan")
    return {"volume": volume, "rotas": float(rotas), "ipr": float(ipr)}


def make_range_slice(df: pd.DataFrame, date_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Recorte inclusivo por intervalo [start, end] (datas normalizadas).
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e < s:
        s, e = e, s
    mask = (df[date_col].dt.normalize() >= s) & (df[date_col].dt.normalize() <= e)
    return df[mask].copy()


def driver_decomposition_mix_perf(
    A: pd.DataFrame,
    B: pd.DataFrame,
    dim: str,
    volume_col: str,
    route_col: str,
    drop_null_routes: bool = True,
) -> pd.DataFrame:
    """
    Decomposição ΔIPR ≈ Δmix + Δperf por dimensão dim.
    w = share de rotas, r = IPR do grupo.
    """
    def prep(df: pd.DataFrame) -> pd.DataFrame:
        dfx = df
        if drop_null_routes and route_col in dfx.columns:
            dfx = dfx[dfx[route_col].notna()].copy()

        if dim not in dfx.columns:
            return pd.DataFrame(columns=[dim, "volume", "rotas", "ipr", "share_rotas"])

        g = dfx.groupby(dim, dropna=False)

        volume_g = g[volume_col].apply(safe_sum) if volume_col in dfx.columns else 0.0
        rotas_g = g[route_col].nunique(dropna=True) if route_col in dfx.columns else 0

        out = pd.DataFrame({dim: volume_g.index, "volume": volume_g.values, "rotas": rotas_g.values})
        out["ipr"] = out.apply(lambda r: (r["volume"] / r["rotas"]) if r["rotas"] > 0 else 0.0, axis=1)

        total_rotas = float(out["rotas"].sum())
        out["share_rotas"] = out["rotas"] / total_rotas if total_rotas > 0 else 0.0
        return out

    a = prep(A).rename(columns={"volume": "volume_A", "rotas": "rotas_A", "ipr": "ipr_A", "share_rotas": "wA"})
    b = prep(B).rename(columns={"volume": "volume_B", "rotas": "rotas_B", "ipr": "ipr_B", "share_rotas": "wB"})

    if a.empty and b.empty:
        return pd.DataFrame()

    merged = pd.merge(a, b, on=dim, how="outer")
    for c in ["volume_A", "rotas_A", "ipr_A", "wA", "volume_B", "rotas_B", "ipr_B", "wB"]:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    merged["mix_contrib"] = (merged["wA"] - merged["wB"]) * merged["ipr_B"]
    merged["perf_contrib"] = merged["wA"] * (merged["ipr_A"] - merged["ipr_B"])
    merged["total_contrib"] = merged["mix_contrib"] + merged["perf_contrib"]

    merged["abs_contrib"] = merged["total_contrib"].abs()
    merged = merged.sort_values("abs_contrib", ascending=False).drop(columns=["abs_contrib"])

    return merged


def format_history(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    trimmed = messages[-max_turns * 2 :]
    lines = []
    for m in trimmed:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines).strip()


def build_agent_prompt(
    user_question: str,
    label_A: str,
    label_B: str,
    summary_A: Dict[str, float],
    summary_B: Dict[str, float],
    drivers: Dict[str, pd.DataFrame],
    messages: List[Dict[str, str]],
) -> str:
    # Drivers como CSV (evita tabulate)
    driver_snippets = []
    for dim, df_dim in drivers.items():
        if df_dim is None or df_dim.empty:
            continue
        top = df_dim.head(5).copy()

        keep_cols = [
            dim,
            "total_contrib",
            "mix_contrib",
            "perf_contrib",
            "ipr_A",
            "ipr_B",
            "wA",
            "wB",
            "rotas_A",
            "rotas_B",
            "volume_A",
            "volume_B",
        ]
        keep_cols = [c for c in keep_cols if c in top.columns]
        top = top[keep_cols]

        driver_snippets.append(f"\nDIMENSÃO: {dim}\n{top.to_csv(index=False)}")

    driver_block = "\n".join(driver_snippets).strip() if driver_snippets else "Sem drivers calculados."

    history_txt = format_history(messages)

    sop = f"""
Você é um analista de dados especializado nesta base (granularidade por pallet/INBOUND_ID).
A métrica OFICIAL é:
- Volume = sum({METRICS_POLICY["volume_col"]})
- Rotas  = COUNTD({METRICS_POLICY["route_col"]}) = nunique({METRICS_POLICY["route_col"]})
- IPR    = Volume / Rotas  (Pacotes por rota)

COMPARAÇÃO (já escolhida pelo usuário):
- A = {label_A}
- B = {label_B}

REGRAS:
1) NÃO invente colunas.
2) NÃO chute números: tudo deve vir do dataframe.
3) Sempre explique a variação do IPR separando:
   - efeito de Volume (pacotes) vs efeito de Rotas
   - e, quando possível, decomposição Mix vs Performance por dimensões (transportadora, warehouse, tipo de veículo).
4) Formato de resposta OBRIGATÓRIO (use exatamente estes títulos):
   (1) Resumo executivo
   (2) Comparação e filtros
   (3) Componentes do IPR
   (4) Drivers do delta (Top 5)
   (5) Diagnóstico
   (6) Próximos passos

Você já tem números pré-calculados do recorte A e B e também drivers Top 5 por dimensão.
Use esses números como base e só calcule algo adicional se a pergunta pedir explicitamente.
""".strip()

    def fnum(x: float) -> str:
        if x != x:
            return "NaN"
        return f"{x:,.4f}"

    summary_block = f"""
NÚMEROS PRÉ-CALCULADOS:
A: volume={summary_A["volume"]:.0f}, rotas={summary_A["rotas"]:.0f}, ipr={fnum(summary_A["ipr"])}
B: volume={summary_B["volume"]:.0f}, rotas={summary_B["rotas"]:.0f}, ipr={fnum(summary_B["ipr"])}
ΔIPR = {fnum(summary_A["ipr"] - summary_B["ipr"])}
""".strip()

    parts = [sop, summary_block, "DRIVERS (Top 5 por dimensão):", driver_block]
    if history_txt:
        parts.append("\nHISTÓRICO RECENTE:\n" + history_txt)
    parts.append("\nPERGUNTA DO USUÁRIO:\n" + user_question)
    return "\n\n".join(parts).strip()


# =========================
# API Key e Modelo (somente via Secrets/env)
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
    st.session_state.agent = None  # recria agente quando DF muda

df = st.session_state.df
df = ensure_datetime(df, METRICS_POLICY["date_col"])
df = ensure_week_iso(df, METRICS_POLICY["date_col"], METRICS_POLICY["week_col"])
st.session_state.df = df

# Valida coluna de data
date_col = METRICS_POLICY["date_col"]
if date_col not in df.columns or not df[date_col].notna().any():
    st.error(f"Coluna de data '{date_col}' não encontrada ou inválida.")
    st.stop()

min_date = df[date_col].min().date()
max_date = df[date_col].max().date()

# =========================
# Sidebar — Datas de comparação (RANGES)
# =========================
st.sidebar.subheader("Datas de comparação (ranges)")

# Defaults: A = últimos 7 dias, B = 7 dias anteriores
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

# =========================
# Main Layout
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("📦 Base carregada")
    st.write(f"Linhas: **{len(df):,}** | Colunas: **{df.shape[1]}**")

    with st.expander("Ver colunas"):
        st.write(list(df.columns))

    st.dataframe(df.head(50), use_container_width=True)

with right:
    st.subheader("📊 Comparação A vs B (prévia)")

    # Recortes por intervalo (inclusive)
    A = make_range_slice(df, date_col, pd.Timestamp(A_start), pd.Timestamp(A_end))
    B = make_range_slice(df, date_col, pd.Timestamp(B_start), pd.Timestamp(B_end))

    label_A = f"{pd.Timestamp(A_start).strftime('%Y-%m-%d')} → {pd.Timestamp(A_end).strftime('%Y-%m-%d')}"
    label_B = f"{pd.Timestamp(B_start).strftime('%Y-%m-%d')} → {pd.Timestamp(B_end).strftime('%Y-%m-%d')}"

    volume_col = METRICS_POLICY["volume_col"]
    route_col = METRICS_POLICY["route_col"]

    # Warnings
    warnings = []
    if volume_col not in df.columns:
        warnings.append(f"Falta coluna de volume: {volume_col}")
    if route_col not in df.columns:
        warnings.append(f"Falta coluna de rotas: {route_col}")

    if volume_col in df.columns:
        if pd.to_numeric(df[volume_col], errors="coerce").fillna(0).lt(0).any() and METRICS_POLICY["guardrails"]["warn_if_negative_volume"]:
            warnings.append(f"Há valores negativos em {volume_col} (dado inconsistente).")

    if route_col in df.columns:
        null_rate_A = float(A[route_col].isna().mean()) if len(A) else 0.0
        null_rate_B = float(B[route_col].isna().mean()) if len(B) else 0.0
        if null_rate_A > 0.05 or null_rate_B > 0.05:
            warnings.append(
                f"Alta taxa de {route_col} nulo: A={null_rate_A:.1%} | B={null_rate_B:.1%} "
                "(pode distorcer denominador)."
            )

    for w in warnings:
        st.warning(w)

    # Métricas A e B
    summary_A = calc_ipr(A, volume_col, route_col, drop_null_routes=METRICS_POLICY["guardrails"]["drop_null_routes"])
    summary_B = calc_ipr(B, volume_col, route_col, drop_null_routes=METRICS_POLICY["guardrails"]["drop_null_routes"])

    delta_ipr = summary_A["ipr"] - summary_B["ipr"]
    delta_vol = summary_A["volume"] - summary_B["volume"]
    delta_rot = summary_A["rotas"] - summary_B["rotas"]

    c1, c2, c3 = st.columns(3)
    c1.metric("IPR (A)", f"{summary_A['ipr']:.4f}" if summary_A["ipr"] == summary_A["ipr"] else "NaN", delta=f"{delta_ipr:+.4f}")
    c2.metric("Volume (A)", f"{summary_A['volume']:.0f}", delta=f"{delta_vol:+.0f}")
    c3.metric("Rotas (A)", f"{summary_A['rotas']:.0f}", delta=f"{delta_rot:+.0f}")

    st.caption(f"A = **{label_A}** | B = **{label_B}**")

    # Drivers
    drivers: Dict[str, pd.DataFrame] = {}
    with st.expander("Drivers (Top 5) — decomposição Mix vs Performance"):
        for dim in METRICS_POLICY["default_driver_dimensions"]:
            if dim not in df.columns:
                st.info(f"Dimensão '{dim}' não existe na base — pulando.")
                continue
            ddf = driver_decomposition_mix_perf(
                A=A,
                B=B,
                dim=dim,
                volume_col=volume_col,
                route_col=route_col,
                drop_null_routes=METRICS_POLICY["guardrails"]["drop_null_routes"],
            )
            drivers[dim] = ddf
            st.markdown(f"**{dim}**")
            st.dataframe(ddf.head(10), use_container_width=True)

# =========================
# Chat
# =========================
st.divider()
st.subheader("💬 Chat com o agente (consciente dos períodos A/B)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_question = st.chat_input("Pergunte algo (ex.: 'por que o IPR caiu?')")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if st.session_state.agent is None:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,  # gpt-3.5-turbo
            temperature=DEFAULT_TEMPERATURE,  # fixo
        )

        st.session_state.agent = create_pandas_dataframe_agent(
            llm,
            st.session_state.df,
            agent_type="openai-functions",
            verbose=False,
            allow_dangerous_code=True,
        )

    prompt = build_agent_prompt(
        user_question=user_question,
        label_A=label_A,
        label_B=label_B,
        summary_A=summary_A,
        summary_B=summary_B,
        drivers=drivers,
        messages=st.session_state.messages,
    )

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
                    "- valide se as colunas PACOTES_COLETADOS_TOTAL e FIRST_ROUTE_ID existem\n"
                    "- tente uma pergunta mais específica\n"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Nota: o agente tem acesso ao dataframe e pode executar Python para responder. "
    "Use com cautela e apenas com dados confiáveis."
)
