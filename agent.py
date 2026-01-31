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
    "guardrails": {
        "drop_null_routes": True,
        "warn_if_negative_volume": True,
        "warn_if_routes_zero": True,
    },
    # dimensões sugeridas (só pra facilitar o seletor)
    "suggested_dims": ["WAREHOUSE_ID", "FIRST_COMPANY_NAME", "VEHICLE_TYPE_BUDGET", "DESTINATION_ID", "ZON_ZONE_NAME"],
}

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0  # fixo (não aparece na UI)

# =========================
# App Setup
# =========================
load_dotenv()
st.set_page_config(page_title="IPR Agent (Pacotes por rota)", layout="wide")
st.title("💬 IPR Agent — Pacotes por rota (IPR = Volume / Rotas)")
st.caption("IPR (oficial) = sum(PACOTES_COLETADOS_TOTAL) / COUNTD(FIRST_ROUTE_ID).")

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


def compute_aggregates(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_keys: List[str],
    extra_numeric_cols: List[str],
    volume_col: str,
    route_col: str,
    drop_null_routes: bool,
) -> pd.DataFrame:
    """
    Retorna tabela agregada com as métricas pedidas.
    Métricas oficiais suportadas:
      - volume: sum(PACOTES_COLETADOS_TOTAL)
      - rotas:  nunique(FIRST_ROUTE_ID)
      - ipr:    volume/rotas
    Métricas extras numéricas: sum(coluna)
    """
    dfx = df.copy()

    if drop_null_routes and route_col in dfx.columns:
        dfx = dfx[dfx[route_col].notna()].copy()

    # Se não tem grupo, cria um grupo artificial para facilitar o agg e depois removemos
    fake_group = False
    if not group_cols:
        dfx["_ALL_"] = "ALL"
        group_cols = ["_ALL_"]
        fake_group = True

    gb = dfx.groupby(group_cols, dropna=False)

    out = pd.DataFrame(index=gb.size().index).reset_index()

    # Oficiais
    if "volume" in metric_keys:
        if volume_col in dfx.columns:
            out["volume"] = gb[volume_col].apply(safe_sum).values
        else:
            out["volume"] = 0.0

    if "rotas" in metric_keys or "ipr" in metric_keys:
        if route_col in dfx.columns:
            out["rotas"] = gb[route_col].nunique(dropna=True).values
        else:
            out["rotas"] = 0

    if "ipr" in metric_keys:
        # garante volume calculado
        if "volume" not in out.columns:
            if volume_col in dfx.columns:
                out["volume"] = gb[volume_col].apply(safe_sum).values
            else:
                out["volume"] = 0.0

        out["ipr"] = out.apply(lambda r: (r["volume"] / r["rotas"]) if r["rotas"] else float("nan"), axis=1)

    # Extras (numéricas)
    for col in extra_numeric_cols:
        if col in dfx.columns:
            out[col] = gb[col].apply(safe_sum).values
        else:
            out[col] = 0.0

    # remove grupo artificial
    if fake_group:
        out = out.drop(columns=["_ALL_"], errors="ignore")

    return out


def merge_and_delta(
    agg_A: pd.DataFrame,
    agg_B: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Junta A e B e cria colunas *_A, *_B, *_delta, *_delta_pct
    """
    if not group_cols:
        # garante merge 1x1
        agg_A["_k"] = 1
        agg_B["_k"] = 1
        group_cols = ["_k"]

    m = pd.merge(agg_A, agg_B, on=group_cols, how="outer", suffixes=("_A", "_B"))

    # normaliza NaNs
    for col in metric_cols:
        a = f"{col}_A"
        b = f"{col}_B"
        if a not in m.columns:
            m[a] = 0.0
        if b not in m.columns:
            m[b] = 0.0
        m[a] = pd.to_numeric(m[a], errors="coerce")
        m[b] = pd.to_numeric(m[b], errors="coerce")

        m[f"{col}_delta"] = m[a] - m[b]
        m[f"{col}_delta_pct"] = m.apply(
            lambda r: (r[a] / r[b] - 1.0) if pd.notna(r[b]) and r[b] not in (0, 0.0) else float("nan"),
            axis=1,
        )

    if "_k" in m.columns:
        m = m.drop(columns=["_k"], errors="ignore")

    return m


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
    comparison_table: pd.DataFrame,
    group_cols: List[str],
    chosen_metrics: List[str],
    messages: List[Dict[str, str]],
) -> str:
    # Tabela dinâmica como CSV (evita tabulate)
    table_csv = comparison_table.head(200).to_csv(index=False)

    sop = f"""
Você é um analista de dados especializado nesta base.
Métrica OFICIAL:
- Volume = sum({METRICS_POLICY["volume_col"]})
- Rotas  = COUNTD({METRICS_POLICY["route_col"]}) = nunique({METRICS_POLICY["route_col"]})
- IPR    = Volume / Rotas  (Pacotes por rota)

COMPARAÇÃO (já escolhida):
- A = {label_A}
- B = {label_B}

O usuário escolheu:
- Dimensões (colunas de comparação): {group_cols if group_cols else "nenhuma (visão total)"}
- Métricas: {chosen_metrics}

REGRAS:
1) NÃO invente colunas.
2) NÃO chute números: tudo deve vir do dataframe/tabela calculada.
3) Sempre explique a variação do IPR separando: efeito Volume vs Rotas (quando IPR estiver entre as métricas).
4) Formato de resposta OBRIGATÓRIO:
   (1) Resumo executivo
   (2) Comparação e filtros
   (3) Componentes do IPR
   (4) Deltas e ranking por dimensão
   (5) Diagnóstico
   (6) Próximos passos
""".strip()

    def fnum(x: float) -> str:
        if x != x:
            return "NaN"
        return f"{x:,.4f}"

    summary_block = f"""
NÚMEROS TOTAIS (pré-calculados):
A: volume={summary_A["volume"]:.0f}, rotas={summary_A["rotas"]:.0f}, ipr={fnum(summary_A["ipr"])}
B: volume={summary_B["volume"]:.0f}, rotas={summary_B["rotas"]:.0f}, ipr={fnum(summary_B["ipr"])}
ΔIPR = {fnum(summary_A["ipr"] - summary_B["ipr"])}
""".strip()

    history_txt = format_history(messages)

    prompt = "\n\n".join(
        [
            sop,
            summary_block,
            "TABELA DINÂMICA (até 200 linhas) — contém *_A, *_B, *_delta e *_delta_pct:",
            table_csv,
            ("HISTÓRICO RECENTE:\n" + history_txt) if history_txt else "",
            "PERGUNTA DO USUÁRIO:\n" + user_question,
        ]
    ).strip()

    return prompt


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
# Sidebar — Datas de comparação (ranges)
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

# Labels
label_A = f"{pd.Timestamp(A_start).strftime('%Y-%m-%d')} → {pd.Timestamp(A_end).strftime('%Y-%m-%d')}"
label_B = f"{pd.Timestamp(B_start).strftime('%Y-%m-%d')} → {pd.Timestamp(B_end).strftime('%Y-%m-%d')}"

# =========================
# Recortes A e B
# =========================
A_df = make_range_slice(df, date_col, pd.Timestamp(A_start), pd.Timestamp(A_end))
B_df = make_range_slice(df, date_col, pd.Timestamp(B_start), pd.Timestamp(B_end))

# =========================
# Comparação dinâmica (UI)
# =========================
st.subheader("📊 Comparação A vs B (dinâmica)")

# Controles da comparação dinâmica
all_cols = list(df.columns)

# dims sugeridas primeiro, mas pode escolher qualquer coluna
suggested = [c for c in METRICS_POLICY["suggested_dims"] if c in all_cols]
other_cols = [c for c in all_cols if c not in suggested]
dim_options = suggested + other_cols

group_cols = st.multiselect(
    "Colunas de comparação (dimensões)",
    options=dim_options,
    default=[c for c in ["WAREHOUSE_ID", "FIRST_COMPANY_NAME"] if c in all_cols],
    help="Escolha por quais colunas você quer comparar (ex.: WAREHOUSE_ID, FIRST_COMPANY_NAME).",
)

# Métricas disponíveis:
# - Oficiais: ipr, volume, rotas
# - Extras: quaisquer colunas numéricas (somadas)
official_metric_options = ["ipr", "volume", "rotas"]

numeric_cols = []
for c in all_cols:
    if c in [METRICS_POLICY["volume_col"]]:
        continue  # já está coberta em "volume"
    # detecta numérico por tentativa
    try:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)
    except Exception:
        pass

metric_options = official_metric_options + numeric_cols

chosen_metrics = st.multiselect(
    "Métricas para comparar",
    options=metric_options,
    default=["ipr", "volume", "rotas"],
    help="As métricas oficiais são ipr, volume e rotas. Outras colunas numéricas entram como soma.",
)

# Separa métricas oficiais e extras
metric_keys = [m for m in chosen_metrics if m in official_metric_options]
extra_metrics = [m for m in chosen_metrics if m not in official_metric_options]

volume_col = METRICS_POLICY["volume_col"]
route_col = METRICS_POLICY["route_col"]
drop_null_routes = METRICS_POLICY["guardrails"]["drop_null_routes"]

# Warnings básicos
warnings = []
if volume_col not in df.columns:
    warnings.append(f"Falta coluna de volume: {volume_col}")
if route_col not in df.columns:
    warnings.append(f"Falta coluna de rotas: {route_col}")

if volume_col in df.columns and METRICS_POLICY["guardrails"]["warn_if_negative_volume"]:
    if pd.to_numeric(df[volume_col], errors="coerce").fillna(0).lt(0).any():
        warnings.append(f"Há valores negativos em {volume_col} (dado inconsistente).")

if warnings:
    for w in warnings:
        st.warning(w)

# Agrega A e B conforme seleção
agg_A = compute_aggregates(
    df=A_df,
    group_cols=group_cols,
    metric_keys=metric_keys,
    extra_numeric_cols=extra_metrics,
    volume_col=volume_col,
    route_col=route_col,
    drop_null_routes=drop_null_routes,
)

agg_B = compute_aggregates(
    df=B_df,
    group_cols=group_cols,
    metric_keys=metric_keys,
    extra_numeric_cols=extra_metrics,
    volume_col=volume_col,
    route_col=route_col,
    drop_null_routes=drop_null_routes,
)

metric_cols_for_delta = []
# quais colunas agregadas vão entrar no delta
for m in metric_keys:
    metric_cols_for_delta.append(m)
for m in extra_metrics:
    metric_cols_for_delta.append(m)

comparison_table = merge_and_delta(
    agg_A=agg_A,
    agg_B=agg_B,
    group_cols=group_cols,
    metric_cols=metric_cols_for_delta,
)

# Ordenação dinâmica
order_metric = st.selectbox(
    "Ordenar por",
    options=[f"{m}_delta" for m in metric_cols_for_delta] + [f"{m}_delta_pct" for m in metric_cols_for_delta],
    index=0,
)

ascending = st.checkbox("Ordem crescente", value=False)
if order_metric in comparison_table.columns:
    comparison_table = comparison_table.sort_values(order_metric, ascending=ascending)

# KPIs totais (sempre mostrados)
def total_summary(dfx: pd.DataFrame) -> Dict[str, float]:
    # sempre calcula volume/rotas/ipr no total
    d = compute_aggregates(
        df=dfx,
        group_cols=[],
        metric_keys=["volume", "rotas", "ipr"],
        extra_numeric_cols=[],
        volume_col=volume_col,
        route_col=route_col,
        drop_null_routes=drop_null_routes,
    )
    row = d.iloc[0].to_dict() if len(d) else {"volume": 0.0, "rotas": 0.0, "ipr": float("nan")}
    return {"volume": float(row.get("volume", 0.0)), "rotas": float(row.get("rotas", 0.0)), "ipr": float(row.get("ipr", float("nan")))}

summary_A = total_summary(A_df)
summary_B = total_summary(B_df)

delta_ipr = summary_A["ipr"] - summary_B["ipr"]
delta_vol = summary_A["volume"] - summary_B["volume"]
delta_rot = summary_A["rotas"] - summary_B["rotas"]

c1, c2, c3 = st.columns(3)
c1.metric("IPR (A)", f"{summary_A['ipr']:.4f}" if summary_A["ipr"] == summary_A["ipr"] else "NaN", delta=f"{delta_ipr:+.4f}")
c2.metric("Volume (A)", f"{summary_A['volume']:.0f}", delta=f"{delta_vol:+.0f}")
c3.metric("Rotas (A)", f"{summary_A['rotas']:.0f}", delta=f"{delta_rot:+.0f}")

st.caption(f"A = **{label_A}** | B = **{label_B}**")

# Exibição da tabela dinâmica
st.markdown("### Tabela de comparação (A vs B + deltas)")
st.dataframe(comparison_table, use_container_width=True)

# =========================
# Chat
# =========================
st.divider()
st.subheader("💬 Chat com o agente (consciente dos períodos e da tabela dinâmica)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_question = st.chat_input("Pergunte algo (ex.: 'por que o IPR caiu em WAREHOUSE_ID X?')")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if st.session_state.agent is None:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=DEFAULT_TEMPERATURE,
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
        comparison_table=comparison_table,
        group_cols=group_cols,
        chosen_metrics=chosen_metrics,
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

st.caption("Nota: o agente tem acesso ao dataframe e pode executar Python para responder. Use com cautela.")
