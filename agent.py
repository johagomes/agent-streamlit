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

    # ✅ As 6 dimensões (as mesmas que você citou)
    "dims6": [
        "CLUSTER_ID",
        "HUB",
        "TIPO_ROTA",
        "TRANSPORTADORA",
        "MODAL",
        "DIA_DA_SEMANA",
    ],
}

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0  # fixo (não aparece na UI)


# =========================
# App Setup
# =========================
load_dotenv()
st.set_page_config(page_title="IPR Agent — 36 Tabelas Fixas", layout="wide")
st.title("💬 IPR Agent — Pacotes por rota (IPR = Volume / Rotas)")
st.caption("IPR (oficial) = sum(PACOTES_REAL) / COUNTD(ROUTE_ID_REAL).")


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
    Se WEEK não existir ou estiver vazio, cria como YYYY/WW (ISO) a partir de date_col.
    Se existir (ex.: SEMANA), mantém.
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
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e < s:
        s, e = e, s
    mask = (df[date_col].dt.normalize() >= s) & (df[date_col].dt.normalize() <= e)
    return df[mask].copy()


def infer_extra_numeric_metrics(
    df: pd.DataFrame,
    forbidden_cols: List[str],
) -> List[str]:
    """
    Detecta colunas numéricas somáveis, excluindo colunas "proibidas" e possíveis IDs.
    Heurística: exclui colunas com 'ID' no nome (ex.: ROUTE_ID_PLAN), e as forbidden.
    """
    extras: List[str] = []
    for c in df.columns:
        if c in forbidden_cols:
            continue
        uc = str(c).upper()

        # Evita IDs e chaves
        if "ID" in uc:
            continue

        # tenta converter p/ numérico
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            extras.append(c)

    # ordena de forma estável
    return sorted(extras)


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
    Oficiais:
      - volume: sum(volume_col)
      - rotas:  nunique(route_col)
      - ipr:    volume/rotas
    Extras: sum(coluna)
    """
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

    # Oficiais
    if "volume" in metric_keys:
        out["volume"] = gb[volume_col].apply(safe_sum).values if volume_col in dfx.columns else 0.0

    if "rotas" in metric_keys or "ipr" in metric_keys:
        out["rotas"] = gb[route_col].nunique(dropna=True).values if route_col in dfx.columns else 0

    if "ipr" in metric_keys:
        if "volume" not in out.columns:
            out["volume"] = gb[volume_col].apply(safe_sum).values if volume_col in dfx.columns else 0.0
        out["ipr"] = out.apply(lambda r: (r["volume"] / r["rotas"]) if r["rotas"] else float("nan"), axis=1)

    # Extras (numéricas)
    for col in extra_numeric_cols:
        out[col] = gb[col].apply(safe_sum).values if col in dfx.columns else 0.0

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
    Junta A e B e cria *_A, *_B, *_delta, *_delta_pct
    """
    if not group_cols:
        agg_A["_k"] = 1
        agg_B["_k"] = 1
        group_cols = ["_k"]

    m = pd.merge(agg_A, agg_B, on=group_cols, how="outer", suffixes=("_A", "_B"))

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


def totals_ipr(df: pd.DataFrame, volume_col: str, route_col: str, drop_null_routes: bool) -> Dict[str, float]:
    dfx = df.copy()
    if drop_null_routes and route_col in dfx.columns:
        dfx = dfx[dfx[route_col].notna()].copy()

    vol = safe_sum(dfx[volume_col]) if volume_col in dfx.columns else 0.0
    rotas = float(dfx[route_col].nunique(dropna=True)) if route_col in dfx.columns else 0.0
    ipr = (vol / rotas) if rotas else float("nan")
    return {"volume": float(vol), "rotas": float(rotas), "ipr": float(ipr)}


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
    totals_A: Dict[str, float],
    totals_B: Dict[str, float],
    tables_preview: Dict[str, pd.DataFrame],
    messages: List[Dict[str, str]],
) -> str:
    """
    Para o chat: não mandamos as 36 tabelas completas (muito grande),
    mas sim um PREVIEW top-3 de cada uma, ordenado por |ipr_delta| quando existir.
    O agente ainda tem acesso ao df para recalcular qualquer detalhe.
    """
    previews_blocks = []
    for name, t in tables_preview.items():
        if t is None or t.empty:
            previews_blocks.append(f"\n# {name}\n(sem linhas no recorte A/B)\n")
            continue
        previews_blocks.append(f"\n# {name}\n{t.to_csv(index=False)}")

    previews_txt = "\n".join(previews_blocks).strip()

    def fnum(x: float) -> str:
        if x != x:
            return "NaN"
        return f"{x:,.4f}"

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
ΔIPR = {fnum(totals_A["ipr"] - totals_B["ipr"])}

IMPORTANTE:
- Você TEM acesso ao dataframe df (granular) e pode recalcular qualquer coisa.
- Além disso, abaixo há PREVIEWS das 36 tabelas fixas (top-3 linhas de cada), contendo A/B e deltas.

Formato obrigatório:
(1) Resumo executivo
(2) Comparação e filtros
(3) Componentes do IPR
(4) Deltas e ranking por dimensão
(5) Diagnóstico
(6) Próximos passos
""".strip()

    hist = format_history(messages)
    prompt_parts = [
        base,
        "PREVIEWS DAS 36 TABELAS FIXAS (top-3 por |ipr_delta| quando houver):",
        previews_txt,
        ("HISTÓRICO RECENTE:\n" + hist) if hist else "",
        "PERGUNTA DO USUÁRIO:\n" + user_question,
    ]
    return "\n\n".join([p for p in prompt_parts if p]).strip()


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
if "tables36" not in st.session_state:
    st.session_state.tables36 = {}


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
    st.session_state.tables36 = {}  # invalida tabelas pré-calculadas quando muda o arquivo

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

# =========================
# Sidebar — Datas de comparação (ranges)
# =========================
st.sidebar.subheader("Datas de comparação (ranges)")

# defaults: A últimos 7 dias, B 7 dias anteriores
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

volume_col = METRICS_POLICY["volume_col"]
route_col = METRICS_POLICY["route_col"]
drop_null_routes = METRICS_POLICY["guardrails"]["drop_null_routes"]

# =========================
# Validações simples
# =========================
warnings = []
if volume_col not in df.columns:
    warnings.append(f"Falta coluna de volume base: {volume_col}")
if route_col not in df.columns:
    warnings.append(f"Falta coluna de rota base: {route_col}")

if volume_col in df.columns and METRICS_POLICY["guardrails"]["warn_if_negative_volume"]:
    if pd.to_numeric(df[volume_col], errors="coerce").fillna(0).lt(0).any():
        warnings.append(f"Há valores negativos em {volume_col} (dado inconsistente).")

for w in warnings:
    st.warning(w)

# =========================
# Métricas "fixas": oficiais + extras numéricas detectadas
# =========================
official_metric_keys = ["ipr", "volume", "rotas"]

forbidden_for_extras = (
    METRICS_POLICY["dims6"]
    + [METRICS_POLICY["date_col"], METRICS_POLICY["week_col"], METRICS_POLICY["route_col"], METRICS_POLICY["volume_col"]]
)
extra_metrics = infer_extra_numeric_metrics(df, forbidden_cols=forbidden_for_extras)

# As tabelas terão: oficiais + extras
metric_cols_for_delta = official_metric_keys + extra_metrics

# =========================
# 36 tabelas fixas (6x6 pares ordenados)
# - se d1 == d2: agrupa só por [d1]
# - senão: agrupa por [d1, d2]
# =========================
dims6_present = [d for d in METRICS_POLICY["dims6"] if d in df.columns]
if len(dims6_present) < 2:
    st.error("Não encontrei as 6 dimensões esperadas na base. Verifique os nomes das colunas.")
    st.stop()

pairs_36: List[Tuple[str, str]] = [(d1, d2) for d1 in dims6_present for d2 in dims6_present]

# Recalcula tabelas somente quando (arquivo ou período) muda
tables_cache_key = (
    st.session_state.df_id,
    str(A_start),
    str(A_end),
    str(B_start),
    str(B_end),
    "|".join(dims6_present),
    "|".join(metric_cols_for_delta),
)

if st.session_state.tables36.get("_cache_key") != tables_cache_key:
    tables_full: Dict[str, pd.DataFrame] = {}

    for d1, d2 in pairs_36:
        group_cols = [d1] if d1 == d2 else [d1, d2]
        name = f"{d1} × {d2}"

        agg_A = compute_aggregates(
            df=A_df,
            group_cols=group_cols,
            metric_keys=official_metric_keys,
            extra_numeric_cols=extra_metrics,
            volume_col=volume_col,
            route_col=route_col,
            drop_null_routes=drop_null_routes,
        )
        agg_B = compute_aggregates(
            df=B_df,
            group_cols=group_cols,
            metric_keys=official_metric_keys,
            extra_numeric_cols=extra_metrics,
            volume_col=volume_col,
            route_col=route_col,
            drop_null_routes=drop_null_routes,
        )
        merged = merge_and_delta(
            agg_A=agg_A,
            agg_B=agg_B,
            group_cols=group_cols,
            metric_cols=metric_cols_for_delta,
        )

        # ordena por |ipr_delta| se existir, senão por |volume_delta|
        if "ipr_delta" in merged.columns:
            merged["_abs_rank"] = merged["ipr_delta"].abs()
        elif "volume_delta" in merged.columns:
            merged["_abs_rank"] = merged["volume_delta"].abs()
        else:
            merged["_abs_rank"] = 0.0

        merged = merged.sort_values("_abs_rank", ascending=False).drop(columns=["_abs_rank"], errors="ignore")
        tables_full[name] = merged

    st.session_state.tables36 = {"_cache_key": tables_cache_key, "tables": tables_full}
else:
    tables_full = st.session_state.tables36["tables"]

# =========================
# KPIs topo (totais)
# =========================
totals_A = totals_ipr(A_df, volume_col, route_col, drop_null_routes)
totals_B = totals_ipr(B_df, volume_col, route_col, drop_null_routes)

delta_ipr = totals_A["ipr"] - totals_B["ipr"]
delta_vol = totals_A["volume"] - totals_B["volume"]
delta_rot = totals_A["rotas"] - totals_B["rotas"]

c1, c2, c3 = st.columns(3)
c1.metric("IPR (A)", f"{totals_A['ipr']:.4f}" if totals_A["ipr"] == totals_A["ipr"] else "NaN", delta=f"{delta_ipr:+.4f}")
c2.metric("Volume (A)", f"{totals_A['volume']:.0f}", delta=f"{delta_vol:+.0f}")
c3.metric("Rotas (A)", f"{totals_A['rotas']:.0f}", delta=f"{delta_rot:+.0f}")
st.caption(f"A = **{label_A}** | B = **{label_B}**")

# =========================
# ✅ Substitui as sessões antigas por 36 tabelas fixas
# =========================
st.subheader("📌 Tabelas fixas (36 combinações das 6 dimensões)")

st.caption(
    "Cada tabela contém TODAS as métricas: ipr/volume/rotas + todas as métricas numéricas detectadas (exceto IDs). "
    "Ordem padrão: maior |ipr_delta| (ou |volume_delta| se ipr não existir)."
)

# Renderização: 36 expanders (prontos)
# Obs: Streamlit aguenta, mas para bases grandes pode ficar pesado.
# Se quiser performance, depois a gente troca pra 'tabs' ou 'selectbox' mantendo as tabelas fixas.
for name in sorted(tables_full.keys()):
    with st.expander(f"🧾 {name}", expanded=False):
        t = tables_full[name]
        st.dataframe(t, use_container_width=True)

        # download
        csv = t.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar CSV desta tabela",
            data=csv,
            file_name=f"tabela_{name.replace(' × ', '_x_')}.csv",
            mime="text/csv",
        )

# =========================
# Previews (top-3) para o chat ler sem estourar tokens
# =========================
tables_preview: Dict[str, pd.DataFrame] = {}
for name, t in tables_full.items():
    if t is None or t.empty:
        tables_preview[name] = t
        continue

    # Colunas principais para preview (mantém compacto)
    preview_cols = []
    # dims (as colunas do grupo estão presentes)
    dims_in = [c for c in t.columns if c in name.replace(" × ", " ").split(" ")]  # simples
    # fallback: pega as primeiras colunas não-métricas (até 2)
    if not dims_in:
        dims_in = [c for c in t.columns if not any(c.endswith(sfx) for sfx in ["_A", "_B", "_delta", "_delta_pct"])]
        dims_in = dims_in[:2]

    core = [
        "ipr_A", "ipr_B", "ipr_delta",
        "volume_A", "volume_B", "volume_delta",
        "rotas_A", "rotas_B", "rotas_delta",
    ]
    preview_cols = dims_in + [c for c in core if c in t.columns]

    # top-3
    tables_preview[name] = t[preview_cols].head(3).copy()

# =========================
# Chat
# =========================
st.divider()
st.subheader("💬 Chat com o agente (lê as 36 tabelas fixas via previews + pode recalcular no df)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_question = st.chat_input("Pergunte algo (ex.: 'qual HUB e CLUSTER_ID mais derrubou o IPR?')")

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

        # O agente tem acesso ao df (granular) para recalcular qualquer coisa.
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
        totals_A=totals_A,
        totals_B=totals_B,
        tables_preview=tables_preview,
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
                    "- valide se as colunas PACOTES_REAL e ROUTE_ID_REAL existem\n"
                    "- tente uma pergunta mais específica (cite a tabela desejada, ex.: 'HUB × CLUSTER_ID')\n"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Nota: para não estourar tokens, o chat recebe apenas TOP-3 de cada tabela fixa. "
    "Se precisar de detalhes completos, o agente pode recalcular no df ou você pode citar a tabela e pedir 'detalhar top N'."
)
