import os
import io
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

COMPARISON_LABELS = {
    "day_vs_day-1": "Dia vs dia anterior (D vs D-1)",
    "day_vs_d-7": "Dia vs D-7 (D vs D-7)",
    "week_vs_week-1": "Semana vs semana-1 (W vs W-1)",
}

# =========================
# App Setup
# =========================
load_dotenv()
st.set_page_config(page_title="IPR Agent (Pacotes por rota)", layout="wide")
st.title("💬 IPR Agent — Pacotes por rota (IPR = Volume / Rotas)")

st.caption(
    "IPR (oficial) = sum(PACOTES_COLETADOS_TOTAL) / COUNTD(FIRST_ROUTE_ID). "
    "Você pode alternar o modo de comparação no menu lateral."
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
        # já existe e tem conteúdo
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
    Retorna componentes do IPR:
    - volume = sum(volume_col)
    - rotas  = nunique(route_col)
    - ipr    = volume / rotas
    """
    dfx = df
    if drop_null_routes and route_col in dfx.columns:
        dfx = dfx[dfx[route_col].notna()].copy()

    volume = safe_sum(dfx[volume_col]) if volume_col in dfx.columns else 0.0
    rotas = safe_nunique(dfx[route_col]) if route_col in dfx.columns else 0

    ipr = (volume / rotas) if rotas > 0 else float("nan")
    return {"volume": volume, "rotas": float(rotas), "ipr": float(ipr)}


def iso_week_of_date(d: pd.Timestamp) -> Tuple[int, int]:
    iso = pd.Timestamp(d).isocalendar()
    return int(iso.year), int(iso.week)


def prev_iso_week(year: int, week: int) -> Tuple[int, int]:
    """
    Retorna (ano, semana) anterior ISO, lidando com virada de ano.
    """
    # pega segunda-feira daquela ISO-week e subtrai 7 dias
    # (ISO week starts Monday)
    monday = pd.Timestamp.fromisocalendar(year, week, 1)
    prev = monday - pd.Timedelta(days=7)
    return iso_week_of_date(prev)


def make_period_slices(
    df: pd.DataFrame,
    comparison_mode: str,
    ref_date: pd.Timestamp,
    date_col: str,
    week_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Cria recortes A e B e também retorna labels legíveis.
    """
    if comparison_mode in ("day_vs_day-1", "day_vs_d-7"):
        dA = pd.Timestamp(ref_date).normalize()
        delta = 1 if comparison_mode == "day_vs_day-1" else 7
        dB = (pd.Timestamp(ref_date) - pd.Timedelta(days=delta)).normalize()

        A = df[df[date_col].dt.normalize() == dA].copy()
        B = df[df[date_col].dt.normalize() == dB].copy()

        label_A = dA.strftime("%Y-%m-%d")
        label_B = dB.strftime("%Y-%m-%d")
        return A, B, label_A, label_B

    if comparison_mode == "week_vs_week-1":
        yA, wA = iso_week_of_date(pd.Timestamp(ref_date))
        yB, wB = prev_iso_week(yA, wA)

        wkA = f"{yA}/{str(wA).zfill(2)}"
        wkB = f"{yB}/{str(wB).zfill(2)}"

        A = df[df[week_col] == wkA].copy()
        B = df[df[week_col] == wkB].copy()
        return A, B, wkA, wkB

    raise ValueError("comparison_mode inválido")


def driver_decomposition_mix_perf(
    A: pd.DataFrame,
    B: pd.DataFrame,
    dim: str,
    volume_col: str,
    route_col: str,
    drop_null_routes: bool = True,
) -> pd.DataFrame:
    """
    Decomposição ΔIPR ≈ Δmix + Δperf por dimensão dim:
      r_g = IPR_g = volume_g / rotas_g
      w_g = share_rotas_g

      mix_contrib  = (wA - wB) * rB
      perf_contrib = wA * (rA - rB)
      total_contrib = mix + perf

    Retorna um dataframe com stats e contribuições por grupo.
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

    a = prep(A).rename(columns={
        "volume": "volume_A", "rotas": "rotas_A", "ipr": "ipr_A", "share_rotas": "wA"
    })
    b = prep(B).rename(columns={
        "volume": "volume_B", "rotas": "rotas_B", "ipr": "ipr_B", "share_rotas": "wB"
    })

    if a.empty and b.empty:
        return pd.DataFrame()

    merged = pd.merge(a, b, on=dim, how="outer")
    for c in ["volume_A","rotas_A","ipr_A","wA","volume_B","rotas_B","ipr_B","wB"]:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # contribuições
    merged["mix_contrib"] = (merged["wA"] - merged["wB"]) * merged["ipr_B"]
    merged["perf_contrib"] = merged["wA"] * (merged["ipr_A"] - merged["ipr_B"])
    merged["total_contrib"] = merged["mix_contrib"] + merged["perf_contrib"]

    # ordena por impacto
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
    comparison_mode: str,
    label_A: str,
    label_B: str,
    summary_A: Dict[str, float],
    summary_B: Dict[str, float],
    drivers: Dict[str, pd.DataFrame],
    messages: List[Dict[str, str]],
) -> str:
    """
    Cria um prompt fixo que "ensina" o agente:
    - métrica oficial IPR
    - modo de comparação
    - template de resposta
    - e injeta números pré-calculados + top drivers
    """
    # Monta snippets compactos de drivers (top 5 por dimensão)
    driver_snippets = []
    for dim, df_dim in drivers.items():
        if df_dim is None or df_dim.empty:
            continue
        top = df_dim.head(5).copy()
        # seleciona colunas essenciais
        keep_cols = [dim, "total_contrib", "mix_contrib", "perf_contrib", "ipr_A", "ipr_B", "wA", "wB", "rotas_A", "rotas_B", "volume_A", "volume_B"]
        keep_cols = [c for c in keep_cols if c in top.columns]
        top = top[keep_cols]
        driver_snippets.append(f"\nDIMENSÃO: {dim}\n{top.to_string(index=False)}\n")

    driver_block = "\n".join(driver_snippets).strip() if driver_snippets else "Sem drivers calculados (dimensões ausentes ou recorte vazio)."

    history_txt = format_history(messages)

    # Política / SOP fixo
    sop = f"""
Você é um analista de dados especializado nesta base (granularidade por pallet/INBOUND_ID).
A métrica OFICIAL é:
- Volume = sum({METRICS_POLICY["volume_col"]})
- Rotas  = COUNTD({METRICS_POLICY["route_col"]}) = nunique({METRICS_POLICY["route_col"]})
- IPR    = Volume / Rotas  (Pacotes por rota)

MODO DE COMPARAÇÃO (já escolhido pelo usuário):
- {COMPARISON_LABELS[comparison_mode]}
Períodos:
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

    # Resumos
    def fnum(x: float) -> str:
        if x != x:  # NaN
            return "NaN"
        return f"{x:,.4f}"

    summary_block = f"""
NÚMEROS PRÉ-CALCULADOS:
A: volume={summary_A["volume"]:.0f}, rotas={summary_A["rotas"]:.0f}, ipr={fnum(summary_A["ipr"])}
B: volume={summary_B["volume"]:.0f}, rotas={summary_B["rotas"]:.0f}, ipr={fnum(summary_B["ipr"])}
ΔIPR = {fnum(summary_A["ipr"] - summary_B["ipr"])}
""".strip()

    prompt_parts = [sop, summary_block, "DRIVERS (Top 5 por dimensão):", driver_block]

    if history_txt:
        prompt_parts.append("\nHISTÓRICO RECENTE:\n" + history_txt)

    prompt_parts.append("\nPERGUNTA DO USUÁRIO:\n" + user_question)

    return "\n\n".join(prompt_parts).strip()


# =========================
# Sidebar — OpenAI + Upload
# =========================
st.sidebar.header("Configurações")

# API Key via Secrets (igual você mostrou) + fallback env + sidebar
default_api_key = ""
try:
    default_api_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    default_api_key = ""

if not default_api_key:
    default_api_key = os.getenv("OPENAI_API_KEY", "")

api_key = st.sidebar.text_input(
    "OPENAI_API_KEY",
    value=default_api_key,
    type="password",
    help="Recomendado: definir em Secrets (TOML): OPENAI_API_KEY = \"...\"",
)

default_model = "gpt-3.5-turbo-0613"
try:
    default_model = st.secrets.get("OPENAI_MODEL", default_model)
except Exception:
    pass

model_name = st.sidebar.text_input("Modelo", value=default_model)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

st.sidebar.divider()
uploaded = st.sidebar.file_uploader("Envie sua base (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

st.sidebar.divider()
comparison_mode = st.sidebar.selectbox(
    "Modo de comparação",
    options=list(COMPARISON_LABELS.keys()),
    format_func=lambda k: COMPARISON_LABELS[k],
    index=1,  # default: D vs D-7 (geralmente bom pra sazonalidade)
)

st.sidebar.caption("Dica: D vs D-7 controla sazonalidade de dia da semana.")

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
left, right = st.columns([1, 1])

with left:
    st.subheader("📦 Base carregada")

    if uploaded is None:
        st.info("Envie o arquivo na barra lateral para começar.")
        st.stop()

    raw = uploaded.getvalue()
    df_id = file_sha256(raw)

    if st.session_state.df is None or st.session_state.df_id != df_id:
        df = read_uploaded_dataframe(uploaded)
        st.session_state.df = df
        st.session_state.df_id = df_id
        st.session_state.agent = None  # força recriar agente quando DF muda

    df = st.session_state.df

    # Normalizações
    df = ensure_datetime(df, METRICS_POLICY["date_col"])
    df = ensure_week_iso(df, METRICS_POLICY["date_col"], METRICS_POLICY["week_col"])
    st.session_state.df = df  # salva normalizado

    # Informações rápidas
    st.write(f"Linhas: **{len(df):,}** | Colunas: **{df.shape[1]}**")

    with st.expander("Ver colunas"):
        st.write(list(df.columns))

    st.dataframe(df.head(50), use_container_width=True)

    # Controle de data de referência
    date_col = METRICS_POLICY["date_col"]
    if date_col in df.columns and df[date_col].notna().any():
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        ref_date = st.date_input("Data de referência (para comparação)", value=max_date, min_value=min_date, max_value=max_date)
        ref_date_ts = pd.Timestamp(ref_date)
    else:
        st.warning(f"Coluna de data '{date_col}' não encontrada ou inválida. Não dá pra montar comparação por tempo.")
        st.stop()

with right:
    st.subheader("📊 Comparação A vs B (prévia)")
    df = st.session_state.df

    # Monta recortes A e B
    A, B, label_A, label_B = make_period_slices(
        df=df,
        comparison_mode=comparison_mode,
        ref_date=pd.Timestamp(ref_date_ts),
        date_col=METRICS_POLICY["date_col"],
        week_col=METRICS_POLICY["week_col"],
    )

    # Sanity checks
    volume_col = METRICS_POLICY["volume_col"]
    route_col = METRICS_POLICY["route_col"]

    warnings = []
    if volume_col not in df.columns:
        warnings.append(f"Falta coluna de volume: {volume_col}")
    if route_col not in df.columns:
        warnings.append(f"Falta coluna de rotas: {route_col}")

    if volume_col in df.columns:
        if pd.to_numeric(df[volume_col], errors="coerce").fillna(0).lt(0).any() and METRICS_POLICY["guardrails"]["warn_if_negative_volume"]:
            warnings.append(f"Há valores negativos em {volume_col} (isso costuma indicar dado inconsistente).")

    if route_col in df.columns:
        null_rate_A = float(A[route_col].isna().mean()) if len(A) else 0.0
        null_rate_B = float(B[route_col].isna().mean()) if len(B) else 0.0
        if null_rate_A > 0.05 or null_rate_B > 0.05:
            warnings.append(
                f"Alta taxa de {route_col} nulo: A={null_rate_A:.1%} | B={null_rate_B:.1%} "
                "(isso pode distorcer o denominador de rotas)."
            )

    if warnings:
        for w in warnings:
            st.warning(w)

    # Calcula IPR A e B
    summary_A = calc_ipr(A, volume_col, route_col, drop_null_routes=METRICS_POLICY["guardrails"]["drop_null_routes"])
    summary_B = calc_ipr(B, volume_col, route_col, drop_null_routes=METRICS_POLICY["guardrails"]["drop_null_routes"])

    delta_ipr = summary_A["ipr"] - summary_B["ipr"]
    delta_vol = summary_A["volume"] - summary_B["volume"]
    delta_rot = summary_A["rotas"] - summary_B["rotas"]

    # Exibe KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("IPR (A)", f"{summary_A['ipr']:.4f}" if summary_A["ipr"] == summary_A["ipr"] else "NaN", delta=f"{delta_ipr:+.4f}")
    c2.metric("Volume (A)", f"{summary_A['volume']:.0f}", delta=f"{delta_vol:+.0f}")
    c3.metric("Rotas (A)", f"{summary_A['rotas']:.0f}", delta=f"{delta_rot:+.0f}")

    st.caption(f"A = **{label_A}** | B = **{label_B}** | Modo: **{COMPARISON_LABELS[comparison_mode]}**")

    # Calcula drivers padrão (Top 5) para ajudar a explicar ΔIPR
    drivers: Dict[str, pd.DataFrame] = {}
    with st.expander("Drivers (Top 5) — decomposição Mix vs Performance"):
        for dim in METRICS_POLICY["default_driver_dimensions"]:
            if dim not in df.columns:
                st.info(f"Dimensão '{dim}' não existe na base — pulando.")
                continue
            ddf = driver_decomposition_mix_perf(
                A=A, B=B, dim=dim,
                volume_col=volume_col, route_col=route_col,
                drop_null_routes=METRICS_POLICY["guardrails"]["drop_null_routes"],
            )
            drivers[dim] = ddf
            st.markdown(f"**{dim}**")
            show = ddf.head(10).copy()
            st.dataframe(show, use_container_width=True)

# =========================
# Chat (Agent)
# =========================
st.divider()
st.subheader("💬 Chat com o agente (consciente do IPR e do modo de comparação)")

# Render histórico
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_question = st.chat_input("Pergunte algo (ex.: 'por que o IPR caiu?')")

if user_question:
    if not api_key:
        st.error("Defina a OPENAI_API_KEY (Secrets ou sidebar).")
        st.stop()

    # registra msg do usuário
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # cria agente se necessário
    if st.session_state.agent is None:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

        # IMPORTANTE: allow_dangerous_code=True permite execução de Python pelo agente (risco).
        # Use apenas com arquivos confiáveis e, idealmente, em ambiente isolado.
        st.session_state.agent = create_pandas_dataframe_agent(
            llm,
            st.session_state.df,
            agent_type="openai-functions",  # compatível com gpt-3.5-turbo-0613
            verbose=False,
            allow_dangerous_code=True,
        )

    # prompt estruturado com SOP + métricas + drivers
    prompt = build_agent_prompt(
        user_question=user_question,
        comparison_mode=comparison_mode,
        label_A=label_A,
        label_B=label_B,
        summary_A=summary_A,
        summary_B=summary_B,
        drivers=drivers,
        messages=st.session_state.messages,
    )

    # resposta
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
                    "- confira se o modelo está liberado na sua conta\n"
                    "- valide se as colunas PACOTES_COLETADOS_TOTAL e FIRST_ROUTE_ID existem\n"
                    "- tente uma pergunta mais específica\n"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Nota: o agente tem acesso ao dataframe e pode executar Python para responder. "
    "Use com cautela e apenas com dados confiáveis."
)
