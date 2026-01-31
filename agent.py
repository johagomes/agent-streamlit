import os
import io
import hashlib
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

st.set_page_config(page_title="Chat com sua base (Pandas Agent)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        # tenta utf-8, se falhar cai pra latin-1
        try:
            return pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    else:
        raise ValueError("Formato não suportado. Envie .csv, .xlsx ou .xls")

def format_history(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    """
    Converte histórico do chat em texto para dar contexto ao agente.
    Mantém poucas interações para não inflar o prompt.
    """
    trimmed = messages[-max_turns*2:]  # user+assistant por turno
    lines = []
    for m in trimmed:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines).strip()

def run_agent(agent, user_question: str, messages: List[Dict[str, str]]) -> str:
    history_txt = format_history(messages)
    if history_txt:
        prompt = (
            "Você é um analista de dados. Responda em português (Brasil).\n"
            "Use o dataframe carregado para calcular e validar resultados.\n\n"
            f"HISTÓRICO RECENTE:\n{history_txt}\n\n"
            f"PERGUNTA ATUAL:\n{user_question}"
        )
    else:
        prompt = (
            "Você é um analista de dados. Responda em português (Brasil).\n"
            "Use o dataframe carregado para calcular e validar resultados.\n\n"
            f"PERGUNTA:\n{user_question}"
        )

    result = agent.invoke(prompt)

    # O retorno pode variar por versão/config; cobrimos os casos comuns:
    if isinstance(result, dict):
        return result.get("output") or result.get("final") or str(result)
    return str(result)

# ----------------------------
# UI - Sidebar
# ----------------------------
st.sidebar.header("Configurações")

api_key = st.sidebar.text_input(
    "OPENAI_API_KEY",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Você pode definir via variável de ambiente OPENAI_API_KEY ou colar aqui."
)

model_name = st.sidebar.text_input(
    "Modelo (ex.: gpt-5-mini ou gpt-4.1-mini)",
    value="gpt-5-mini",
    help="Consulte a lista de modelos na doc da OpenAI."
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

st.sidebar.divider()
uploaded = st.sidebar.file_uploader(
    "Envie sua base (.csv, .xlsx, .xls)",
    type=["csv", "xlsx", "xls"]
)

# ----------------------------
# Estado
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "df_id" not in st.session_state:
    st.session_state.df_id = None

if "agent" not in st.session_state:
    st.session_state.agent = None

# ----------------------------
# Main
# ----------------------------
st.title("💬 Converse com a sua base (Pandas DataFrame Agent)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Base carregada")
    if uploaded is None:
        st.info("Envie um arquivo na barra lateral para começar.")
    else:
        raw = uploaded.getvalue()
        df_id = file_sha256(raw)

        # Recarrega DF e agente se o arquivo mudou
        if st.session_state.df is None or st.session_state.df_id != df_id:
            try:
                df = load_dataframe(uploaded)
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")
                st.stop()

            st.session_state.df = df
            st.session_state.df_id = df_id
            st.session_state.agent = None  # força recriação do agente

        df = st.session_state.df
        st.write(f"Linhas: **{len(df):,}** | Colunas: **{df.shape[1]}**")
        st.dataframe(df.head(50), use_container_width=True)

with col2:
    st.subheader("Chat")

    # Render histórico
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input do usuário
    user_question = st.chat_input("Pergunte algo sobre sua base (ex.: 'qual a soma de vendas por estado?')")

    if user_question:
        if not api_key:
            st.error("Defina sua OPENAI_API_KEY na barra lateral (ou via variável de ambiente).")
            st.stop()

        if st.session_state.df is None:
            st.error("Envie um arquivo primeiro.")
            st.stop()

        # Mostra msg do usuário
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Cria agente se necessário
        if st.session_state.agent is None:
            llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )

            # IMPORTANTE: opt-in para execução de código
            st.session_state.agent = create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                agent_type="tool-calling",
                verbose=False,
                allow_dangerous_code=True,
            )

        # Resposta
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    answer = run_agent(st.session_state.agent, user_question, st.session_state.messages)
                except Exception as e:
                    answer = (
                        "Deu erro ao executar a análise.\n\n"
                        f"**Detalhes:** {e}\n\n"
                        "Dicas: tente uma pergunta mais específica (nome de colunas, filtros) "
                        "ou verifique o modelo configurado."
                    )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Nota: este agente executa Python para responder perguntas sobre o DataFrame. Use com cautela e apenas com dados confiáveis."
)
