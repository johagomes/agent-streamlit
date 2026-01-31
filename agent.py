import os
import io
import hashlib
from typing import List, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Carrega .env local (útil fora do Streamlit Cloud)
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
        # tenta utf-8; se falhar, cai pra latin-1
        try:
            return pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Formato não suportado. Envie .csv, .xlsx ou .xls")


def format_history(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    """
    Converte histórico do chat em texto para dar contexto ao agente.
    Mantém poucas interações para não inflar o prompt.
    """
    trimmed = messages[-max_turns * 2 :]  # user+assistant por turno
    lines = []
    for m in trimmed:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines).strip()


def run_agent(agent, user_question: str, messages: List[Dict[str, str]]) -> str:
    history_txt = format_history(messages)

    base_instructions = (
        "Você é um analista de dados. Responda em português (Brasil).\n"
        "Use o dataframe carregado para calcular e validar resultados.\n"
        "Quando fizer contas, explique o raciocínio e mostre números-chave.\n"
        "Se a pergunta for ambígua, faça uma suposição razoável e diga qual foi.\n"
    )

    if history_txt:
        prompt = (
            f"{base_instructions}\n"
            f"HISTÓRICO RECENTE:\n{history_txt}\n\n"
            f"PERGUNTA ATUAL:\n{user_question}"
        )
    else:
        prompt = f"{base_instructions}\nPERGUNTA:\n{user_question}"

    result = agent.invoke(prompt)

    # O retorno pode variar por versão/config; cobrimos os casos comuns:
    if isinstance(result, dict):
        return result.get("output") or result.get("final") or str(result)
    return str(result)


# ----------------------------
# Sidebar - Configurações
# ----------------------------
st.sidebar.header("Configurações")

# Lê do Streamlit Secrets primeiro, depois env, depois vazio
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
    help="Defina em Secrets (recomendado) ou via variável de ambiente OPENAI_API_KEY.",
)

default_model = "gpt-3.5-turbo-0613"
try:
    default_model = st.secrets.get("OPENAI_MODEL", default_model)
except Exception:
    pass

model_name = st.sidebar.text_input(
    "Modelo",
    value=default_model,
    help='Ex.: "gpt-3.5-turbo-0613"',
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "Envie sua base (.csv, .xlsx, .xls)",
    type=["csv", "xlsx", "xls"],
)

if st.sidebar.button("Limpar conversa"):
    st.session_state.pop("messages", None)


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
            st.session_state.agent = None  # força recriação do agente (DF novo)

        df = st.session_state.df
        st.write(f"Linhas: **{len(df):,}** | Colunas: **{df.shape[1]}**")

        with st.expander("Ver colunas"):
            st.write(list(df.columns))

        st.dataframe(df.head(50), use_container_width=True)

        st.caption(
            "Dica: pergunte coisas como 'sumarize as colunas', 'total por categoria', "
            "'top 10 por valor', 'filtre por data', etc."
        )

with col2:
    st.subheader("Chat")

    # Render histórico
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input do usuário
    user_question = st.chat_input(
        "Pergunte algo sobre sua base (ex.: 'qual a soma de vendas por estado?')"
    )

    if user_question:
        if not api_key:
            st.error("Defina sua OPENAI_API_KEY (Secrets ou sidebar).")
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

            # IMPORTANTE:
            # - allow_dangerous_code=True permite execução de Python pelo agente (risco).
            # - use apenas com arquivos confiáveis e, idealmente, em ambiente isolado.
            st.session_state.agent = create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                agent_type="openai-functions",  # bom para gpt-3.5-turbo-0613
                verbose=False,
                allow_dangerous_code=True,
            )

        # Resposta
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    answer = run_agent(
                        st.session_state.agent,
                        user_question,
                        st.session_state.messages,
                    )
                except Exception as e:
                    answer = (
                        "Deu erro ao executar a análise.\n\n"
                        f"**Detalhes:** {e}\n\n"
                        "Tente:\n"
                        "- checar se o modelo existe/está liberado na sua conta\n"
                        "- fazer uma pergunta mais específica (com nomes de colunas)\n"
                        "- validar se o arquivo carregou corretamente\n"
                    )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(
    "Nota: este agente pode executar Python para responder perguntas sobre o DataFrame. "
    "Use com cautela e apenas com dados confiáveis."
)
