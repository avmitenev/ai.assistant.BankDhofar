import os
import streamlit as st
import logging
import time
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

# Override sqlite3 before importing langchain_chroma
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from langchain_chroma import Chroma # Import Chroma from langchain_chroma

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_data_files, load_file
from vector_store import EmbeddingProxy 
from memory import clean_session_history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="BankDhofar's Assistant")

def show_ui(qa, prompt_to_user="How may I help you?"):
    """
    Displays the Streamlit chat UI and handles user interactions.

    Args:
        qa: The LangChain chain for question answering.
        prompt_to_user: The initial prompt to display to the user.
    """
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Say something")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        response = None
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retry_num = 5
                for i in range(retry_num):
                    try:
                        logging.info('Start processing')
                        session_id = get_script_run_ctx().session_id
                        response = ask_question(qa, prompt, session_id)
                        logging.info('End processing')
                        st.markdown(response.content)
                        break
                    except Exception as e:
                        logging.error(f"Error during question answering: {e}")
                        logging.info(f"Attempt {i} failed")
                        if i == (retry_num - 1):
                            st.write("Sorry, there was an error processing your request.")
                        else:
                            time.sleep(2)
                            logging.info("Retrying")
        
        message = {"role": "assistant", "content": response.content if response else "Error"}
        st.session_state.messages.append(message)

@st.cache_resource
def get_retriever(openai_api_key=None):
    """
    Creates and caches the document retriever.

    Args:
        openai_api_key: The OpenAI API key.

    Returns:
        An ensemble document retriever.
    """
    try:
        docs = load_data_files(data_dir="data")  
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        embeddings = HuggingFaceEmbeddings()
        return ensemble_retriever_from_docs(docs, embeddings=embeddings)
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        logging.exception(f"message")
        st.error("Error initializing the application. Please check the logs.")
        st.stop()  # Stop execution if retriever creation fails


def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    """
    Creates the question answering chain.

    Args:
        openai_api_key: The OpenAI API key.
        huggingfacehub_api_token: The Hugging Face Hub API token.

    Returns:
        A LangChain question answering chain.
    """
    try:
        logging.info('Start creating chain')
        ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
        chain = create_full_chain(
            ensemble_retriever,
            hf_api_key=huggingfacehub_api_token,
            openai_api_key=openai_api_key,
        )
        logging.info('Chain creating complete')
        return ensemble_retriever, chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        logging.exception(f"message")
        st.error("Error initializing the application. Please check the logs.")
        st.stop()  # Stop execution if chain creation fails

def get_secret_or_input(secret_key, secret_name, info_link=None):
    """
    Retrieves a secret from Streamlit secrets or prompts the user for input.

    Args:
        secret_key: The key of the secret in Streamlit secrets.
        secret_name: The user-friendly name of the secret.
        info_link: An optional link to provide information about the secret.

    Returns:
        The secret value.
    """
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def reset(prompt_to_user="How may I help you?"):
    session_id = get_script_run_ctx().session_id
    clean_session_history(session_id)
    st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

def run():
    """
    Main function to run the Streamlit application.
    """
    ready = True
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    openai_api_key = os.getenv('OPENAI_API_KEY')

    if ready:
        # try:
        logging.info('run loop')

        if not st.session_state.get('init', False):
            st.session_state['ensemble_retriever'], st.session_state['chain'] = get_chain(
                openai_api_key=openai_api_key,
                huggingfacehub_api_token=huggingfacehub_api_token
            )
            st.session_state['init'] = True

        # Chat Interface
        st.title("BankDhofar's Assistant")
        st.caption("The artificial intelligence application was trained on a selection of publicly available documents from the organization's website over a period of three days. Notably, the accuracy of its responses can be further enhanced with access to additional training materials.")
        st.subheader("Enquire about our range of BankDhofar products and services:")
        show_ui(st.session_state['chain'], "How can I assist you today?")
        st.button("Reset history", on_click=reset)

        # except Exception as e:
        #     logging.error(f"Error initializing application: {e}")
        #     st.error("Error initializing the application. Please check the logs.")
    else:
        st.stop()

run()
