import os
import logging

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from basic_chain import get_model
from ensemble import ensemble_retriever_from_docs
from local_loader import load_data_files
from memory import create_memory_chain
from rag_chain import make_rag_chain
from langchain_core.runnables import ConfigurableField

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_full_chain(retriever, repo_id="ChatGPT", hf_api_key=None, openai_api_key=None):
    # try:
    model = get_model(repo_id, hf_api_key=hf_api_key, openai_api_key=openai_api_key)
    

    system_prompt = """You are a helpful and knowledgeable financial consultant. 
Use the provided context from BankDhofar's products and services to answer the user's questions. 
If you cannot find an answer in the context, inform the user that you need more information or that the question is outside your expertise. 

Context: {context}

Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = create_memory_chain(model, rag_chain, contextualize_q_prompt)
    return chain
    # except Exception as e:
    #     logging.error(f"Error creating full chain: {e}")
    #     # Handle the error:
    #     # - You could return a simpler chain or a default response
    #     # - Raise an exception to stop execution


def ask_question(chain, query, session_id):
    # try:
    logging.info(f"Send request: {query}")
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": session_id}},
    )
    return response
    # except Exception as e:
    #     logging.error(f"Error asking question: {e}")
    #     # Handle the error, e.g., return an error message
    #     return "Sorry, there was an error processing your request."


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    try:
        docs = load_data_files()
        ensemble_retriever = ensemble_retriever_from_docs(docs)
        chain = create_full_chain(ensemble_retriever)

        queries = [ 
            "What are the benefits of opening an BankDhofar Ordinary Account?",
            "What are the interest rates for a home loan at BankDhofar Bank?",
            "Can you compare the BankDhofar Gold Credit Card to the Classic Credit Card?",
            "How much does it cost to send money to an M-Pesa account using BankDhofar Mobile Banking?",
        ]

        for query in queries:
            response = ask_question(chain, query)
            console.print(Markdown(response.content))

    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
