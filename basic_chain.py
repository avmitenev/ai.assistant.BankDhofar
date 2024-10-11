import os
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
ZEPHYR_ID = "HuggingFaceH4/zephyr-7b-beta"



def get_model(repo_id="ChatGPT", **kwargs):
    """
    Loads and configures the specified language model.

    Args:
        repo_id: The model identifier ("ChatGPT", MISTRAL_ID, or ZEPHYR_ID).
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        A configured ChatOpenAI or ChatHuggingFace model.
    """
    try:
        if repo_id == "ChatGPT":
            model_name = kwargs.get("model_name", "gpt-4o-mini")
            chat_model = ChatOpenAI(
                openai_api_key = kwargs.get("openai_api_key", None),
                base_url = "https://openrouter.ai/api/v1", 
                model = "google/gemini-flash-1.5",
                temperature = 0
            )
        else:
            logging.info(f"Loading Hugging Face model: {repo_id}")
            huggingfacehub_api_token = kwargs.get("hf_api_key", None)
            if not huggingfacehub_api_token:
                huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
            if not huggingfacehub_api_token:
                raise ValueError("HuggingFace Hub API token not found. "
                                 "Set HUGGINGFACEHUB_API_TOKEN environment variable.")
            os.environ["HF_TOKEN"] = huggingfacehub_api_token

            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                temperature=0.1,
                huggingfacehub_api_token=huggingfacehub_api_token,
            )
            chat_model = ChatHuggingFace(llm=llm).bind(max_tokens=1024)
        return chat_model
    except Exception as e:
        logging.error(f"Error loading model '{repo_id}': {e}")


def basic_chain(model=None, prompt=None):
    """
    Creates a basic LangChain chain with a prompt and a language model.

    Args:
        model: The language model to use.
        prompt: The prompt template.

    Returns:
        A LangChain chain.
    """
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template("Tell me the most noteworthy books by the author {author}")

    chain = prompt | model
    return chain


def main():
    """
    Main function to demonstrate the basic chain.
    """
    load_dotenv()

    prompt = ChatPromptTemplate.from_template("Tell me the most noteworthy books by the author {author}")
    chain = basic_chain(prompt=prompt) | StrOutputParser()

    try:
        results = chain.invoke({"author": "William Faulkner"})
        print(results)
    except Exception as e:
        logging.error(f"Error during chain execution: {e}")


if __name__ == '__main__':
    main()