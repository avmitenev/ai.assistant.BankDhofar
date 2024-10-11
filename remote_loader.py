import os

from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader, OnlinePDFLoader
from langchain.docstore.document import Document
import requests

CONTENT_DIR = os.path.dirname(__file__)

def load_web_page(page_url):
    """Loads a web page and returns a list of Document objects."""
    try:
        loader = WebBaseLoader(page_url)
        return loader.load()
    except requests.exceptions.RequestException as e:
        print(f"Error loading web page from {page_url}: {e}")
        return []

def load_online_pdf(pdf_url):
    """Loads an online PDF and returns a list of Document objects."""
    try:
        loader = OnlinePDFLoader(pdf_url)
        return loader.load()
    except Exception as e:
        print(f"Error loading online PDF from {pdf_url}: {e}")
        return []

def load_wiki_articles(query, load_max_docs=2):
    """Fetches Wikipedia articles related to a query and returns a list of Document objects."""
    try:
        wiki_loader = WikipediaLoader(query=query, load_max_docs=load_max_docs)
        return wiki_loader.load()
    except Exception as e:
        print(f"Error loading Wikipedia articles for query '{query}': {e}")
        return []
