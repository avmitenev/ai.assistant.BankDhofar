import os
import logging

from langchain_community.retrievers import BM25Retriever, TavilySearchAPIRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser

from basic_chain import get_model
from rag_chain import make_rag_chain
from remote_loader import load_web_page
from splitter import split_documents
from vector_store import create_vector_db
from dotenv import load_dotenv


def ensemble_retriever_from_docs(docs, embeddings=None):
    texts = split_documents(docs)
    vs = create_vector_db(texts, embeddings)
    vs_retriever = vs.as_retriever(search_kwargs={'k': 2})

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])
    bm25_retriever.k = 2

    tavily_retriever = MyTavilySearchAPIRetriever(k=3, include_domains=['https://www.bankdhofar.com/'])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever, tavily_retriever],
        weights=[0.33, 0.33, 0.33])

    return ensemble_retriever


class MyTavilySearchAPIRetriever(TavilySearchAPIRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ):
        try:
            return super()._get_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            logging.error(f"TavilySearch error: {e}")
            return []


def main():
    load_dotenv()

    problems_of_philosophy_by_russell = "https://www.gutenberg.org/ebooks/5827.html.images"
    docs = load_web_page(problems_of_philosophy_by_russell)
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    model = get_model("ChatGPT")
    chain = make_rag_chain(model, ensemble_retriever) | StrOutputParser()

    result = chain.invoke("What are the key problems of philosophy according to Russell?")
    print(result)


if __name__ == "__main__":
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

