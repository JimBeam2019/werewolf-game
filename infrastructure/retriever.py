import streamlit as st

from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


def build_vector_store(documents: list[Document]):
    """
    Function for retrieving the vector store
    """
    if documents:
        # If the vector store is not already present in the session state
        if not st.session_state.vector_store:
            with st.spinner(text=":red[Please wait while we fetch the information...]"):
                embedding = OllamaEmbeddings(model="qwen3-embedding", dimensions=768)
                vector_store = InMemoryVectorStore(embedding=embedding)
                uuids = [str(uuid4()) for _ in documents]
                vector_store.add_documents(documents=documents, ids=uuids)
                st.session_state.vector_store = vector_store
                return vector_store
        else:
            return st.session_state.vector_store
    else:
        st.error("No game content was found...")
        return None


def retrieve_from_vector_store(vector_store: InMemoryVectorStore, query: str):
    """
    Function for retrieving the relevant chunks from the vector store
    """
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
    )
    results = retriever.invoke(query)
    return results
