import streamlit as st

from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_vector_store(chunks: list[Document]):
    """
    Function for retrieving the vector store
    """
    if chunks:
        # If the vector store is not already present in the session state
        if not st.session_state.vector_store:
            with st.spinner(text=":red[Please wait while we fetch the information...]"):
                embedding = OllamaEmbeddings(model="qwen3-embedding", dimensions=768)
                vector_store = InMemoryVectorStore(embedding=embedding)

                uuids = [str(uuid4()) for _ in chunks]
                vector_store.add_documents(documents=chunks, ids=uuids)

                st.session_state.vector_store = vector_store
                return vector_store
        else:
            return st.session_state.vector_store
    else:
        st.error("No game content was found...")
        return None


# def _format_docs(docs):
#     return "\n\n---\n\n".join(
#         f"Source: {doc.metadata}\n{doc.page_content}" for doc in docs
#     )


def retrieve_from_vector_store(
    vector_store: InMemoryVectorStore, personality_query: str, strategy_query: str
):
    """
    Function for retrieving the relevant chunks from the vector store
    """
    personality_results = vector_store.similarity_search_with_score(
        personality_query, k=1, where_document={"type": "personality"}
    )
    personality_result, score = personality_results[0]
    personality_doc = personality_result.page_content

    print(f"Score: {score}")
    print(f"personality: {personality_doc}")

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "lambda_mult": 0.25}
    )

    documents = retriever.invoke(strategy_query)

    # prompt = ChatPromptTemplate.from_template("""
    # Answer the question based only on the following context. If the context
    # does not contain enough information to answer the question, say so clearly.

    # Context:
    # {context}

    # Question: {question}

    # Answer:
    # """)

    # llm = ChatOllama(model="llama3.2:3b")

    # rag_chain = (
    #     {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # result = rag_chain.invoke(query)

    print(documents)

    return (
        personality_doc
        + "\n\n"
        + "-----\n\n".join(document.page_content for document in documents)
    )
