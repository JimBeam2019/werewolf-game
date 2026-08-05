import streamlit as st

from uuid import uuid4
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_vector_store(
    chunks: list[Document], embedding: OpenAIEmbeddings | OllamaEmbeddings
):
    """Builds (once per session) or returns the cached vector store.

    Caching in st.session_state matters here specifically because
    building this means embedding every chunk of every knowledge file -
    re-doing that on every Streamlit rerun (which happens on essentially
    every click) would be slow and pointless, since the knowledge base
    content never changes mid-session.
    """
    if not chunks:
        st.error("No game content was found...")
        return None

    # If the vector store is not already present in the session state
    if not st.session_state.vector_store:
        with st.spinner(text=":red[Please wait while we fetch the information...]"):
            vector_store = InMemoryVectorStore(embedding=embedding)

            uuids = [str(uuid4()) for _ in chunks]
            vector_store.add_documents(documents=chunks, ids=uuids)

            st.session_state.vector_store = vector_store

    return st.session_state.vector_store


def retrieve_from_vector_store(
    vector_store: InMemoryVectorStore, player_name: str, role_value: str
) -> str:
    """Retrieves one player's personality profile plus a relevant
    strategy snippet for their role.

    Personality lookup is an *exact* metadata match (type + entity), not a
    similarity search: there's exactly one right document per player, so
    treating it as "find the most similar thing" risks returning a
    different player's profile if a small/quantized embedding model
    doesn't discriminate well between two people's writeups. `k=3` (not
    1) with that exact filter is still safe *and* handles personality
    files long enough to have been split into more than one chunk by the
    text splitter upstream - every chunk actually belonging to this
    player still comes back, ordered by `start_index` so they read in
    the original order rather than however the search happened to rank
    them.

    Strategy lookup is a genuine similarity search (MMR, for some result
    diversity) since there's no single "correct" strategy doc - but it's
    still filtered to `type == "strategy"` so it can never surface
    another player's personality text as if it were strategy advice.
    """
    personality_results = vector_store.similarity_search_with_score(
        f"Personality for {player_name}",
        k=3,
        filter=lambda doc: (
            doc.metadata.get("type") == "personality"
            and doc.metadata.get("entity") == player_name
        ),
    )

    personality_chunks = sorted(
        (doc for doc, _score in personality_results),
        key=lambda doc: doc.metadata.get("start_index", 0),
    )
    personality_text = "".join(doc.page_content for doc in personality_chunks)
    if not personality_text:
        personality_text = f"(No personality profile found for {player_name}.)"

    strategy_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 1,
            "fetch_k": 5,
            "lambda_mult": 0.25,
            "filter": lambda doc: doc.metadata.get("type") == "strategy",
        },
    )

    strategy_docs = strategy_retriever.invoke(f"Strategy for a {role_value}")
    strategy_text = "\n\n-----\n\n".join(doc.page_content for doc in strategy_docs)

    return f"{personality_text}\n\n-----\n\n{strategy_text}"


def _format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata}\n{doc.page_content}" for doc in docs
    )


def retrieve_from_vllm_vector_store(
    vector_store: InMemoryVectorStore,
    player_name: str,
    role_value: str,
    vllm_model: str,
    inference_server_url: str,
):
    """
    Function for retrieving the relevant chunks from the vector store
    """
    query = (
        "Use 3-4 sentences to present personality and strategy for "
        f"{player_name}, who is a {role_value}"
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "fetch_k": 5},
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context. If the context
    does not contain enough information to answer the question, say so clearly.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    llm = ChatOpenAI(
        model=vllm_model,
        api_key="EMPTY",  # type: ignore
        base_url=inference_server_url,
        temperature=0.7,
    )

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)
