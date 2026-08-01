import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from infrastructure.retriever import build_vector_store


@st.cache_data
def load_local_md_files(bot_names: list[str]):
    """
    Function for loading local MD files
    """

    root_path = "knowledge"
    personality_path = "personalities"
    strategy_path = "strategies"
    strategy_files = [
        "communication",
        "villager_basic",
        "voting_patterns",
        "werewolf_basic",
        "werewolf_bluffing",
    ]

    documents: list[Document] = []

    for bot_name in bot_names:
        file_path = f"{root_path}/{personality_path}/{bot_name.lower()}.md"
        with open(file_path, "r") as file:
            file_content = file.read()

        document = Document(
            page_content=file_content,
            metadata={"type": "personality", "entity": bot_name},
        )
        documents.append(document)

    for strategy in strategy_files:
        file_path = f"{root_path}/{strategy_path}/{strategy}.md"
        with open(file_path, "r") as file:
            file_content = file.read()

        document = Document(
            page_content=file_content,
            metadata={"type": "strategy", "entity": strategy},
        )
        documents.append(document)

    return documents


def load_files(bot_names: list[str]):
    """
    Function for loading files
    """
    with st.spinner("Loading game content. Please wait around a minute..."):
        content = load_local_md_files(bot_names)

    if content:
        return content
    else:
        st.error("Game content not found")
        return None


@st.cache_resource
def initialize_knowledge_base(
    bot_names: list[str], embedding: OpenAIEmbeddings | OllamaEmbeddings
):
    """
    Initialize game knowledge base
    """
    documents = load_files(bot_names)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    chunks = splitter.split_documents(documents)  # type: ignore

    return build_vector_store(chunks, embedding)  # type: ignore
