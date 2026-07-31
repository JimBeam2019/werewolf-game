from langchain_core.vectorstores import InMemoryVectorStore
from domain.entities import Player
from infrastructure.retriever import retrieve_from_vector_store


class RAGBackgroundKnowledgeProvider:
    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self.vector_store = vector_store

    def get_background(self, player: Player) -> str:
        # query = (
        #     f"Personality and strategy for "
        #     f"{player.name}, who is a {player.role.value}"
        # )

        return retrieve_from_vector_store(
            self.vector_store,
            f"Personality for {player.name}",
            f"Strategy for a {player.role.value}",
        )

        # documents = self.vector_store.similarity_search(query, k=3)

        # return "-----\n\n".join(document.page_content for document in documents)
