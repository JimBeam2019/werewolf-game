from langchain_core.vectorstores import InMemoryVectorStore
from domain.entities import Player
from infrastructure.retriever import (
    retrieve_from_vector_store,
    retrieve_from_vllm_vector_store,
)


class RAGBackgroundKnowledgeProvider:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        use_vllm: bool = False,
        vllm_model: str = "",
        vllm_base_url: str = "",
    ) -> None:
        self.vector_store = vector_store
        self.use_vllm = use_vllm
        self.vllm_model = vllm_model
        self.vllm_base_url = vllm_base_url

    def get_background(self, player: Player) -> str:

        return (
            retrieve_from_vllm_vector_store(
                self.vector_store,
                (
                    "Use 3-4 sentences to present personality and strategy for "
                    f"{player.name}, who is a {player.role.value}"
                ),
                self.vllm_model,
                self.vllm_base_url,
            )
            if self.use_vllm
            else retrieve_from_vector_store(
                self.vector_store,
                f"Personality for {player.name}",
                f"Strategy for a {player.role.value}",
            )
        )
