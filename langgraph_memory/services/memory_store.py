import json
import os

import dill
from langgraph.checkpoint.memory import MemorySaver

from langgraph_memory.models.chat_message import ChatMessage
from langgraph_memory.protocols.i_memory_store import IMemoryStore

# this can be redis, for simpification we use file system

BASE_FOLDER = ".graph_store"
HISTORY_FILE = "history.json"
GRAPH_MEMORY_FILE = "graph_memory.pkl"

os.makedirs(BASE_FOLDER, exist_ok=True)


class MemoryStore(IMemoryStore):
    def read_history(self, key: str) -> list[ChatMessage]:
        dir_path = os.path.join(BASE_FOLDER, key)
        history_file_path = os.path.join(dir_path, HISTORY_FILE)

        if os.path.exists(history_file_path):
            with open(history_file_path, "r") as f:
                return [ChatMessage(**x) for x in json.load(f)]
        return []

    def write_history(self, key: str, history: list[ChatMessage]) -> None:
        dir_path = os.path.join(BASE_FOLDER, key)
        history_file_path = os.path.join(dir_path, HISTORY_FILE)
        all_history = self.read_history(key) + history

        os.makedirs(dir_path, exist_ok=True)
        with open(history_file_path, "w") as f:
            json.dump([msg.model_dump() for msg in all_history], f)

    def read_graph_memory(self, key: str) -> bytes | None:
        dir_path = os.path.join(BASE_FOLDER, key)
        graph_memory_file_path = os.path.join(dir_path, GRAPH_MEMORY_FILE)

        if os.path.exists(graph_memory_file_path):
            with open(graph_memory_file_path, "rb") as fp:
                return dill.load(fp)

        return None

    def write_graph_memory(self, key: str, graph_memory: MemorySaver) -> None:
        dir_path = os.path.join(BASE_FOLDER, key)
        graph_memory_file_path = os.path.join(dir_path, GRAPH_MEMORY_FILE)

        os.makedirs(dir_path, exist_ok=True)
        with open(graph_memory_file_path, "wb") as fp:
            dill.dump(graph_memory.__dict__, fp)

    def put(
        self, key: str, messages: list[ChatMessage], graph_memory: MemorySaver
    ) -> None:
        self.write_history(key, messages)
        self.write_graph_memory(key, graph_memory)

    def restore(self, key: str) -> MemorySaver:
        graph_memory = self.read_graph_memory(key)

        memory = MemorySaver()

        if graph_memory is None:
            return memory

        memory.__dict__.update(graph_memory)  # type: ignore
        return memory

    def get_chat_history(self, key: str) -> list[ChatMessage]:
        return self.read_history(key)
