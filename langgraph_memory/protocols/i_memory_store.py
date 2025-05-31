from typing import Protocol

from langgraph.checkpoint.memory import MemorySaver

from langgraph_memory.models.chat_message import ChatMessage


class IMemoryStore(Protocol):
    def put(
        self, key: str, messages: list[ChatMessage], graph_memory: MemorySaver
    ) -> None:
        """
        Store a value in the memory store with the given key.

        :param key: The key under which to store the value.
        :param messages: The list of ChatMessage instances to store.
        :param graph_memory: The MemorySaver instance to use for storing the memory.
        """
        ...

    def restore(self, key: str) -> MemorySaver:
        """
        Restore a value from the memory store with the given key.

        :param key: The key under which the value is stored.
        :return: The MemorySaver instance containing the restored memory.
        """
        ...

    def get_chat_history(self, key: str) -> list[ChatMessage]:
        """
        Read the history of messages stored under the given key.

        :param key: The key under which the history is stored.
        :return: A list of ChatMessage instances representing the history.
        """
        ...
