import random
from datetime import datetime, timezone
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from langgraph_memory.hosting import container
from langgraph_memory.models.chat_message import ChatMessage
from langgraph_memory.protocols.i_azure_openai_service import IAzureOpenAIService
from langgraph_memory.protocols.i_memory_store import IMemoryStore

llm = container[IAzureOpenAIService].get_model()
memory_store = container[IMemoryStore]


def get_lucky_number(city: str) -> int:
    """Get a lucky number."""
    return random.randint(1, 100)


number_picker = create_react_agent(
    llm,
    tools=[get_lucky_number],
    prompt=(
        "You are a lucky number picker agent. You are tasked to provide a lucky number "
        "between 1 and a number that the user provides. User may also ask you to list "
        "the lucky numbers that you have provided so far."
    ),
)


def call_number_picker(
    state: MessagesState,
) -> Command[Literal["human"]]:
    response = number_picker.invoke(state)
    return Command(update=response, goto="human")


def human_node(state: MessagesState, config) -> Command[Literal["number_picker"]]:
    """A node for collecting user input."""
    return interrupt(value="Ready for user input.")


def setup(thread_id: str) -> tuple[CompiledStateGraph, MemorySaver]:
    builder = StateGraph(MessagesState)
    builder.add_node("number_picker", call_number_picker)
    builder.add_node("human", human_node)

    # We'll always start with a general travel advisor.
    builder.add_edge(START, "number_picker")

    # restore the memory from the store
    checkpointer = memory_store.restore(thread_id)
    graph = builder.compile(checkpointer=checkpointer)
    return graph, checkpointer


def serve(thread_id: str, input_message: str) -> str:
    # capture the user input in history
    history = [
        ChatMessage(
            message=input_message,
            role="user",
            domain="demo",
            ts=datetime.now(timezone.utc).timestamp(),
        )
    ]

    graph, checkpointer = setup(thread_id)
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    user_input = {"messages": [{"role": "user", "content": input_message}]}
    while True:
        for update in graph.stream(
            user_input,
            config=thread_config,
            stream_mode="updates",
        ):
            for node_id, value in update.items():
                if isinstance(value, dict) and value.get("messages", []):
                    last_message = value["messages"][-1]
                    if isinstance(last_message, dict) or last_message.type != "ai":
                        continue

                    # capture the ai input in history
                    history.append(
                        ChatMessage(
                            message=last_message.content,
                            role="ai",
                            domain="demo",
                            ts=datetime.now(timezone.utc).timestamp(),
                        )
                    )

                    # store the updated history and memory
                    memory_store.put(thread_id, history, checkpointer)

                    # return the last AI message content to main.py
                    return last_message.content
