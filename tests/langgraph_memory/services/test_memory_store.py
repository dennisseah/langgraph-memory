import json
import os
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from langgraph_memory.models.chat_message import ChatMessage
from langgraph_memory.services.memory_store import (
    BASE_FOLDER,
    GRAPH_MEMORY_FILE,
    HISTORY_FILE,
    MemoryStore,
)


@pytest.fixture
def mock_chat_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", message="Hello", domain="test", ts=1234567890.0),
        ChatMessage(role="ai", message="Hi there!", domain="test", ts=1234567891.0),
    ]


def test_read_history(mocker: MockerFixture, mock_chat_messages: list[ChatMessage]):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "builtins.open",
        mocker.mock_open(
            read_data=json.dumps([msg.model_dump() for msg in mock_chat_messages])
        ),
    )
    assert len(MemoryStore().read_history("test_key")) == 2


def test_read_history_none(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)
    assert MemoryStore().read_history("test_key") == []


def test_write_history(mocker: MockerFixture, mock_chat_messages: list[ChatMessage]):
    mocker.patch("os.makedirs")
    open_handle = mocker.mock_open()
    mocker.patch("builtins.open", open_handle)

    MemoryStore().write_history("test_key", mock_chat_messages)

    open_handle.assert_called_once_with(
        os.path.join(BASE_FOLDER, "test_key", HISTORY_FILE), "w"
    )


def test_read_graph_memory(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data=b"graph_memory_data"),
    )
    mocker.patch("dill.load", return_value=b"graph_memory_data")
    assert MemoryStore().read_graph_memory("test_key") == b"graph_memory_data"


def test_read_graph_memory_none(mocker: MockerFixture):
    mocker.patch("os.path.exists", return_value=False)
    assert MemoryStore().read_graph_memory("test_key") is None


def test_write_graph_memory(mocker: MockerFixture):
    mocker.patch("os.makedirs")
    open_handle = mocker.mock_open()
    mocker.patch("builtins.open", open_handle)
    dill_dump = mocker.patch("dill.dump")

    MemoryStore().write_graph_memory("test_key", MagicMock())

    open_handle.assert_called_once_with(
        os.path.join(BASE_FOLDER, "test_key", GRAPH_MEMORY_FILE), "wb"
    )
    dill_dump.assert_called_once()


def test_put(mocker: MockerFixture, mock_chat_messages: list[ChatMessage]):
    memory_store = MemoryStore()
    memory_store.write_history = mocker.MagicMock()
    memory_store.write_graph_memory = mocker.MagicMock()

    memory_store.put("test_key", mock_chat_messages, MagicMock())
    memory_store.write_history.assert_called_once()
    memory_store.write_graph_memory.assert_called_once()


def test_restore(mocker: MockerFixture):
    memory_store = MemoryStore()
    memory_store.read_graph_memory = mocker.MagicMock(return_value=None)
    restored_memory = memory_store.restore("test_key")
    assert len(restored_memory.blobs) == 0


def test_restore_data(mocker: MockerFixture):
    mocker.patch("langgraph_memory.services.memory_store.MemorySaver")
    memory_store = MemoryStore()
    memory_store.read_graph_memory = mocker.MagicMock(return_value=MagicMock())
    result = memory_store.restore("test_key")
    assert result is not None


def test_get_chat_history(mocker: MockerFixture):
    memory_store = MemoryStore()
    memory_store.read_history = mocker.MagicMock(return_value=MagicMock())
    result = memory_store.get_chat_history("test_key")
    assert result is not None
