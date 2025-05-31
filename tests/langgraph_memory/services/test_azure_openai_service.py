from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from langgraph_memory.services.azure_openai_service import (
    AzureOpenAIService,
)


def test_get_client(mocker: MockerFixture):
    patched_azure_openai = mocker.patch(
        "langgraph_memory.services.azure_openai_service.AzureChatOpenAI",
        autospec=True,
    )
    svc = AzureOpenAIService(env=MagicMock())
    assert svc.get_model() is not None
    patched_azure_openai.assert_called_once()


def test_get_client_no_api_key(mocker: MockerFixture):
    patched_azure_openai = mocker.patch(
        "langgraph_memory.services.azure_openai_service.AzureChatOpenAI",
        autospec=True,
    )
    patched_token = mocker.patch(
        "langgraph_memory.services.azure_openai_service.DefaultAzureCredential",
    )
    env = MagicMock()
    env.azure_openai_api_key = None
    svc = AzureOpenAIService(env=env)
    assert svc.get_model() is not None

    patched_azure_openai.assert_called_once()
    patched_token.assert_called_once()
