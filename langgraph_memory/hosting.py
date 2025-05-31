"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging
import os

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from langgraph_memory.protocols.i_azure_openai_service import IAzureOpenAIService
from langgraph_memory.protocols.i_memory_store import IMemoryStore

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def logger() -> logging.Logger:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "ERROR"))
    logging.Formatter(fmt=" %(name)s :: %(levelname)-8s :: %(message)s")
    return logging.getLogger("langgraph_memory")


@dependency_definition(container, singleton=True)
def azure_openai_service() -> IAzureOpenAIService:
    from langgraph_memory.services.azure_openai_service import (
        AzureOpenAIService,
    )

    return container[AzureOpenAIService]


@dependency_definition(container, singleton=True)
def memory_store() -> IMemoryStore:
    from langgraph_memory.services.memory_store import MemoryStore

    return container[MemoryStore]
