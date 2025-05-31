from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    message: str
    role: Literal["user", "ai"]
    domain: str
    ts: float
