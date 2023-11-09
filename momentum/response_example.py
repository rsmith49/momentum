from typing import Any, Dict, List

from pydantic import BaseModel, Extra


class InContextExample(BaseModel):
    """Example for use in In-Context Learning that can stored in a vectorstore"""
    id: str
    prompt: str
    content: str

    class Config:
        extra = Extra.forbid


class ResponseExample(InContextExample):
    """Response/Example from querying an LLM agent

    Responses and examples are inherently linked in the design of momentum, since
    the agent learns from the responses it produces, treating them like examples.
    """
    in_context_examples: List["ResponseExample"]
