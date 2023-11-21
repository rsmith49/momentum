import asyncio
from typing import Any, Callable, Coroutine, Dict, List

from dotenv import load_dotenv
from httpx import Timeout
import openai
import pandas as pd
from tenacity import retry, wait_exponential
from tqdm.asyncio import tqdm as atqdm

async_openai_client: openai.AsyncOpenAI | None = None


def _init_openai_client() -> None:
    global async_openai_client

    if async_openai_client is None:
        load_dotenv()
        async_openai_client = openai.AsyncOpenAI(
            timeout=Timeout(120, connect=5),
        )


async def async_apply(
    df: pd.DataFrame,
    fn: Callable[[pd.Series], Coroutine],
    max_concurrent: int = 200,
) -> pd.Series:
    """Apply a function to each row of a dataframe asynchronously."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await fn(*args, **kwargs)

    new_col = await atqdm.gather(
        *(wrapper(row) for ndx, row in df.iterrows())
    )
    return new_col


def get_embedding_apply_fn(
    create_prompt_fn: Callable[[pd.Series], str],
    model: str = "text-embedding-ada-002",
    **kwargs: Any,
):
    """Create a function that can be used to apply an embedding model to a dataframe."""
    if async_openai_client is None:
        _init_openai_client()

    @retry(wait=wait_exponential(multiplier=1.5, min=1, max=10))
    async def async_apply_fn(row: pd.Series) -> List[float]:
        embedding_response = await async_openai_client.embeddings.create(
            input=create_prompt_fn(row),
            model=model,
            **kwargs,
        )
        # TODO: Do we want to count tokens anywhere?
        return embedding_response.data[0].embedding

    return async_apply_fn


def get_inference_apply_fn(
    create_prompt_fn: Callable[[pd.Series], List[Dict[str, Any]]],
    model: str = "gpt-3.5-turbo-1106",
    temperature: float = 0,
    **kwargs: Any,
):
    """Create a function that can be used to apply an inference model to a dataframe."""
    if async_openai_client is None:
        _init_openai_client()

    @retry(wait=wait_exponential(multiplier=1.5, min=1, max=10))
    async def async_apply_fn(row: pd.Series) -> openai.ChatCompletion:
        completion_response = await async_openai_client.chat.completions.create(
            messages=create_prompt_fn(row),
            model=model,
            temperature=temperature,
            **kwargs,
        )
        # TODO: Do we want to count tokens anywhere?
        return completion_response.choices[0].message.content

    return async_apply_fn
