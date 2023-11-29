import asyncio
from typing import Any, Callable, Coroutine, Dict, List

from dotenv import load_dotenv
from httpx import Timeout
from langchain.utils.math import cosine_similarity
import openai
import numpy as np
import pandas as pd
from tenacity import retry, wait_exponential
import tiktoken
from tqdm.asyncio import tqdm as atqdm

async_openai_client: openai.AsyncOpenAI | None = None


def _init_openai_client() -> None:
    global async_openai_client

    if async_openai_client is None:
        load_dotenv()
        async_openai_client = openai.AsyncOpenAI(
            timeout=Timeout(90, connect=5),
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
    return pd.Series(new_col, index=df.index)


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


def num_tokens_for_messages(
    messages: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo-1106",
) -> int:
    """Count the number of tokens in a list of chat messages."""
    enc = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for msg in messages:
        assert "content" in msg
        if type(msg["content"]) is str:
            num_tokens += len(enc.encode(msg["content"]))
        elif type(msg["content"]) is list:
            num_tokens += sum([len(enc.encode(submsg)) for submsg in msg["content"]])
        else:
            raise ValueError(f"Unknown message content type: {type(msg['content'])}")

    return num_tokens


# Cost per input token in USD
MODEL_INPUT_PRICING_DICT = {
    "gpt-3.5-turbo-1106": 0.001 / 1_000,
    "gpt-3.5-turbo-instruct": 0.0015 / 1_000,
    "gpt-3.5-turbo": 0.001 / 1_000,
    "gpt-4": 0.03 / 1_000,
    "gpt-4-1106-preview": 0.01 / 1_000,
}

MODEL_OUTPUT_PRICING_DICT = {
    "gpt-3.5-turbo-1106": 0.002 / 1_000,
    "gpt-3.5-turbo-instruct": 0.002 / 1_000,
    "gpt-3.5-turbo": 0.002 / 1_000,
    "gpt-4": 0.06 / 1_000,
    "gpt-4-1106-preview": 0.03 / 1_000,
}


def cost_for_messages(
    messages: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo-1106",
):
    """Calculate the cost of a list of chat messages."""
    num_tokens = num_tokens_for_messages(messages, model)
    return num_tokens * MODEL_INPUT_PRICING_DICT[model]


def maximal_marginal_relevance(
    query_embedding: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    all_computed_sims: np.ndarray | None = None,
    query_sims: np.ndarray | None = None,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    # Validate inputs
    if query_embedding is None and query_sims is None:
        raise ValueError(
            "Either `query_embedding` or `query_sims` must be supplied"
        )
    if embeddings is None and all_computed_sims is None:
        raise ValueError(
            "Either `embeddings` or `all_computed_sims` must be supplied"
        )

    if query_sims is None:
        query_sims = cosine_similarity(query_embedding, embeddings)

    if all_computed_sims is None:
        all_computed_sims = cosine_similarity(embeddings, embeddings)

    most_similar_ndxs = [
        int(np.argmax(query_sims))
    ]

    while len(most_similar_ndxs) < min(k, all_computed_sims.shape[0] - 1):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = all_computed_sims[most_similar_ndxs]

        for i, query_score in enumerate(query_sims):
            if i in most_similar_ndxs:
                continue
            # Find the highest similarity to any of the selected examples
            redundant_score = similarity_to_selected[:, i].max()
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        most_similar_ndxs.append(idx_to_add)

    return most_similar_ndxs
