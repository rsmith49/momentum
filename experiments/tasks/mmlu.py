import asyncio
from enum import Enum
import re
from typing import Any, Dict, List

from langchain.utils.math import cosine_similarity
from langchain.vectorstores.utils import maximal_marginal_relevance
import numpy as np
import pandas as pd

from experiments.utils import async_apply, get_embedding_apply_fn, get_inference_apply_fn
from momentum.response_example import InContextExample

SYSTEM_PROMPT = """
You are a well rounded professional who is an expert in multiple domains. Answer the questions asked of you and enclose the final answer in [brackets].
"""
QUESTION_PROMPT_TEMPLATE = "Question: {question}\nPossible Choices: {choices}"


class ExampleSelectionMethod(str, Enum):
    # TODO: Add FIXED method
    NONE = "none"
    RANDOM = "random"
    MOMENTUM_MMR = "momentum_mmr"
    MOMENTUM_SIM = "momentum_sim"


def _df_to_icl_examples(df: pd.DataFrame) -> List[InContextExample]:
    """Converts a dataframe to a list of InContextExample objects"""
    return [
        InContextExample(
            id=str(row.name),
            prompt=row["ex_prompt"],
            content=row["ex_content"],
        )
        for ndx, row in df.iterrows()
    ]


def select_examples(
    row: pd.Series,
    df: pd.DataFrame,
    method: ExampleSelectionMethod,
    num_examples: int,
) -> pd.DataFrame | None:
    """Selects examples from the provided dataframe based on the provided method"""
    if method == ExampleSelectionMethod.NONE or num_examples == 0:
        return []

    # Filter to just the correct examples
    df = df[df["is_correct"]]

    if method == ExampleSelectionMethod.RANDOM:
        return _df_to_icl_examples(
            df.sample(n=num_examples)
        )

    elif method == ExampleSelectionMethod.MOMENTUM_MMR:
        best_indices = maximal_marginal_relevance(
            query_embedding=np.array(row["embedding"]),
            embedding_list=df["embedding"].tolist(),
            k=num_examples,
        )
        return _df_to_icl_examples(
            df.iloc[best_indices]
        )

    elif method == ExampleSelectionMethod.MOMENTUM_SIM:
        # Can assume the current row is not in df
        sims = cosine_similarity([row["embedding"]], df["embedding"].tolist())
        # Get indices of num_examples most similar examples
        closest_indices = sims[0].argsort()[::-1][:num_examples]
        return _df_to_icl_examples(
            df.iloc[closest_indices]
        )

    else:
        raise ValueError(f"Unknown example selection method: {method}")


def create_user_question(row: pd.Series) -> str:
    return QUESTION_PROMPT_TEMPLATE.format(
        question=row["question"],
        choices=row["choices"],
    )


def create_mmlu_prompt_msgs(
    row: pd.Series,
    examples: List[InContextExample] | None = None,
) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if examples is not None:
        for example in examples:
            messages.extend([
                {
                    "role": "user",
                    "content": example.prompt,
                },
                {
                    "role": "assistant",
                    "content": example.content,
                }
            ])

    messages.append(
        {
            "role": "user",
            "content": create_user_question(row),
        },
    )
    return messages


def compare_mmlu_answer(row: pd.Series, llm_completion: str) -> bool:
    """Helper to check if the predicted and true answers match"""
    def clean(s: str) -> str:
        s = s.lower().strip()
        symbols_to_clear = [".", "$", "#", "@", "(", ")", "&", "!", "?", ";", '"', "'", "`", "’", "‘"]
        if len(s) > 0 and s[-1] in symbols_to_clear:
            s = s[:-1]
            s = clean(s)
        if len(s) > 0 and s[0] in symbols_to_clear:
            s = s[1:]
            s = clean(s)

        return s

    true_answer = row["choices"][row["answer"]]
    # Fairly naive string matching
    matches = re.findall(r"\[([^\[\]]*)]", llm_completion)
    if len(matches) > 0:
        return clean(matches[-1]) == clean(true_answer)

    return False


async def get_completions(
    eval_df: pd.DataFrame,
    examples_df: pd.DataFrame | None = None,
    num_examples: int = 5,
    example_selection_method: ExampleSelectionMethod = ExampleSelectionMethod.NONE,
    seed: int = 123,
) -> pd.Series:
    """Runs inference over the provided eval_df and returns accuracy via `compare_mmlu_answer`"""
    np.random.seed(seed)

    if examples_df is None:
        examples_df = eval_df.copy()

    for col in ["ex_prompt", "ex_content", "is_correct"]:
        if num_examples == 0 or example_selection_method == ExampleSelectionMethod.NONE:
            # This is fine since we don't need examples
            pass
        elif col not in examples_df.columns:
            raise ValueError(f"examples_df must have {col} column")

    def create_prompt_fn(row: pd.Series) -> List[Dict[str, str]]:
        try:
            # Cannot use the current row as an example if it is present
            filtered_examples_df = examples_df.drop(row.name)
        except KeyError:
            filtered_examples_df = examples_df

        examples = select_examples(
            row=row,
            df=filtered_examples_df,
            method=example_selection_method,
            num_examples=num_examples,
        )
        return create_mmlu_prompt_msgs(row, examples=examples)

    return await async_apply(
        eval_df,
        get_inference_apply_fn(
            create_prompt_fn=create_prompt_fn,
        )
    )


async def populate_examples(df: pd.DataFrame):
    """Populates the ex_prompt and ex_content columns of the provided dataframe

    NOTE: Mutates df
    """
    df["ex_prompt"] = df.apply(lambda row: create_user_question(row), axis=1)

    if "ex_content" not in df.columns:
        llm_completions = await get_completions(
            eval_df=df,
            num_examples=0,
            example_selection_method=ExampleSelectionMethod.NONE,
        )
        df["ex_content"] = llm_completions

    df["is_correct"] = df.apply(
        lambda row: compare_mmlu_answer(row, row["ex_content"]),
        axis=1,
    )

    if "embedding" not in df.columns:
        df["embedding"] = await async_apply(
            df,
            get_embedding_apply_fn(
                create_prompt_fn=lambda row: row["ex_prompt"],
            ),
        )


async def arun_experiment(
    eval_df: pd.DataFrame,
    *,
    llm_completions: pd.Series | None = None,
    examples_df: pd.DataFrame | None = None,
    num_examples: int = 5,
    example_selection_method: ExampleSelectionMethod = ExampleSelectionMethod.NONE,
    seed: int = 123,
) -> float:
    """Runs the MMLU experiment asynchronously and returns the accuracy"""
    if llm_completions is None:
        llm_completions = await get_completions(
            eval_df=eval_df,
            examples_df=examples_df,
            num_examples=num_examples,
            example_selection_method=example_selection_method,
            seed=seed,
        )

    return eval_df.apply(
        lambda row: compare_mmlu_answer(row, llm_completions[row.name]),
        axis=1,
    ).mean()


def run_experiment(*args: Any, **kwargs: Any):
    """Runs the MMLU experiment and returns the accuracy"""
    return asyncio.run(arun_experiment(*args, **kwargs))
