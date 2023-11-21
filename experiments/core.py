import abc
import asyncio
from enum import Enum
import re
from typing import Any, Dict, List

from langchain.utils.math import cosine_similarity
from langchain.vectorstores.utils import maximal_marginal_relevance
import numpy as np
import pandas as pd
from pydantic import BaseModel

from experiments.utils import async_apply, get_embedding_apply_fn, get_inference_apply_fn
from momentum.response_example import InContextExample

EXAMPLE_PROMPT_COL = "ex_prompt"
EXAMPLE_CONTENT_COL = "ex_content"
EXAMPLE_CORRECT_COL = "is_correct"
EMBEDDING_COL = "embedding"


class ExampleSelectionMethod(str, Enum):
    # TODO: Add FIXED method
    NONE = "none"
    RANDOM = "random"
    MOMENTUM_MMR = "momentum_mmr"
    MOMENTUM_SIM = "momentum_sim"


class ExperimentConfig(BaseModel):
    """Parameters for running an experiment"""
    num_examples: int
    example_selection_method: ExampleSelectionMethod
    seed: int

    @property
    def uses_examples(self) -> bool:
        """Checks if the experiment uses examples"""
        return self.num_examples > 0 and self.example_selection_method != ExampleSelectionMethod.NONE

    @property
    def needs_embedding(self) -> bool:
        """Checks if the experiment requires an embedding"""
        return self.example_selection_method in [ExampleSelectionMethod.MOMENTUM_MMR, ExampleSelectionMethod.MOMENTUM_SIM]

    def select_examples(
        self,
        row: pd.Series,
        df: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Selects examples from the provided dataframe for the row"""
        if self.params.method == ExampleSelectionMethod.NONE or self.params.num_examples == 0:
            return []

        # Filter to just the correct examples
        df = df[df[EXAMPLE_CORRECT_COL]]

        if self.params.method == ExampleSelectionMethod.RANDOM:
            return self._df_to_icl_examples(
                df.sample(n=self.params.num_examples)
            )

        elif self.params.method == ExampleSelectionMethod.MOMENTUM_MMR:
            best_indices = maximal_marginal_relevance(
                query_embedding=np.array(row["embedding"]),
                embedding_list=df["embedding"].tolist(),
                k=self.params.num_examples,
            )
            return self._df_to_icl_examples(
                df.iloc[best_indices]
            )

        elif self.params.method == ExampleSelectionMethod.MOMENTUM_SIM:
            # Can assume the current row is not in df
            sims = cosine_similarity([row["embedding"]], df["embedding"].tolist())
            # Get indices of num_examples most similar examples
            closest_indices = sims[0].argsort()[::-1][:self.params.num_examples]
            return self._df_to_icl_examples(
                df.iloc[closest_indices]
            )

        else:
            raise NotImplementedError

    @staticmethod
    def _df_to_icl_examples(df: pd.DataFrame) -> List[InContextExample]:
        """Converts a dataframe to a list of InContextExample objects"""
        return [
            InContextExample(
                id=str(row.name),
                prompt=row[EXAMPLE_PROMPT_COL],
                content=row[EXAMPLE_CONTENT_COL],
            )
            for ndx, row in df.iterrows()
        ]


class BaseTask(abc.ABC):
    """Base class for running momentum experiments for a specific task"""
    def __init__(
        self,
        eval_df: pd.DataFrame,
        exp_config: ExperimentConfig,
        *,
        llm_completions: pd.Series | None = None,
        examples_df: pd.DataFrame | None = None,
    ):
        self.eval_df = eval_df
        self.exp_config = exp_config
        self.existing_llm_completions = llm_completions
        self.artifacts = {}

        if examples_df is None:
            examples_df = eval_df.copy()
        self.examples_df = examples_df

    async def arun_experiment(self, populate_examples: bool = False) -> float:
        """Runs the experiment asynchronously and returns the accuracy"""
        np.random.seed(self.exp_config.seed)

        if not self._examples_df_has_required_cols():
            if populate_examples:
                await self.populate_examples()
            else:
                raise ValueError("examples_df is missing required columns")

        llm_completions = await self.generate_completions()
        self.store_artifact("llm_completions", llm_completions)

        return await self.evaluate_accuracy(llm_completions=llm_completions)

    def run_experiment(self, *args, **kwargs) -> float:
        """Runs the experiment and returns the accuracy"""
        return asyncio.run(self.arun_experiment(*args, **kwargs))

    @abc.abstractmethod
    async def evaluate_accuracy(self, llm_completions: pd.Series) -> float:
        """Evaluates the accuracy of the provided LLM completions"""
        raise NotImplementedError

    @abc.abstractmethod
    def create_user_question(self, row: pd.Series) -> str:
        """Creates a user question from the provided row"""
        raise NotImplementedError

    @abc.abstractmethod
    def create_system_prompt(self, row: pd.Series) -> str:
        """Creates a system prompt from the provided row"""
        raise NotImplementedError

    def create_prompt_msgs(
        self,
        row: pd.Series,
        examples: List[InContextExample],
    ) -> List[Dict[str, str]]:
        """Creates the prompt messages to send to the LLM"""
        messages = [
            {"role": "system", "content": self.create_system_prompt(row)},
        ]
        if examples is not None:
            for example in examples:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": example.prompt,
                        },
                        {
                            "role": "assistant",
                            "content": example.content,
                        }
                    ]
                )

        messages.append(
            {
                "role": "user",
                "content": self.create_user_question(row),
            },
        )
        return messages

    def store_artifact(self, key: str, value: Any) -> None:
        """Stores an artifact from the experiment"""
        self.artifacts[key] = value

    def get_artifact(self, key: str) -> Any:
        """Retrieves an artifact from the experiment"""
        return self.artifacts[key]

    async def generate_completions(self) -> pd.Series:
        """Runs inference over the provided eval_df and returns the completions"""
        if self.existing_llm_completions is not None:
            # Shortcut to exit early if already provided
            return self.existing_llm_completions

        return await self._generate_completions()

    async def populate_examples(self, df: pd.DataFrame):
        """Populates the ex_prompt and ex_content columns of the provided dataframe

        NOTE: Mutates df
        """
        df["ex_prompt"] = df.apply(lambda row: self.create_user_question(row), axis=1)

        if "ex_content" not in df.columns:
            # Generate completions for the base Zero Shot case
            llm_completions = await self._generate_completions(
                exp_config=ExperimentConfig(
                    num_examples=0,
                    example_selection_method=ExampleSelectionMethod.NONE,
                    seed=self.exp_config.seed,
                )
            )
            df["ex_content"] = llm_completions

        # TODO
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

    async def _generate_completions(
        self,
        exp_config: ExperimentConfig | None = None,
    ) -> pd.Series:
        """Runs inference over eval_df and returns the completions"""
        if exp_config is None:
            exp_config = self.exp_config

        def create_prompt_fn(row: pd.Series) -> List[Dict[str, str]]:
            try:
                # Cannot use the current row as an example if it is present
                # TODO: Account for Row IDs being from different sets
                filtered_examples_df = self.examples_df.drop(row.name)
            except KeyError:
                filtered_examples_df = self.examples_df

            examples = exp_config.select_examples(
                row=row,
                df=filtered_examples_df,
            )
            return self.create_prompt_msgs(row, examples=examples)

        return await async_apply(
            self.eval_df,
            get_inference_apply_fn(
                create_prompt_fn=create_prompt_fn,
            )
        )

    def _examples_df_has_required_cols(self) -> bool:
        """Checks if the provided dataframe has the required columns"""
        if not self.exp_config.uses_examples:
            # Don't need examples, so nothing to check
            return True

        required_cols = [EXAMPLE_PROMPT_COL, EXAMPLE_CONTENT_COL, EXAMPLE_CORRECT_COL]
        if self.exp_config.needs_embedding:
            # Need embeddings for these methods
            required_cols.append(EMBEDDING_COL)

        for col in required_cols:
            if col not in self.examples_df.columns:
                return False
        return True
