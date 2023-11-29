import abc
import asyncio
from hashlib import md5
from itertools import product
from enum import Enum
from typing import Any, Dict, List, Type, cast

from langchain.utils.math import cosine_similarity
import numpy as np
import pandas as pd
from pydantic import BaseModel

from experiments.utils import (
    MODEL_OUTPUT_PRICING_DICT,
    async_apply,
    get_embedding_apply_fn,
    get_inference_apply_fn,
    cost_for_messages,
    maximal_marginal_relevance,
)
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
    model: str
    seed: int

    @property
    def uses_examples(self) -> bool:
        """Checks if the experiment uses examples"""
        return self.num_examples > 0 and self.example_selection_method != ExampleSelectionMethod.NONE

    @property
    def needs_embedding(self) -> bool:
        """Checks if the experiment requires an embedding"""
        return self.example_selection_method in [ExampleSelectionMethod.MOMENTUM_MMR, ExampleSelectionMethod.MOMENTUM_SIM]

    @property
    def run_name(self) -> str:
        """Returns the name of an experiment run with these params"""
        named_props = sorted(
            {
                "num_examples": self.num_examples,
                "method": self.example_selection_method.value,
                "seed": self.seed,
                "model": self.model,
            }.items()
        )
        return "__".join([
            f"{key}-{val}"
            for key, val in named_props
        ])

    def select_examples(
        self,
        row: pd.Series,
        df: pd.DataFrame,
        all_computed_sims: np.ndarray | None = None,
    ) -> pd.DataFrame | None:
        """Selects examples from the provided dataframe for the row"""
        if self.example_selection_method == ExampleSelectionMethod.NONE or self.num_examples == 0:
            return []

        # Filter to just the correct examples
        df = df[df[EXAMPLE_CORRECT_COL]]

        if all_computed_sims is not None:
            # Row index must be ints corresponding to the index of the similarity matrix
            assert type(row.name) is int

        if self.example_selection_method == ExampleSelectionMethod.RANDOM:
            return self._df_to_icl_examples(
                df.sample(n=self.num_examples)
            )

        elif self.example_selection_method == ExampleSelectionMethod.MOMENTUM_MMR:
            if all_computed_sims is None:
                best_indices = maximal_marginal_relevance(
                    query_embedding=np.array(row[EMBEDDING_COL]),
                    embeddings=np.array(df[EMBEDDING_COL].tolist()),
                    k=self.num_examples,
                )
            else:
                best_indices = maximal_marginal_relevance(
                    query_sims=all_computed_sims[row.name, df.index],
                    all_computed_sims=all_computed_sims[df.index][:, df.index],  # Only consider examples in df
                    k=self.num_examples,
                )

            return self._df_to_icl_examples(
                df.iloc[best_indices]
            )

        elif self.example_selection_method == ExampleSelectionMethod.MOMENTUM_SIM:
            if all_computed_sims is not None:
                sims = all_computed_sims[row.name, df.index]
            else:
                sims = cosine_similarity([row[EMBEDDING_COL]], df[EMBEDDING_COL].tolist())[0]

            # Get indices of num_examples most similar examples
            closest_indices = sims.argsort()[::-1][:self.num_examples]
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

    def __hash__(self) -> int:
        return int(md5(self.run_name.encode()).hexdigest(), 16)


class ArtifactStore(abc.ABC):
    """Class to represent how to save and load artifacts"""
    @abc.abstractmethod
    def store(self, key: str, artifact: Any) -> None:
        pass

    @abc.abstractmethod
    def load_artifact(self, key: str) -> Any:
        pass


class ArtifactDictStore(ArtifactStore):
    """Simple in-memory dict storage method for artifacts"""
    def __init__(self):
        self._storage: Dict[str, Any] = {}

    def store(self, key: str, artifact: Any) -> None:
        self._storage[key] = artifact

    def load_artifact(self, key: str) -> Any:
        return self._storage[key]


class BaseTask(abc.ABC):
    """Base class for running momentum experiments for a specific task"""
    task_registry: Dict[str, Type["BaseTask"]] = {}
    task_name: str

    def __init_subclass__(cls, **kwargs):
        """Registers the subclass in the registry"""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "task_name"):
            raise NotImplementedError("Subclasses must define a task_name attribute")
        BaseTask.task_registry[cls.task_name] = cls

    def __init__(
        self,
        eval_df: pd.DataFrame,
        exp_config: ExperimentConfig,
        *,
        llm_completions: pd.Series | None = None,
        examples_df: pd.DataFrame | None = None,
        artifact_store: ArtifactStore | None = None,
        artifact_key_prefix: str = "",
        all_computed_sims: np.ndarray | None = None,
    ):
        self.eval_df = eval_df
        self.exp_config = exp_config
        self.existing_llm_completions = llm_completions
        self.examples_df = examples_df if examples_df is not None else eval_df.copy()
        self.artifact_store = artifact_store or ArtifactDictStore()
        self.artifact_key_prefix = artifact_key_prefix
        self.all_computed_sims = all_computed_sims

    async def arun_eval(self, populate_examples: bool = False) -> float:
        """Runs the experiment asynchronously and returns the accuracy"""
        np.random.seed(self.exp_config.seed)

        if not self._examples_df_has_required_cols():
            if populate_examples:
                await self.populate_examples(self.examples_df)
            else:
                raise ValueError("examples_df is missing required columns")

        llm_completions = await self.generate_completions()
        self.artifact_store.store(self.artifact_key_prefix + "llm_completions", llm_completions)

        return await self.evaluate_accuracy(llm_completions=llm_completions)

    async def arun_cost_estimate(self) -> float:
        """Returns a cost estimate of the experiment"""
        if not self._examples_df_has_required_cols():
            raise ValueError("examples_df is missing required columns")

        llm_input_costs = await self.generate_completions(estimate_costs=True)
        # Making some aggressive assumptions here
        # TODO: Make these assumptions configurable
        avg_tokens_output_per_row = 70
        llm_output_cost = MODEL_OUTPUT_PRICING_DICT[self.exp_config.model] * avg_tokens_output_per_row

        return (llm_input_costs + llm_output_cost).sum()

    def run_eval(self, *args, **kwargs) -> float:
        """Runs the experiment and returns the accuracy"""
        return asyncio.run(self.arun_eval(*args, **kwargs))

    @abc.abstractmethod
    def create_user_question(self, row: pd.Series) -> str:
        """Creates a user question from the provided row"""
        raise NotImplementedError

    @abc.abstractmethod
    def create_system_prompt(self, row: pd.Series) -> str:
        """Creates a system prompt from the provided row"""
        raise NotImplementedError

    @abc.abstractmethod
    def is_llm_correct(self, row: pd.Series, llm_completion: str) -> bool:
        """Checks if the provided LLM completion is correct for the provided row"""
        raise NotImplementedError

    async def evaluate_accuracy(self, llm_completions: pd.Series) -> float:
        """Evaluates the accuracy of the provided LLM completions

        NOTE: Override if the accuracy metric is not simply the mean of the is_correct column"""
        return self.eval_df.apply(
            lambda row: self.is_llm_correct(row, llm_completions[row.name]),
            axis=1,
        ).mean()

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

    async def generate_completions(self, **kwargs) -> pd.Series:
        """Runs inference over the provided eval_df and returns the completions"""
        if self.existing_llm_completions is not None:
            # Shortcut to exit early if already provided
            return self.existing_llm_completions

        return await self._generate_completions(**kwargs)

    async def populate_examples(self, df: pd.DataFrame) -> None:
        """Populates the ex_prompt and ex_content columns of the provided dataframe

        NOTE: Mutates df
        """
        df[EXAMPLE_PROMPT_COL] = df.apply(lambda row: self.create_user_question(row), axis=1)

        if EXAMPLE_CONTENT_COL not in df.columns:
            # Generate completions for the base Zero Shot case
            llm_completions = await self._generate_completions(
                exp_config=ExperimentConfig(
                    num_examples=0,
                    example_selection_method=ExampleSelectionMethod.NONE,
                    seed=self.exp_config.seed,
                    model=self.exp_config.model,
                )
            )
            df[EXAMPLE_CONTENT_COL] = llm_completions

        df[EXAMPLE_CORRECT_COL] = df.apply(
            lambda row: self.is_llm_correct(row, row[EXAMPLE_CONTENT_COL]),
            axis=1,
        )

        if EMBEDDING_COL not in df.columns:
            df[EMBEDDING_COL] = await async_apply(
                df,
                get_embedding_apply_fn(
                    create_prompt_fn=lambda row: row[EXAMPLE_PROMPT_COL],
                ),
            )

    async def _generate_completions(
        self,
        exp_config: ExperimentConfig | None = None,
        estimate_costs: bool = False,
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
                all_computed_sims=self.all_computed_sims,
            )
            return self.create_prompt_msgs(row, examples=examples)

        if estimate_costs:
            # TODO: Escalate model name here as an experiment config param
            async def apply_fn(row: pd.Series) -> float:
                return cost_for_messages(
                    create_prompt_fn(row),
                    model=self.exp_config.model,
                )
        else:
            apply_fn = get_inference_apply_fn(
                create_prompt_fn=create_prompt_fn,
                model=self.exp_config.model,
            )
        return await async_apply(self.eval_df, apply_fn)

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


def create_experiment_list_from_grid(param_grid: Dict[str, List[Any]]) -> List[ExperimentConfig]:
    """Helper to create a list of experiment configs from the cross product of params"""
    param_keys_list = sorted(param_grid.keys())
    all_param_combos = product(*[
        param_grid[key] for key in param_keys_list
    ])
    param_kwargs_list = [
        {
            param_key: param_combo[key_ndx]
            for key_ndx, param_key in enumerate(param_keys_list)
        }
        for param_combo in all_param_combos
    ]

    return [
        ExperimentConfig(**kwargs)
        for kwargs in param_kwargs_list
    ]


class ExperimentResults:
    def __init__(
        self,
        exp_configs: List[ExperimentConfig],
        accs: List[float],
        artifact_store: ArtifactStore,
    ):
        self.exp_configs = exp_configs
        self.accs = accs
        self.artifact_store = artifact_store

    def to_df(self) -> pd.DataFrame:
        """Output the accuracy results as a DataFrame"""
        return pd.DataFrame(
            [
                {
                    "accuracy": acc,
                    **exp_config.dict(),
                }
                for acc, exp_config in zip(self.accs, self.exp_configs)
            ]
        )


async def arun_experiment(
    task_name: str,
    eval_df: pd.DataFrame,  # TODO: Implement a task based DF loading mechanism,
    *,
    examples_df: pd.DataFrame | None = None,
    experiment_configs: List[ExperimentConfig] | None = None,
    param_grid: Dict[str, List[Any]] | None = None,
    verbose: bool = True,
    estimate_costs: bool = False,
) -> ExperimentResults:
    """"""
    if verbose:
        log_fn = print
    else:
        log_fn = lambda *args, **kwargs: None

    if experiment_configs is None and param_grid is None:
        raise ValueError("Must specify one of `experiment_configs` or `param_grid`")

    if experiment_configs is None:
        experiment_configs = create_experiment_list_from_grid(param_grid)

    if len({exp_config.model for exp_config in experiment_configs}) > 1:
        # TODO: Figure out what to do with multiple models
        raise NotImplementedError("Cannot currently run experiments with multiple models")

    all_results: List[float] = []
    artifact_store = ArtifactDictStore()
    task_cls = BaseTask.task_registry[task_name]

    # Populate the examples DF appropriately
    if examples_df is None:
        examples_df = eval_df.copy()

    # Populating Examples DF with necessary required columns
    log_fn(f"Creating {task_name} task for populating examples_df")
    simple_task = task_cls(
        eval_df=eval_df,
        exp_config=ExperimentConfig(
            num_examples=0,
            example_selection_method=ExampleSelectionMethod.NONE,
            seed=123,
            model=experiment_configs[0].model,
        ),
        examples_df=examples_df,
        artifact_store=artifact_store,
        artifact_key_prefix="BASE_ZERO_SHOT_RUN",
    )
    # TODO: Account for if this will overwrite existing columns & run unnecessarily
    if not estimate_costs:
        log_fn("Populating examples...")
        await simple_task.populate_examples(examples_df)

    # Compute all similarities upfront if needed
    if any(exp_config.needs_embedding for exp_config in experiment_configs):
        log_fn("Computing similarities...")
        all_computed_sims = cosine_similarity(
            # TODO: Fix this too
            # eval_df[EMBEDDING_COL].tolist(),
            examples_df[EMBEDDING_COL].tolist(),
            examples_df[EMBEDDING_COL].tolist(),
        )
    else:
        all_computed_sims = None

    # TODO: Look into making this loop async?
    for experiment_config in experiment_configs:
        log_fn(f"Running {task_name} task for {experiment_config.run_name}...")
        task = task_cls(
            eval_df=eval_df,
            exp_config=experiment_config,
            examples_df=examples_df,
            artifact_store=artifact_store,
            artifact_key_prefix=experiment_config.run_name,
            all_computed_sims=all_computed_sims,
        )
        if estimate_costs:
            all_results.append(
                await task.arun_cost_estimate()
            )
        else:
            all_results.append(
                await task.arun_eval(populate_examples=False)
            )

    if not estimate_costs:
        # TODO: This is hacky please fix
        # Adding a result for the zero shot case
        simple_llm_completions = examples_df[EXAMPLE_CONTENT_COL]
        simple_task.existing_llm_completions = simple_llm_completions
        all_results.insert(0, await simple_task.arun_eval())
        experiment_configs.insert(0, simple_task.exp_config)

    return ExperimentResults(
        exp_configs=experiment_configs,
        accs=all_results,
        artifact_store=artifact_store,
    )


def run_experiment(*args, **kwargs):
    return asyncio.run(arun_experiment(*args, **kwargs))
