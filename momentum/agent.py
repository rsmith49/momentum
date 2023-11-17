"""An interface to the agent. Currently only supports LangChain specified LLMs"""
# TODO: Figure out if we want to make this actually just a part of Langchain so people
#       can drag and drop much easier - discoverability is the biggest thing we care about
from typing import Any, Dict, Optional, Type, Union
from uuid import uuid4

from langchain.embeddings.base import Embeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.language_model import LanguageModelInput

from .lc_utils import RATING_FIELD_NAME, KeywordExampleSelector, MomentumCassandra, MomentumExampleFinder
from .rating import Rating, ThumbRating
from .response_example import InContextExample, ResponseExample

# TODO: Infer the example format from the input prompt template
DEFAULT_EXAMPLE_TEMPLATE = PromptTemplate(
    input_variables=["input", "output"],
    template="""Input: {input}\nOutput: {output}""",
)
DEFAULT_VECTORSTORE_CLS = MomentumCassandra
DEFAULT_VS_KEYSPACE = "momentum"
DEFAULT_RATING_CLS = ThumbRating

LLM_RESPONSE_KEY = "llm_response"
PROMPT_DICT_KEY = "prompt_dict"


class AdaptiveAgent:
    """LLM Agent that learns based on user feedback

    Parameters
    ----------
    llm_runnable : Runnable
        The LLM to be used by the agent.
        Should be a LangChain Runnable with InputType of LanguageModelInput and
        OutputType of str.
    prompt_template : Union[PromptTemplate, str]
        Template for the prompt to be sent to the LLM, which must include an input
        for `examples`.
    example_template : Union[PromptTemplate, str] | None, optional
        Template for the few-shot prompt, by default None.
        If None, will use the default template:
        ```
        Question: {input}\nAnswer: {output}
        ```
    vectorstore : VectorStore | None, optional
        VectorStore to use for the agent, by default will use the Cassandra VectorStore.
        NOTE: If None, requires embeddings and table_name to be provided.
    embeddings : Embeddings | None, optional
        Embeddings to use for the agent, by default None.
        NOTE: If None, requires vectorstore to be provided.
    table_name : str | None, optional
        Table name to use for the agent, by default None.
        NOTE: If None, requires vectorstore and embeddings to be provided.
    num_few_shot_examples : int, optional
        Number of few-shot examples to use, by default 4.
    num_examples_to_consider : int, optional
        Number of examples to return from vector store when determining best examples
        via Maximum Marginal Relevance (MMR), by default 20.
    """
    def __init__(
        self,
        llm_runnable: Runnable[LanguageModelInput, str],
        prompt_template: Union[PromptTemplate, str],
        example_template: Optional[Union[PromptTemplate, str]] = None,
        vectorstore: Optional[MomentumCassandra] = None,
        embeddings: Optional[Embeddings] = None,
        table_name: Optional[str] = None,
        num_few_shot_examples: int = 4,
        num_examples_to_consider: int = 20,
        rating_cls: Type[Rating] = DEFAULT_RATING_CLS,
        # TODO: Add param to embed query + response instead of just query
    ):
        # TODO: Logic to account for the following scenarios
        # - Runnable with a prompt template as the first step
        # - Runnable without a prompt template
        # - something else?
        self.llm_runnable = llm_runnable

        if isinstance(prompt_template, str):
            prompt_template = PromptTemplate.from_template(prompt_template)
        self._base_prompt_template = prompt_template

        if example_template is None:
            example_template = DEFAULT_EXAMPLE_TEMPLATE
        elif isinstance(example_template, str):
            example_template = PromptTemplate.from_template(example_template)
        self.example_template = example_template

        if vectorstore is None:
            if table_name is None or embeddings is None:
                raise ValueError(
                    "If vectorstore is None, table_name and embeddings must be provided"
                )
            vectorstore = DEFAULT_VECTORSTORE_CLS(
                embedding=embeddings,
                keyspace=DEFAULT_VS_KEYSPACE,
                table_name=table_name,
            )
        self.vectorstore = vectorstore

        self.num_few_shot_examples = num_few_shot_examples
        self.num_examples_to_consider = num_examples_to_consider
        self.rating_cls = rating_cls

        self.example_finder = MomentumExampleFinder(
            input_prompt_template=self._base_prompt_template,
            vectorstore=self.vectorstore,
            k=self.num_few_shot_examples,
            fetch_k=self.num_examples_to_consider,
            rating_filter=rating_cls.get_good_examples_filter(RATING_FIELD_NAME),
        )
        self.few_shot_template = self._create_few_shot_template()
        # Create main Langchain Runnable for the agent via LCEL
        self.pipeline = self.example_finder | {
            PROMPT_DICT_KEY: RunnablePassthrough(),
            LLM_RESPONSE_KEY: self.few_shot_template | self.llm_runnable,
        }

    def _create_few_shot_template(self) -> FewShotPromptTemplate:
        """Helper method to hackily convert a PromptTemplate to a FewShotPromptTemplate"""
        if "examples" not in self._base_prompt_template.input_variables:
            raise ValueError(
                "Base prompt template must accept `examples` as an input variable"
            )

        prompt_template_str = self._base_prompt_template.template
        split_on_str = "{examples}" if self._base_prompt_template.template_format == "f-string" else "{{ examples }}"
        prompt_sections = prompt_template_str.split(split_on_str)
        if len(prompt_sections) != 2:
            raise ValueError(
                "Currently can only support a single {examples} input in the prompt template"
            )
        prefix, suffix = prompt_sections

        return FewShotPromptTemplate(
            input_variables=[
                var for var in self._base_prompt_template.input_variables
                if var != "examples"
            ],
            template_format=self._base_prompt_template.template_format,
            prefix=prefix,
            suffix=suffix,
            example_selector=KeywordExampleSelector(),
            example_prompt=self.example_template,
        )

    def _is_using_chat_model(self) -> bool:
        """Check if the LLM is using a chat model"""
        # TODO: Check back in if this matters - looks like it currently doesn't?
        # Langchain OpenAIChat just sends messages converted to text in a single HumanMessage best I can tell
        return False

    def _add_example_to_vs(self, response_example: ResponseExample) -> None:
        """Add an example to the vector store"""
        self.vectorstore.add_texts(
            ids=[response_example.id],
            texts=[response_example.prompt],
            metadatas=[{
                "example_content": response_example.content,
                "in_context_examples": [ex.id for ex in response_example.in_context_examples],
                # TODO: Allow additional kwargs for metadata (figure out entrypoint for this)
            }]
        )

    def execute(self, query: Union[str, Dict[str, Any]]) -> ResponseExample:
        """Execute a query against the agent

        Parameters
        ----------
        query : str
            The query to execute. Can take the form of a string if the prompt template
            only expects a single input, otherwise should be a dictionary with keys
            corresponding to the input variables of the prompt template.
        """
        if isinstance(query, str):
            if len(self.few_shot_template.input_variables) != 1:
                raise ValueError(
                    "If query is a string, the prompt template must only expect a single input"
                )
            query = {self.few_shot_template.input_variables[0]: query}

        # TODO: Check how to get any internal traces
        pipeline_output = self.pipeline.invoke(query)
        response_text = pipeline_output[LLM_RESPONSE_KEY]
        prompt = self._base_prompt_template.format(examples="", **query)

        response_example = ResponseExample(
            id=uuid4().hex,
            prompt=prompt,
            content=response_text,
            # TODO: Consider whether all the hacks are worth it and we actually need the examples
            in_context_examples=[
                InContextExample(**ex_kwargs)
                for ex_kwargs in pipeline_output[PROMPT_DICT_KEY][self.example_finder.examples_field]
            ],
        )

        self._add_example_to_vs(response_example)

        return response_example

    def add_rating(self, example_id: str, rating: Rating) -> None:
        """Add a rating to an example

        TODO: Add source of rating
        Parameters
        ----------
        example_id : str
            ID of the example to add a rating to
        rating : Rating
            What to rate the example
        """
        # TODO: Make this less hacky by accessing the internal internal table
        self.vectorstore.table.table.put(
            row_id=example_id,
            metadata={
                RATING_FIELD_NAME: rating.to_vectorstore(),
            }
        )
