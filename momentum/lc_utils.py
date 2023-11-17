"""This module defines LangChain modified classes required by momentum needs.

NOTE: This module could be abstracted out if certain changes are made to LangChain
"""
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.vectorstores import Cassandra
from pydantic import BaseModel

from .types import SearchFilterType

RATING_FIELD_NAME = "rating"
EXAMPLE_CONTENT_FIELD_NAME = "example_content"
DEFAULT_EXAMPLES_FIELD = "examples"


class MomentumCassandra(Cassandra):
    """Need to override the init method to allow for customizable search fields"""
    def __init__(
        self,
        embedding: Embeddings,
        keyspace: str,
        table_name: str,
        ttl_seconds: Optional[int] = None,
        **vt_init_kwargs,
    ) -> None:
        try:
            from cassio.vector import VectorTable
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )

        self.embedding = embedding
        self.keyspace = keyspace
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self._embedding_dimension = None

        self.table = VectorTable(
            session=None,
            keyspace=keyspace,
            table=table_name,
            embedding_dimension=self._get_embedding_dimension(),
            primary_key_type="TEXT",
            # Setting the content field to not be searchable to avoid
            # performance hit from SAI on huge text fields
            # TODO: Make the content field name configurable
            metadata_indexing=("denylist", EXAMPLE_CONTENT_FIELD_NAME),
            **vt_init_kwargs,
        )


class MomentumExampleFinder(Runnable[Dict, Dict], BaseModel):
    """
    Select the examples that should be used for a few-shot prompt via MMR search,
    and pass to the next link in the chain along with the original input.
    """
    input_prompt_template: PromptTemplate
    vectorstore: MomentumCassandra
    k: int = 4
    fetch_k: int = 20
    examples_field: str = DEFAULT_EXAMPLES_FIELD
    rating_filter: SearchFilterType

    def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform the search of the vectorstore

        NOTE: Can be subclassed to implement other search methods
        """
        return self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=self.k,
            fetch_k=self.fetch_k,
            filter=self.rating_filter,
        )

    def select_examples(self, prompt_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the vectorstore with the prompt formatted with empty examples"""
        vs_query = self.input_prompt_template.format(
            **{self.examples_field: ""},
            **prompt_input,
        )
        return self._perform_search(vs_query)

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        return {
            self.examples_field: self.select_examples(input),
            **input,
        }


class KeywordExampleSelector(BaseExampleSelector, BaseModel):
    """Simply selects the examples from the key of the input dict"""
    examples_key: str = DEFAULT_EXAMPLES_FIELD

    def select_examples(self, input_variables: Dict[str, Any]) -> List[dict]:
        return input_variables[self.examples_key]

    def add_example(self, example: Dict[str, str]) -> Any:
        # Because of our semi-hacky workaround, this does not work
        raise NotImplementedError
