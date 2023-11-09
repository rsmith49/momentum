import abc


class Rating(metaclass=abc.ABCMeta):
    """Abstract 'Rating' for an LLM response.

    Can be from a user, automated system, or anything else. The rating is
    used to improve the LLM's performance.
    """
    @abc.abstractmethod
    def to_vectorstore(self) -> str:
        """Convert the rating to a string representation for vectorstore."""
        raise NotImplementedError

    @abc.abstractmethod
    def from_vectorstore(self, rating_str: str) -> "Rating":
        """Convert a string representation from vectorstore to a rating object."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_filter(self) -> str:
        """Convert the rating to a string representation for filtering."""
        # TODO
        raise NotImplementedError
