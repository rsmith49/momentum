import abc

from .types import SearchFilterType


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

    @classmethod
    @abc.abstractmethod
    def get_good_examples_filter(cls, rating_field_name: str) -> SearchFilterType:
        """Return a filter that can be used to get good examples from vector queries."""
        raise NotImplementedError


THUMBS_UP_STR = "thumbs_up"
THUMBS_DOWN_STR = "thumbs_down"


class ThumbRating(Rating):
    """A simple thumbs up/down rating."""
    # NOTE: One design decision on calling this 'ThumbRating' instead of 'CorrectRating'
    #       is to distinguish between user provided (and sometimes inaccurate) feedback
    #       vs gold standard feedback that would be assumed based on 'CorrectRating'
    def __init__(self, is_thumbs_up: bool = True):
        self.is_thumbs_up = is_thumbs_up

    def to_vectorstore(self) -> str:
        if self.is_thumbs_up:
            return THUMBS_UP_STR
        else:
            return THUMBS_DOWN_STR

    def from_vectorstore(self, rating_str: str) -> "Rating":
        if rating_str == THUMBS_UP_STR:
            return ThumbRating(is_thumbs_up=True)
        elif rating_str == THUMBS_DOWN_STR:
            return ThumbRating(is_thumbs_up=False)
        else:
            raise ValueError(f"Unknown rating_str={rating_str}")

    @classmethod
    def get_good_examples_filter(cls, rating_field_name: str) -> SearchFilterType:
        return {rating_field_name: THUMBS_UP_STR}


class ThumbRatings:
    """A collection of ThumbRatings."""
    THUMBS_UP = ThumbRating(True)
    THUMBS_DOWN = ThumbRating(False)
