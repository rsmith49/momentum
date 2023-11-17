"""Tests the overall usage of the package with simulated user paths."""
from momentum.rating import ThumbRatings


def test_basic_case():
    """Check that the agent can respond, and incorporate feedback to surface that response as context"""
    query = "What is the weather in Santa Clara on Friday?"
    agent = init_agent()

    response1 = agent.execute(query)
    assert response1.content == "The weather in Santa Clara on Friday is sunny."

    agent.add_rating(response1.id, thumb=ThumbRatings.THUMBS_UP)
    response2 = agent.execute(query)
    assert response2.content == "The weather in Santa Clara on Friday is sunny."
    assert response2.examples == [Example.from_response(response1)]
