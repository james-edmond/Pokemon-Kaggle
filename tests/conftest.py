import random
import pytest


@pytest.fixture(scope="session")
def sample_deck():
    from ptcg.engine import load_sample_deck
    return load_sample_deck()


@pytest.fixture()
def rng():
    return random.Random(7)
