import jax
import pytest

import jaxsim

IRONCLAD_STARTER = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH"]


@pytest.fixture(scope="session")
def env():
    return jaxsim.make_env()


@pytest.fixture(scope="session")
def start():
    return jaxsim.make_start(IRONCLAD_STARTER, hp=80, relics=["BURNING_BLOOD"])


@pytest.fixture
def key():
    return jax.random.key(0)
