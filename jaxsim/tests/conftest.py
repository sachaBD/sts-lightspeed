from pathlib import Path

import jax
import pytest

# compiled programs are cached on disk, so reruns of unchanged code skip compilation
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).resolve().parents[1] / ".jax-cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

import jaxsim

IRONCLAD_STARTER = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH", "ASCENDERS_BANE"]


@pytest.fixture(scope="session")
def env():
    return jaxsim.make_env()


@pytest.fixture(scope="session")
def start():
    return jaxsim.make_start(IRONCLAD_STARTER, hp=80, relics=["BURNING_BLOOD"])


@pytest.fixture
def key():
    return jax.random.key(0)
