import os
from pathlib import Path

import jax
import pytest

# compiled programs are cached on disk, so reruns of unchanged code skip compilation
jax.config.update("jax_compilation_cache_dir", str(Path(__file__).resolve().parents[1] / ".jax-cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

import jaxsim

IRONCLAD_STARTER = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH", "ASCENDERS_BANE"]
_REPO = Path(__file__).resolve().parents[2]


def _oracle_problem() -> str | None:
    """Why the oracle can't be trusted (missing, or older than the C++ it is built from), else None."""
    from tests.oracle.client import BINARY
    if not BINARY.exists():
        return f"oracle not built: run tools/build_oracle.sh ({BINARY})"
    built = BINARY.stat().st_mtime
    sources = [p for d in ("src", "include") for p in (_REPO / d).rglob("*") if p.suffix in (".cpp", ".h")]
    sources.append(_REPO / "jaxsim" / "oracle" / "oracle.cpp")
    stale = [p for p in sources if p.stat().st_mtime > built]
    if stale:
        return f"oracle older than {stale[0].relative_to(_REPO)} (+{len(stale) - 1}): rerun tools/build_oracle.sh"
    return None


def pytest_collection_modifyitems(config, items):
    """Tests that use the C++ oracle fail (not skip) when it is missing or stale. JAXSIM_NO_ORACLE=1 skips them."""
    users = [it for it in items if "oracle" in getattr(it, "fixturenames", ())]
    problem = _oracle_problem() if users else None
    if problem is None:
        return
    mark = (pytest.mark.skip(reason=problem) if os.environ.get("JAXSIM_NO_ORACLE")
            else pytest.mark.usefixtures("_oracle_broken"))
    for it in users:
        it.add_marker(mark)


@pytest.fixture
def _oracle_broken():
    pytest.fail(_oracle_problem())


@pytest.fixture(scope="session")
def oracle():
    from tests.oracle.client import Oracle
    o = Oracle()
    yield o
    o.close()


@pytest.fixture(scope="session")
def env():
    return jaxsim.make_env()


@pytest.fixture(scope="session")
def start():
    return jaxsim.make_start(IRONCLAD_STARTER, hp=80, relics=["BURNING_BLOOD"])


@pytest.fixture
def key():
    return jax.random.key(0)
