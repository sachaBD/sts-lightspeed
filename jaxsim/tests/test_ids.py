import subprocess
import sys
from pathlib import Path

import pytest

import jaxsim

ROOT = Path(__file__).resolve().parents[1]


def test_ids_in_sync_with_cpp_headers():
    if not (ROOT.parent / "include" / "constants" / "Cards.h").exists():
        pytest.skip("C++ headers not available")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_ids.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("names", [jaxsim.CARD_NAMES, jaxsim.RELIC_NAMES, jaxsim.POTION_NAMES, jaxsim.MOVE_NAMES,
                                   jaxsim.PLAYER_STATUS_NAMES, jaxsim.MONSTER_STATUS_NAMES])
def test_vocab_shape(names):
    assert names[0] == "NONE"
    assert len(set(names)) == len(names)
    assert "INVALID" not in names and "EMPTY_POTION_SLOT" not in names


def test_known_ids():
    for name in ("STRIKE_RED", "BASH", "WOUND"):
        assert name in jaxsim.CARD_NAMES
    assert "THE_CHAMP_EXECUTE" in jaxsim.MOVE_NAMES
