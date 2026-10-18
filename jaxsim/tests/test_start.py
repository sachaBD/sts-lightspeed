import numpy as np
import pytest

import jaxsim


def test_identical_cards_share_a_slot():
    s = jaxsim.make_start(["STRIKE_RED"] * 3 + [("STRIKE_RED", 1), "BASH"], hp=50)
    used = np.asarray(s.count) > 0
    assert used.sum() == 3 and np.asarray(s.count).sum() == 5
    assert s.card.shape == (jaxsim.DEFAULT_SPEC.D,)


@pytest.mark.parametrize("kw", [dict(deck=["NOT_A_CARD"], hp=10), dict(deck=["BASH"], hp=0),
                                dict(deck=["BASH"], hp=10, max_hp=5), dict(deck=["BASH"], hp=10, relics=["X"])])
def test_bad_input_raises(kw):
    with pytest.raises(ValueError):
        jaxsim.make_start(**kw)


def test_too_many_distinct_cards():
    spec = jaxsim.EnvSpec(D=2)
    with pytest.raises(ValueError):
        jaxsim.make_start(["BASH", "STRIKE_RED", "DEFEND_RED"], hp=10, spec=spec)
