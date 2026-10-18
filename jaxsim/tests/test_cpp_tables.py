"""Consistency checks on C++ data tables jaxsim depends on (skipped without the C++ headers)."""
import re
from pathlib import Path

import pytest

HEADERS = Path(__file__).resolve().parents[2] / "include" / "constants"
pytestmark = pytest.mark.skipif(not (HEADERS / "Cards.h").exists(), reason="C++ headers not available")


def _list(text, pattern):
    body = re.search(pattern, text, re.S).group(1)
    return [x.strip().split("::")[-1].strip() for x in re.sub(r"//[^\n]*", "", body).split(",") if x.strip()]


def test_card_colors_match_card_pools():
    """Cards.h cardColors must agree with the per-colour pools in CardPools.h (RarityCardPool offsets 0/72/143/214
    are red/green/blue/purple; ColorlessRarityCardPool is colorless)."""
    cards, pools = (HEADERS / "Cards.h").read_text(), (HEADERS / "CardPools.h").read_text()
    names = [n.split("=")[0].strip() for n in _list(cards, r"enum class CardId[^{]*\{(.*?)\};")]
    colors = dict(zip(names, _list(cards, r"cardColors\[\] = \{(.*?)\};")))
    blob = _list(pools, r"namespace RarityCardPool \{.*?cardBlob\[\] \{(.*?)\};")
    truth = {c: ("RED", "GREEN", "BLUE", "PURPLE")[(i >= 72) + (i >= 143) + (i >= 214)] for i, c in enumerate(blob)}
    truth |= {c: "COLORLESS" for c in _list(pools, r"colorlessCardBlob\[\] \{(.*?)\};")}
    assert len(names) == len(colors)
    bad = {c: (colors[c], want) for c, want in truth.items() if colors[c] != want}
    assert not bad, f"cardColors disagrees with CardPools (have, want): {bad}"
    assert sum(v == "RED" for v in colors.values()) == 75
