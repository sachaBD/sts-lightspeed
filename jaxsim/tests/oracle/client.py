"""Client for the C++ jaxsim_oracle process (JSON lines). Build it with tools/build_oracle.sh."""
import json
import subprocess
from pathlib import Path

BINARY = Path(__file__).resolve().parents[2] / "oracle" / "build" / "jaxsim_oracle"


class Oracle:
    def __init__(self, binary: Path = BINARY):
        self.p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def _req(self, r: dict) -> dict:
        self.p.stdin.write(json.dumps(r) + "\n")
        self.p.stdin.flush()
        out = json.loads(self.p.stdout.readline())
        if "error" in out:
            raise RuntimeError(f"oracle: {out['error']} for {r}")
        return out

    def start(self, seed, hp, max_hp, deck, relics=(), potions=(), potion_capacity=2):
        return self._req(dict(cmd="start", seed=seed, hp=hp, max_hp=max_hp, deck=[list(c) for c in deck],
                              relics=list(relics), potions=list(potions), potion_capacity=potion_capacity))

    def step(self, bits):
        return self._req(dict(cmd="step", bits=bits))

    def cards(self):
        return self._req(dict(cmd="cards"))["cards"]

    def close(self):
        self.p.stdin.close()
        self.p.wait()
