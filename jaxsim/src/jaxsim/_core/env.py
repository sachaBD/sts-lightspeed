"""Binds the private implementation to the public Env contract for one EnvSpec."""
from functools import partial

from jaxsim import api
from jaxsim._core import init, observe, step


def build(spec: api.EnvSpec) -> api.Env:
    return api.Env(
        spec=spec,
        init=partial(init.init, spec),
        step=partial(step.step, spec),
        legal_mask=partial(step.legal_mask, spec),
        observe=partial(observe.observe, spec),
        outcome=partial(observe.outcome, spec),
    )
