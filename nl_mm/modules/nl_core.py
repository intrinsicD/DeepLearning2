"""Core abstractions for the Nested Learning hierarchy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

import torch


@dataclass
class LevelSpec:
    """Specification for a parameter level with its own update cadence."""

    name: str
    chunk_size: int
    lr: float
    optimizer: str
    params: List[torch.nn.Parameter] = field(default_factory=list)


class LevelState:
    """Runtime state maintained by :class:`NLScheduler`."""

    def __init__(self, spec: LevelSpec, optimizer: torch.optim.Optimizer):
        self.spec = spec
        self.optimizer = optimizer
        self.step = 0

    def maybe_step(self, global_step: int) -> bool:
        if (global_step + 1) % self.spec.chunk_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return True
        return False


class NLScheduler:
    """Coordinates optimizer steps for the hierarchical levels.

    The scheduler simply routes ``optimizer.step`` calls according to the
    ``chunk_size`` specified by each :class:`LevelSpec`.  Gradients can be
    accumulated between steps which mirrors Eq. (30–31) where the slow levels
    update less frequently.
    """

    def __init__(self, level_states: Iterable[LevelState]):
        level_list = list(level_states)
        self._level_states: Dict[str, LevelState] = {state.spec.name: state for state in level_list}
        if not self._level_states:
            raise ValueError("NLScheduler requires at least one level state")
        if len(self._level_states) != len(level_list):
            raise ValueError("Duplicate level names detected")

    def state_dict(self) -> Dict[str, Dict[str, int]]:
        return {name: {"step": state.step} for name, state in self._level_states.items()}

    def load_state_dict(self, state_dict: Dict[str, Dict[str, int]]) -> None:
        for name, payload in state_dict.items():
            if name not in self._level_states:
                continue
            self._level_states[name].step = payload.get("step", 0)

    def maybe_step(self, level_name: str, global_step: int) -> bool:
        if level_name not in self._level_states:
            raise KeyError(f"Unknown level {level_name}")
        return self._level_states[level_name].maybe_step(global_step)

    def step_all(self, global_step: int) -> Dict[str, bool]:
        return {name: state.maybe_step(global_step) for name, state in self._level_states.items()}

    @property
    def levels(self) -> List[str]:
        return list(self._level_states.keys())


def build_level_states(level_specs: Iterable[LevelSpec], optimizers: Dict[str, torch.optim.Optimizer]) -> List[LevelState]:
    states: List[LevelState] = []
    for spec in level_specs:
        if spec.optimizer not in optimizers:
            raise KeyError(f"Optimizer {spec.optimizer} not provided for level {spec.name}")
        opt = optimizers[spec.optimizer]
        states.append(LevelState(spec, opt))
    return states
