import torch
from abc import ABC, abstractmethod

from csmnist.constraints import Constraint


class Generator(ABC):
    def __init__(
        self,
        constraints=None,
        seed=None
    ):
        self.constraints = constraints or []
        self.rng = torch.Generator()

        if seed is not None:
            self.rng.manual_seed(seed)

        assert all(isinstance(c, Constraint) for c in self.constraints), \
            "All constraints must be instances of Constraint class"

    @abstractmethod
    def generate(self):
        pass

    def generate_dataset(self, size: int):
        return [self.generate() for _ in range(size)]

