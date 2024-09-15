import torch
from abc import ABC, abstractmethod

from csmnist.constraints import Constraint


class Generator(ABC):
    def __init__(
        self,
        constraints=None,
        seed=None,
        error_on_duplicate=True
    ):
        self.constraints = constraints or []
        self.rng = torch.Generator()
        self.seen_sequences = set()
        self.error_on_duplicate = error_on_duplicate

        if seed is not None:
            self.rng.manual_seed(seed)

        assert all(isinstance(c, Constraint) for c in self.constraints), \
            "All constraints must be instances of Constraint class"

    @abstractmethod
    def _generate(self):
        pass

    def generate(self):
        sequence = self._generate()

        if not all(c.satisfy(sequence) for c in self.constraints):
            raise RuntimeError("Generated sequence does not satisfy the constraints.")

        if self.error_on_duplicate and self.is_duplicate(sequence):
            raise RuntimeError("Generated sequence has already been seen.")

        self.seen_sequences.add(tuple(sequence))

        return sequence

    def is_duplicate(self, sequence):
        return tuple(sequence) in self.seen_sequences
    
    def reset_epoch(self):
        self.seen_sequences = set()

    def generate_dataset(self, size: int):
        return [self.generate() for _ in range(size)]

