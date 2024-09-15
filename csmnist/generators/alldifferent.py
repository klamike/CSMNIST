import torch

from csmnist.constraints import AllDifferent, LengthConstraint
from csmnist.generators.generator import Generator


class AllDifferentGenerator(Generator):
    def __init__(self, length=5, seed=None, **kwargs):
        super().__init__([AllDifferent(), LengthConstraint(length)], seed, **kwargs)

        assert length <= 10, \
            "AllDifferent length must be less than or equal to 10 " \
            "since we only have 10 unique digits."

        self.length = length

    def _generate(self):
        return torch.randperm(
            10,
            generator=self.rng
        )[:self.length]
