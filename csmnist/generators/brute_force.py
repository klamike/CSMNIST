import torch

from csmnist.generators.generator import Generator


class BruteForceGenerator(Generator):
    def __init__(self, constraints=None, min_length=1, max_length=10, max_tries_per_sample=1000, seed=None):
        super().__init__(constraints, seed)

        assert min_length <= max_length, \
            "min_length must be less than or equal to max_length"

        assert min_length > 0, \
            "min_length must be greater than 0"

        assert max_tries_per_sample > 0, \
            "max_tries_per_sample must be greater than 0"

        self.min_length = min_length
        self.max_length = max_length
        self.max_tries_per_sample = max_tries_per_sample

    def generate(self):
        counter = 0
        while counter < self.max_tries_per_sample:
            counter += 1

            length = torch.randint(
                self.min_length,
                self.max_length + 1,
                (1,),
                generator=self.rng
            )
            
            sequence = torch.randint(
                0,
                10,
                (length,),
                generator=self.rng
            )

            if all(c.satisfy(sequence) for c in self.constraints):
                return sequence