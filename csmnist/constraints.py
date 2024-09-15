from __future__ import annotations
from abc import ABC, abstractmethod

from numpy import ndarray
from torch import Tensor


class Constraint(ABC):
    def satisfy(self, sequence: list[int] | Tensor | ndarray):
        if isinstance(sequence, (Tensor, ndarray)):
            assert sequence.ndim == 1, "Sequence must be a 1D Tensor"
            # NOTE: Perhaps we can support 2d sequences...
            sequence = sequence.tolist()

        assert isinstance(sequence, list), "Sequence must be a list or a Tensor"

        return self._satisfy(sequence)

    @abstractmethod
    def _satisfy(self, sequence: list[int]):
        pass

class AllDifferent(Constraint):
    def _satisfy(self, sequence: list[int]):
        return len(set(sequence)) == len(sequence)

class LengthConstraint(Constraint):
    def __init__(self, length: int):
        self.length = length

    def _satisfy(self, sequence: list[int]):
        return len(sequence) == self.length

class SumConstraint(Constraint):
    def __init__(self, target_sum: int):
        self.target_sum = target_sum

    def _satisfy(self, sequence: list[int]):
        return sum(sequence) == self.target_sum

class OrderedConstraint(Constraint):
    def __init__(self, ascending: bool=True):
        self.ascending = ascending

    def _satisfy(self, sequence: list[int]):
        L = range(len(sequence)-1)
        return all(sequence[i] < sequence[i+1] for i in L) if self.ascending else \
               all(sequence[i] > sequence[i+1] for i in L)

class EvenOddAlternatingConstraint(Constraint):
    def _satisfy(self, sequence: list[int]):
        return all((d % 2 == i % 2) for i, d in enumerate(sequence))

class NoConsecutiveConstraint(Constraint):
    def _satisfy(self, sequence: list[int]):
        return all(sequence[i] != sequence[i+1] for i in range(len(sequence)-1))

class DigitOccurrenceConstraint(Constraint):
    def __init__(self, digit: int, occurrences: int):
        self.digit = digit
        self.occurrences = occurrences

    def _satisfy(self, sequence: list[int]):
        return sequence.count(self.digit) == self.occurrences

class PalindromeConstraint(Constraint):
    def _satisfy(self, sequence: list[int]):
        return sequence == sequence[::-1]