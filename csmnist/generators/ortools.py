import torch
from ortools.constraint_solver.pywrapcp import Solver

from csmnist.generators.generator import Generator


class ORToolsGenerator(Generator):
    """ Use ORTools to get new sequences. The solver should already contain
        all the constraints and the search phase should be started.
        For example, to encode N-queens:
        ```python
        solver = Solver("n-queens")
        B = range(board_size)
        queens = [solver.IntVar(0, board_size - 1, f"x{i}") for i in B]
        solver.Add(solver.AllDifferent(queens))
        solver.Add(solver.AllDifferent([queens[i] + i for i in B]))
        solver.Add(solver.AllDifferent([queens[i] - i for i in B]))
        db = solver.Phase(queens, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
        solver.NewSearch(db)

        generator = ORToolsGenerator(solver, queens, seed=42)
        ```
    """
    def __init__(self, solver: Solver, vars, seed=None, **kwargs):
        super().__init__(None, seed, **kwargs)

        self.solver = solver
        self.vars = vars


    def _generate(self):
        has_next = self.solver.NextSolution()
        if not has_next:
            self.solver.EndSearch()
            raise ValueError("No more solutions.")

        return torch.tensor([v.Value() for v in self.vars])