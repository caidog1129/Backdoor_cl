class Solution:
    def __init__(self, solution=None, objective=None):
        self.solution = solution
        self.objective = objective

    def __lt__(self, other):
        return self.objective < other.objective