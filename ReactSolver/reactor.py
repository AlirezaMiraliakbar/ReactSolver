import numpy as np
from scipy.integrate import solve_ivp

class ReactorSolver:
    def __init__(self, params):
        self.params = params

    def solve_odes(self, odes, t_span, y0):
        sol = solve_ivp(odes, t_span, y0, args=(self.params,))
        return sol
