import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class CSTR:
    def __init__(self, params, reaction):
        
        self.phase = params['phase']
        self.reaction = reaction
        self.feed = params['feed']
    def solve_ss(self):
        # solving for the mole balances of species involved in the reaction
        sol_guess = np.ones(self.reaction.n_species)
        sol = fsolve(self.mole_balance, sol_guess)
    # def solve_transient(self):
    def mole_balance(self,x):
        # x(1) = V
        # x(2) = C_A
        
        y = []
        for i in range(self.reaction.n_reactants):
            eqn = self.feed[i] - x + self.reaction.r[i]*
            y.append()

        return y

    def plot(self, sol):
        plt.plot(sol.t, sol.y.T)
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend(['A', 'B', 'C', 'D'])
        plt.show()
