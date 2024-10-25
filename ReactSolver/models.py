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

    def mole_balance_ss(self,x):
        # x[0:n_reactants] = F[reactant]
        # x[-1] = V
        Ft = np.sum(x[:-1])
        y = []
        C = np.zeros(self.reaction.n_reactants)
        P = self.calc_pressure(T, Ft)
        T = self.energy_balance(x)
        ri = np.zeros(self.reaction.n_reactants)
        # multiple reaction shows itself in the rate 
        for i in range(self.reaction.n_reactants):
            vi = self.calc_vol_rate(T, P, Ft)
            C[i] = x[i]/self.calc_vol_rate(T, P, Ft)
            ri[i] = vi*self.reaction.calc_rate(i,C[i],T)
            eqn = self.feed[i] - x[i] + ri[i]*x[-1]
            y.append(eqn)

        return y
    
    def energy_balance(self,x):
        return self.T0
    
    def calc_vol_rate(self, T, P, Ft):
        # F is the flowrate
        # T is the temperature
        # P is the pressure
        if self.phase == 'liquid':
            self.nu = self.nu0
        elif self.phase == 'gas':
            self.nu = self.nu0 * (T/self.T0) * (self.P0/P)
        return self.nu
    
    def calc_pressure(self, T, Ft):
        return self.P0

    def plot(self, sol):
        plt.plot(sol.t, sol.y.T)
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend(['A', 'B', 'C', 'D'])
        plt.show()
