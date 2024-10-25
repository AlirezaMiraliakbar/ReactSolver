import numpy as np

class SingleReaction:
    def __init__(self,params):
        self.n_reactants = params['n_reactants']
        self.n_products = params['n_products']
        self.n_species = self.n_products + self.n_reactants
        self.stoich_matrix = params['stoich_matrix']
        self.order = params['order']
        self.k = params['k']
        self.reversible = params['reversible']
        self.Ea = params['Ea']
        self.A = params['A']
    # the reaction rate needs to be upadated coupled with the reactor model
        self.rate = np.zeros([self.n_reactants,1])
    def calc_rate(self, index,C):
        # C is the concentration of the reactants
        self.rate[index] = self.stoich_matrix[index] * self.k * (C**self.order)
        return self.rate

class MultiReaction:
    def __init__(self,):
        self.reactions = []