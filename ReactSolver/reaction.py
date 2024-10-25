import numpy as np

class SingleReaction:
    def __init__(self,params):
        self.n_reactions = params['n_reactions']
        self.n_species = params['n_species']
        self.stoich_matrix = params['stoich_matrix']
        self.order = params['order']
        self.k = params['k']
        self.reversible = params['reversible']
        self.Ea = params['Ea']
        self.A = params['A']
        

class MultiReaction:
    def __init__(self,):
        self.reactions = []