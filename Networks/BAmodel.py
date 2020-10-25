# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:00:02 2020

@author: jakob
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


class BAmodel:
    
    def __init__(self, N, m, seed=None, model='PA'):
        
        # Initialise user defined variables
        self.N = N  # Number of vertices
        self.m = m  # Number of edges to add at each step
        self.seed = seed  # Set seed of random numbers
        self.model = model
        if seed is None:
            random.seed()
        else:
            random.seed(seed)
        
    def iterate(self, n):
        
        def randsel():
            if self.model == 'PA':
                v2 = random.choice(random.choice(self.edges))
            if self.model == 'random':
                v2 = random.randint(0, n-1)
            return v2
        
        v2 = np.zeros(self.m)
        self.G.add_node(n)  
        
        for m in range(self.m):
            unique = False
            while unique == False:
                v2[m] = randsel()
                if v2[m] not in v2[:m]:
                    unique = True

            self.G.add_edge(n, v2[m])
        
        if self.model == 'PA':
            for m in range(self.m):
                self.edges.append((n, int(v2[m])))  # Add afterwards to avoid self-loops


    def run(self, plot=False):
        
        # Initialise
        self.G = nx.complete_graph(self.m)
        self.edges = list(self.G.edges)
        
        for n in range(self.m, self.N):
            self.iterate(n)
        
        if self.model == 'random':
            self.edges = list(self.G.edges)

        if plot == True:
            plt.figure()
            nx.draw(self.G, with_labels = True) 
            plt.show()
             
# network = BAmodel(10, 3, model='random')
# network.run(plot=True)
    