# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:00:02 2020

@author: jakob
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


class RWmodel:
    
    def __init__(self, N, m, q=0.5, seed=None):
        
        # Initialise user defined variables
        self.N = N  # Number of vertices
        self.m = m  # Number of edges to add at each step
        self.q = q
        self.seed = seed  # Set seed of random numbers
        if seed is None:
            random.seed()
        else:
            random.seed(seed)
        
    def iterate(self, n):
        
        v1 = np.zeros(self.m)
        self.G.add_node(n)
        
        for m in range(self.m):
            end = False
            v1[m] = int(random.choice((self.nodes)))
            
            while end == False:
                if random.random() < self.q:
                    v1[m] = int(random.choice(list(self.G.neighbors(v1[m]))))
                else:
                    if v1[m] not in v1[:m]: 
                        self.G.add_edge(n, v1[m])
                        end = True
                    else:
                        v1[m] = int(random.choice((self.nodes)))
        self.nodes.append(n) 


    def run(self, plot=False):
        
        # Initialise
        self.G = nx.complete_graph(self.m)
        self.edges = list(self.G.edges)
        self.nodes = list(self.G.nodes)
        
        
        for n in range(self.m, self.N):
            self.iterate(n)

        self.edges = list(self.G.edges)        

        if plot == True:
            plt.figure()
            nx.draw(self.G, with_labels = True) 
            plt.show()
             
# network = RWmodel(7, 3, q=0)
# network.run(plot=True)
    