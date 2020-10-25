# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:07:20 2020

@author: jakob
"""


import numpy as np
import random
import time
import pickle


start = time.time()


class OsloModel:
    """
    OsloModel class, that includes all the functions required to run the model
    Input: N = iterations, L = lattice size, p = probability
    
    """
    def __init__(self,L, N):
        """ Initialsier """
        self.L = L
        self.N = N
        self.z = np.zeros(self.L)
        self.height = 0
        self.tm = np.zeros(N)
        self.grains = 0 # Grains in the system
        self.tc = 0
        self.cross_over = 0

            
    def Drive(self):
        """Drives the system by adding a grain at i = 1 """
        # add grain to i = 0
        self.z[0] += 1
        self.height += 1
        self.grains += 1  
        
        
        
    def Relaxation(self, p):
        """
        Implementation of the Oslo model algorhitm, that checks if sites need
        relaxing after being driven. Only checks neighbouring sites that have
        tumbled. Keeps checking until checklist is empty.
        
        Input: p = probability of assigning slope to 1 or 2
        
        """
        
        checks = [0]
        while len(checks) != 0:
            for i in checks:
                self.count += 1
                if self.z[i] > self.z_th[i]:
                    if i == 0:
                        #print('true')
                        self.z[0] -= 2
                        self.z[1] += 1
                        self.height -= 1
                        checks.append(i+1)

                    elif i == self.L - 1:
                        self.z[i] -= 1
                        self.z[i-1] += 1
                        checks.append(i - 1)
                        checks.append(i)
                        if self.cross_over == 0:  # Saves the cross-over time
                            self.tc = self.grains - 1
                        self.cross_over += 1
                        
                        
                    else:
                        self.z[i] -= 2
                        self.z[i+1] += 1
                        self.z[i-1] += 1
                        checks.append(i+1)
                        checks.append(i-1)
                    
                    # new threshold slope for relaxed sites
                    if p > random.random():
                        self.z_th[i] = 1
                    else:
                        self.z_th[i] = 2
                        
                    self.s += 1
                    checks.remove(i)
                else:    
                    checks.remove(i)
                    
    
    def Iteration(self, N, p):
        """
        Function that iterates through the Oslo model, initilises the system,
        drives it, relaxes it and drives it for N iterations.
        Input: N = iterations, p = probability
        """
        # initialise threshold slopes randonmly
        self.z_th = np.random.choice([1,2], self.L, p=[p, 1-p])
        self.heights = np.zeros(N)
        self.s_arr = np.zeros(N)  # Avalanche at every time step
        self.count = 0  # Nymber of checks

        for t in range(N):
            self.s = 0
            self.Drive()
            self.Relaxation(p)  # return first and last
            self.heights[t] = self.height
            self.tm[t] = t
            self.s_arr[t] = self.s

            
    def save(self):
        """Saves the iterated model, using pickle """
        filename = 'Oslo_model_'+str(self.N)+'N_'+str(self.L)+'L.obj'
        fileh = open(filename, 'wb')
        pickle.dump(self, fileh, -1)  # -1 sets data stream to highest protocol  
        
    
    def load(self):
        """Loads the iterated model, using pickle """
        filename = 'Oslo_model_'+str(self.N)+'N_'+str(self.L)+'L.obj'
        fileh = open(filename, 'rb') 
        model = pickle.load(fileh)
        return model


def run(N, L, p):
    """
    Main function, that runs the system and prints data at the end.
    Input: N = iterations, L = lattice size, p = probability
    """
    model = OsloModel(L, N)
    model.Iteration(N, p)
    #model.save()
    #model = model.load()
    
    print("Average heights ", np.mean(model.heights))
    print('checks', model.count)
    print("Elapsed time: ", time.time() - start)

run(50000, 4, 0.5)

    

