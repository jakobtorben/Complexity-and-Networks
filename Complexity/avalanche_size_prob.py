# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:17:22 2020

@author: jakob
"""


import logbin230119 as logbin
import Oslo_model as om
from Oslo_model import OsloModel  # Necessary for pickling class
import numpy as np
import matplotlib.pyplot as plt


def P_s(s, sdata, tot):
    """
    Finds avalanche probability by taking ratio of No. observed 
    configureations with avalanche s, to total configurations.
    
    Inputs: s = avalanche size (int)
            sdata = array of the avalanche sizes,
            tot = total number of avalanches
            
    Returns: P(s;L)
    """
    return np.sum(sdata == s)/tot   


def avalanche(L, N, tc, comparison=False, zeros=False):
    """
    Performs the data-binning on the steady state data that has been pre calculated.
    
    Input: runs = No. realisations to average over
            N = iterations, L = system size, comparison = compares data-binning,
            zeros = includes s = 0 if True
            
    Returns: Data-binned avalanche size s and frequency (P^tilda)
    """
    
    model = om.OsloModel(L, N).load()
    
    ind = np.where(model.tm==tc)[0][0]
    model.s_arr = np.array(model.s_arr[ind:], dtype=int)
    s, freq = logbin.logbin(model.s_arr, scale = 1.3, zeros=zeros)
    
    
    if comparison == True:
        fig, ax = plt.subplots(figsize=(8,4.5))
        maxs = int(max(model.s_arr))
        probs = [P_s(s, model.s_arr, N-tc) for s in range(maxs + 1)]
        ax.plot(np.arange(maxs + 1), probs, 'o', markersize=2,
                 label=(r'$Raw \, P_{N}(s;L)$'))
        ax.plot(s, freq, '-o', markersize=2.5, label=(r'$Data-binned \, \tilde{P}_{N}(s^{j};L)$'))
        ax.set_xlabel('s')
        ax.set_ylabel(r'${P}(s;512)$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        fig.tight_layout()

    return s, freq



def avalanche_scaling(N):
    """
    Plots the data binned avalanche size and peforms data collapse
    
    Input: N = number of iterations
    """
        
    L = [4, 8, 16, 32, 64, 128, 256, 512]
    tc = [14.7, 55.73, 215.1, 854.83, 3438.3, 13953.93, 56211.4, 225265.25]

        
    avg_array = [avalanche(L[i], N, int(tc[i])) for i in range(len(L))]
    ss, freqs = zip(*avg_array)  # splits array        
    
    fig1, ax1 = plt.subplots(figsize=(8,4.5))
    for i in range(len(freqs)):
        label = ('L = ' + str(int(L[i])))

        ax1.plot(ss[i], freqs[i], '-o', markersize=2.5, label=label)
        ax1.legend()
        ax1.set_xlabel('s')
        ax1.set_ylabel(r'$\tilde{P}(s;L)$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        fig1.tight_layout()
        
        
    D = 2.18
    ts = 1.56
    fig2, ax2 = plt.subplots(figsize=(8,4.5))
    for i in range(len(freqs)):
        label = ('L = ' + str(int(L[i])))
        
        ax2.plot(ss[i]/L[i]**D, freqs[i]*ss[i]**ts, '--o', markersize=2, label=label)
        ax2.legend()
        ax2.set_xlabel('$s/L^D$')
        ax2.set_ylabel(r'$s^{\tau_s} \tilde{P}(s;L)$')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        fig2.tight_layout()
        
        
def kmoment(N):
    """
    Finds the realationship between the kth moment and uses it to estimate
    D and tau from the gradients.
    
    
    Input: Number of iterations
    """

    L = [4, 8, 16, 32, 64, 128, 256, 512]
    tc = [14.7, 55.73, 215.1, 854.83, 3438.3, 13953.93, 56211.4, 225265.25]
    k_arr = [1, 2, 3, 4, 5]
    sk = np.zeros([len(L), len(k_arr)])
        
    avg_array = [avalanche(L[i], N, int(tc[i]), zeros=True) for i in range(len(L))]
    ss, freqs = zip(*avg_array)  # splits array        

    for k in range(len(k_arr)):  # k-moment 1,2,3..
        for Li in range(len(L)):  # s^k for every system size
            # Sum up all avalanches for each k and L value
            sk[Li, k] = sum([s**k_arr[k] for s in ss[Li] ]) / (N-tc[Li])

    
    fig3, ax3 = plt.subplots(figsize=(8,4.5))
    FIT = np.zeros([len(k_arr), 2])
    COV = np.zeros([len(k_arr), 2])
    
    for k in range(len(k_arr)):
        label = ('k = ' + str(int(k_arr[k])))
        
        FIT[k, :], cov = np.polyfit(np.log(L), np.log(sk[:, k]), 1, cov=True)
        COV[k, :] =  np.sqrt(np.diag(cov))
        fit = np.poly1d(FIT[k, :])    
        
        ax3.plot(L, np.exp(fit(np.log(L))), label=label)
        ax3.plot(L, sk[:, k], 'o', markersize=2.5, color='black')
        ax3.legend()
        ax3.set_xlabel('L')
        ax3.set_ylabel(r'$<s^k>$')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        fig3.tight_layout()
        

    fig4, ax4 = plt.subplots(figsize=(8,4.5))
    FIT2, COV2 = np.polyfit(k_arr, FIT[:, 0], 1, cov=True)
    fit2 = np.poly1d(FIT2)
    
    ax4.plot(k_arr, FIT[:, 0], 'o', markersize=3, label='k moment gradients')
    ax4.plot(k_arr, fit2(k_arr), label='Linear fit')
    ax4.legend()
    ax4.set_xlabel('k')
    ax4.set_ylabel(r'$D(1 + k - \tau_s )$')
    fig4.tight_layout()
       
    

#avalanche(512, 1000000, 225265, comparison=True)
avalanche_scaling(1000000)
#kmoment(1000000)


