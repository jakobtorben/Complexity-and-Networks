# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:35:54 2020

@author: jakob
"""
from randomwalk_class import RWmodel
import logbin230119_modified as lb_mod
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pickle

plt.rcParams['font.size'] = 13
        
def network_gen(n, N, m, q=0.5, mode='load'):
    """
    n = number of runs (int)
    N = number of vertices (int)
    m = number of edges to add to a new vertex (int)
    mode = save or load object (str)
    """
    
    filename = ("Datafiles2/RWnetwork_"+str(n)+"n_"+str(N)+"N_"+str(m)+"m_"+str(q)+"q.dat")
    
    if mode == 'save':
        degrees = np.zeros((n, N))
        for i in range(n):
            network = RWmodel(N, m, q)
            network.run()
            edges = np.array(network.edges)
            edges_flattened = edges.flatten().astype(int)
            degrees[i,:] = np.bincount(edges_flattened)
            
        data = degrees
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)
                
    if mode == 'load':
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    
    
def average(n, N, m, q, scale=1.2):
    
    k_arr = []
    kprob_arr = []

    degrees = np.load(("Datafiles2/RWnetwork_"+str(n)+"n_"+str(N)+"N_"+str(m)
                       +"m_"+str(q)+"q.dat"), allow_pickle=True)
    degrees = degrees.astype(int)
    for ni in range(n):
        k, kprob = lb_mod.logbin(degrees[ni, :], scale=scale)
        k_arr.append(k)
        kprob_arr.append(kprob)
    
    max_ki = 0
    for ki in range(1, len(k_arr)):
        if len(k_arr[ki]) > len(k_arr[max_ki]):
            max_ki = ki
            
    for ki in range(len(k_arr)):
        diff = len(k_arr[max_ki]) - len(k_arr[ki])
        if diff > 0:
            kprob_arr[ki] = np.hstack((kprob_arr[ki], np.zeros(diff)))

    kprob_arr = np.array(kprob_arr)        
    avg_kprob = np.mean(kprob_arr, axis=0)
    std_kprob = np.std(kprob_arr, axis=0)
    k = k_arr[max_ki]
    
    return k, avg_kprob, std_kprob


def degree_dist(n, N, m, q, scale=1.1):
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    
    for i in range(len(q)):
        k, avg_kprob, std_kprob = average(n, N, m, q=q[i], scale=scale)
        k, avg_kprob, std_kprob = k[avg_kprob!=0], avg_kprob[avg_kprob!=0], std_kprob[avg_kprob!=0]
        transformed_err = np.log10(np.e)*std_kprob/avg_kprob
        
        
        ax2.errorbar(np.log10(k), np.log10(avg_kprob), yerr=transformed_err, fmt='o', markersize=2,
                label=("Data for q = " +str(q[i])))
        
        
        if q[i] == 0:
            prob_theory = (k-m)*np.log10(m) - (k - m + 1)*np.log10((m + 1))
            prob_theory = 10**prob_theory
            ax2.plot(np.log10(k), np.log10(prob_theory), label=('$p_{∞}(k)$ Random network'))
        
        if q[i] == 0.95:
            prob_theory = 2*m*(m + 1)/((k + 2)*(k + 1)*k)
            ax2.plot(np.log10(k), np.log10(prob_theory), label=('$p_{∞}(k)$ PA network'))
            ax3.plot(k, avg_kprob/prob_theory, 'x', label=('q = '+str(q[i])) )
            
    
        
    ax2.set_xlabel('k')
    ax2.set_ylabel('p(k)')
    ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax2.xaxis.set_ticks([np.log10(x) for p in range(0,3) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax2.yaxis.set_ticks([np.log10(x) for p in range(-8,0) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    ax2.legend()
    fig2.tight_layout()
    
    ax3.set_xlabel('k')
    ax3.set_ylabel('$p(k)/p_{∞}(k)$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    fig3.tight_layout()
    
    
def chisqg(ydata,ymod, sd, weighted=True):  

      if weighted == True:  
         chisq=np.sum( ((ydata-ymod)/sd)**2 )    
      else:  
         chisq=np.sum((ydata-ymod)**2)  
        
      return chisq  


def statistics(n, N, m, scale=1.1):
    
    q = [0, 0.95]

    for i in range(len(q)):
        k, avg_kprob, std_kprob = average(n, N, m, q=q[i], scale=scale)
        k, avg_kprob, std_kprob = k[1:-1], avg_kprob[1:-1], std_kprob[1:-1]
        k, avg_kprob, std_kprob = k[std_kprob!=0], avg_kprob[std_kprob!=0], std_kprob[std_kprob!=0]
        
        if q[i] == 0:
            prob_theory = (k-m)*np.log10(m) - (k - m + 1)*np.log10((m + 1))
            prob_theory = 10**prob_theory
            chisq = chisqg(avg_kprob[:], prob_theory[:], sd=std_kprob[:])
            p_value = chi2.sf(chisq, len(avg_kprob[:])-2)
            print('q = ', q[i], ' chisq = ', chisq, 'p value = ', p_value)

        if q[i] == 0.95:
            prob_theory = 2*m*(m + 1)/((k + 2)*(k + 1)*k)
            chisq = chisqg(avg_kprob[:], prob_theory[:], sd=std_kprob[:])
            p_value = chi2.sf(chisq, len(avg_kprob[:])-2)
            print('q = ', q[i], ' chisq = ', chisq, 'p value = ', p_value)


# network_gen(50, 100000, 2, q=0.5, mode='save')

q = [0, 0.1, 0.5, 0.95]

degree_dist(50, 100000, 2, q, scale=1.1)
statistics(50, 100000, 2, scale=1.1)
