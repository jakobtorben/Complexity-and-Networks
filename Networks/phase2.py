# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:00:56 2020

@author: jakob
"""

import BAmodel as BA
import logbin230119_modified as lb_mod
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pickle


def network_gen(n, N, m, mode='load'):
    """
    n = number of runs (int)
    N = number of vertices (int)
    m = number of edges to add to a new vertex (int)
    mode = save or load object (str)
    """
    
    filename = ("Datafiles2/Randomnetwork_"+str(n)+"n_"+str(N)+"N_"+str(m)+"m.dat")
    
    if mode == 'save':
        degrees = np.zeros((n, N))
        for i in range(n):
            network = BA.BAmodel(N, m, model='random')
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
    
def average(n, N, m, scale=1.2):
    
    k_arr = []
    kprob_arr = []

    degrees = np.load(("Datafiles2/Randomnetwork_"+str(n)+"n_"+str(N)+"N_"
                       +str(m)+"m.dat"), allow_pickle=True)
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
    

def binned_plot(n, N, scale=1.2):
    
    m = [2, 4, 8, 16, 32, 64]

    
    fig2, ax2 = plt.subplots(figsize=(7,5))

    for i in range(len(m)):
        k, avg_kprob, std_kprob = average(n, N, m[i], scale=scale)
        transformed_err = np.log10(np.e)*std_kprob/avg_kprob

        ax2.errorbar(k, np.log10(avg_kprob), yerr=transformed_err, fmt='o', markersize=2,
                label=("Data for m = " +str(m[i])))        
        
        prob_theory = (k-m[i])*np.log10(m[i]) - (k - m[i] + 1)*np.log10((m[i] + 1))
        if i < len(m) - 1:
            ax2.plot(k, prob_theory, color='black')
        else:
            ax2.plot(k, prob_theory, color='black', label='$Theoretical \, p_{∞}(k)$' )

    ax2.set_xlabel('k')
    ax2.set_ylabel('p(k)')
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax2.yaxis.set_ticks([np.log10(x) for p in range(-12,0) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    ax2.legend()
    fig2.tight_layout()
    
    
def theory_check(n, N, scale=1.2):
    
    m = [2, 4, 8, 16, 32, 64]
    fig3, ax3 = plt.subplots(figsize=(7,5))

    for i in range(len(m)):
        k, avg_kprob, std_kprob = average(n, N, m[i], scale=scale)
        prob_theory = (k-m[i])*np.log10(m[i]) - (k - m[i] + 1)*np.log10((m[i] + 1))      
        x = np.log(k/np.sqrt(m[i]))
        ratio = np.log10(avg_kprob) - prob_theory

        ax3.plot(x, ratio,'x', label=("m = " +str(m[i]))) 
       
    ax3.set_xlabel(r'$k/\sqrt{m}$')
    ax3.set_ylabel('$p(k)/p_{∞}(k)$')
    ax3.xaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax3.xaxis.set_ticks([np.log10(x) for p in range(0,5) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    ax3.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax3.yaxis.set_ticks([np.log10(x) for p in range(-3,0) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)

    ax3.legend()
    fig3.tight_layout()
        
    
def largest_degree(n, N, m):
        
    k1 = np.zeros((n, len(N)))
    for i in range(len(N)):
        degrees = np.load(("Datafiles2/Randomnetwork_"+str(n)+"n_"+str(int(N[i]))+"N_"
                       +str(m)+"m.dat"), allow_pickle=True)
        for ni in range(n):            
            k1[ni, i] = max(degrees[ni, :])
            
    k1_mean = np.mean(k1, axis=0)
    k1_err = np.std(k1, axis=0)
    k1_theory = m - np.log10(N)/(np.log10(m) - np.log10(m + 1))
    
    
    fig4, ax4 = plt.subplots(figsize=(7,5))
    ax4.errorbar(N, k1_mean, yerr=k1_err, fmt='o', label="Measured values")
    ax4.plot(N, k1_theory, label="Theoretical values")
    ax4.set_xlabel('N')
    ax4.set_ylabel('$k_1$')
    ax4.legend()
    fig4.tight_layout()

    fig5, ax5 = plt.subplots(figsize=(7,5))
    ax5.errorbar(N, k1_mean, yerr=k1_err, fmt='o', label="Measured values")
    ax5.plot(N, k1_theory, label="Theoretical values")
    ax5.set_xlabel('N')
    ax5.set_ylabel('$k_1$')
    ax5.set_xscale('log')
    # ax5.set_yscale('log')
    ax5.legend()
    fig5.tight_layout()
    
    
def chisqg(ydata, ymod, sd, weighted=True):  
    """
    Parameters
    ----------
    ydata : array
        Observed data.
    ymod : array
        Expected model .
    sd : array
        Error on each measured data.
    weighted : Boolean, optional
        Weigh the sum by the standard deviation. The default is True.

    Returns
    -------
    chisq : float
        Chi squared value.
    """
    
    if weighted == True:  
         chisq=np.sum( ((ydata-ymod)/sd)**2 )    
    else:  
         chisq=np.sum((ydata-ymod)**2)  
        
    return chisq  


def statistics(n, N, scale=1.1):
    
    m = [2, 4, 8, 16, 32, 64]
    chisq_arr = np.zeros(len(m))
    fig6, ax6 = plt.subplots(figsize=(7,5))

    for i in range(len(m)):
        k, avg_kprob, std_kprob = average(n, N, m[i], scale=scale)
        prob_theory = (k-m[i])*np.log10(m[i]) - (k - m[i] + 1)*np.log10((m[i] + 1))
        prob_theory = 10**prob_theory

        chisq1 = chisqg(avg_kprob[:], prob_theory[:], sd=std_kprob[:])
        chisq2 = chisqg(avg_kprob[1:-1], prob_theory[1:-1], sd=std_kprob[1:-1])
        p_value1 = chi2.sf(chisq1, len(avg_kprob[:])-2)
        p_value2 = chi2.sf(chisq2, len(avg_kprob[1:-1])-2)

        chisq_arr[i] = chisq2/(len(avg_kprob[1:-1])-2)
     
        print('m = ', m[i], 'all points, chisq = ', chisq1, 'p value = ', p_value1)
        print('m = ', m[i], 'Sliced points, chisq = ', chisq2, 'p value = ', p_value2)
   
    ax6.plot(m, chisq_arr, 'o')
    ax6.set_xlabel('m')
    ax6.set_ylabel('$\chi^2/N_{dof}$')
    fig6.tight_layout()
        


# network_gen(1, 100000, 2, mode='save')

    
N = np.zeros(50)
N[0] = 1000 
for i in range(1, 50):
    N[i] = int(N[i-1]*1.1)


# binned_plot(1, 100000, scale=1)
binned_plot(50, 100000, scale=1.2)
# theory_check(50, 100000, scale=1.2)
# statistics(50, 100000, scale=1.2)
# largest_degree(30, N, 3)


