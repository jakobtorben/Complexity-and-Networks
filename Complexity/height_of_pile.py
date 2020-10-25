# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:22:18 2020

@author: jakob
"""

import Oslo_model as om
from Oslo_model import OsloModel  # Necessary for pickling class
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

start = time.time()


def timeseries_plot(L):
    """
    Plots the height of the pile at every time step.
    
    Input: L = array of system sizes.
    """

    fig8, ax8 = plt.subplots(figsize=(8,4.5))
    for Li in L:
        model = om.OsloModel(Li, 1000000).load()
        t = model.tm[:500000]
        heights = model.heights[:500000]
        label = ('L = ' + str(int(Li)))
        ax8.plot(t, heights, label=label)
    ax8.legend()
    ax8.set_xlabel('t')
    ax8.set_ylabel('h(t;L)')
    fig8.tight_layout()


def smooth_data(runs, N, L, mode='load'):
    """
    Runs the model for different realisations and saves the average height, to 
    smooth out the data.
    
    Input: runs = No. realisations to average over
           N = iterations, L = system size, mode = function mode
           
    Returns: array of smooth heights in load mode
    
    """
    
    if mode == 'load':
        heights = np.zeros(len(L))
        
        for i in range(len(L)):
            heights[i] = np.load("Smooth_data_"+str(runs)+"runs_"+str(N)+"N_"+str(L[i])
                +"L_", allow_pickle=True)
        return heights
        
    if mode == 'generate':
        
        for Li in L:
            data = np.zeros([runs, N])
            tc_arr = np.zeros(runs)
            for i in range(runs):
                model = om.OsloModel(Li, N)
                model.Iteration(N, 0.5)
                data[i, :] = model.heights
                tc_arr[i] = model.tc
            mean = np.mean(data, axis=0)
            tc_mean = np.mean(tc_arr)
            mean.dump("Smooth_data_"+str(runs)+"runs_"+str(N)+"N_"+str(Li)+"L_")
            print(tc_mean)




def cross_over_plot(L, tc):
    """ 
    Plots the measured cross-over time to the theoretical time, as a fucntion of
    L.
    Inputs: L = array of system size
            tc = array of cross-over times for every system size
    """
    L = np.array(L, dtype=float)
    z = np.array([1.5875, 1.56875, 1.6563, 1.6906, 1.7039, 1.7105, 1.7195,
                  1.7109,], dtype=float)  # Average slopes
    
    tc_theory = z/2 * L*L*(1+ 1/L)
    tc_theory_avg = np.mean(z)/2 * L*L*(1+ 1/L)
    
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(L, tc/tc_theory, '-o', label='<z> dependent on L')
    ax.plot(L, tc/tc_theory_avg, '-o', label='<z> independent of L')
    ax.axhline(1,  color='black')
    ax.legend()
    ax.set_xlabel('L')
    ax.set_ylabel('$<t_c^{measured}>  / \, <t_c^{theory}>$')
    fig.tight_layout()

    


def data_collapse(L):
    """
    Performs data collapse for heights vs time.
    Inputs: L = array of system size
    """

    N = 500000
    data_arr = np.zeros([len(L), N])
    tm = np.arange(N)
    
    for i  in range(len(L)):    
        data_arr[i, :] = np.load("Smooth_data_"+str(20)+"runs_"+str(N)+"N_"+str(L[i])+"L_", allow_pickle=True)
    
    fig9, ax9 = plt.subplots(figsize=(8,4.5))
    for i in range(len(data_arr)):
        time = tm/L[i]**2
        height = data_arr[i, :]/L[i]
        ax9.plot(time, height, label=("L = "+str(L[i])) )

    ax9.set_xlim(-0.2, max(tm/L[-1]**2)+0.2)
    ax9.legend()
    ax9.set_xlabel("$t / L^2$")
    ax9.set_ylabel("h(t;L) / L")
    fig9.tight_layout()
    

def P_h(h, heights, tot):
    """
    Finds height probability by taking ratio of No. observed 
    configureations with height h, to total configurations.
    """
    return np.sum(heights == h)/tot

    
def avg_height(L, N, tc):
    """
    Loads in data from precalculated and returns the steady state data.
    
    Inputs: L = array of system size
            N = number of iterations
            tc = array of cross-over times for every system size
            
    Returns: array of avg height, std_dev, probability and maxh for every 
             system size
    """
    
    model = om.OsloModel(L, N).load()
    
    ind = np.where(model.tm==tc)[0][0]
    model.heights = model.heights[ind:]
    avg_height = np.mean(model.heights)
    avg_heightsqr = np.mean(np.square(model.heights))
    std_dev = np.sqrt(avg_heightsqr - avg_height*avg_height)    

    maxh = int(max(model.heights))
    probs = [P_h(h, model.heights, N-tc) for h in range(maxh + 1)]
    #total_prob = np.sum(probs)
    #print("Average height for length " + str(L)+ " = " + str(avg_height))
    #print("Standard deviation for length " + str(L)+ " = " + str(std_dev))
    #print("Total probability = " + str(total_prob))
    
    return avg_height, std_dev, probs, maxh
    
    
def height_scaling(N):
    """
    Finds the scaling relations for the height of the pile and plots the result
    
    Inputs: L = array of system size
            N = number of iterations
    """
    
    L = [4, 8, 16, 32, 64, 128, 256, 512]
    # Specific values for the pre calculated model
    tc = [12, 54, 192, 844, 3328, 13730, 55307, 227489]  

    avg_array = [avg_height(L[i], N, int(tc[i])) for i in range(len(L))]
    avg_heights, std_devs, probs, maxh = zip(*avg_array)  # splits array into
    
    r_sq = np.zeros(12)
    a0 = np.linspace(1.7, 1.78, 12)
    L = np.array(L, dtype=float)
    probs = np.array([np.array(probsi) for probsi in probs])
    x = np.log(L)
    
    for i in range(len(r_sq)):    
        y = np.log(1 - avg_heights/(a0[i]*L))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        r_sq[i] = r_value*r_value
        
    # plot a0 vs R^2 values to find best linear fit
    fig2, ax2 = plt.subplots(figsize=(8,4.5))
    ax2.plot(a0, r_sq)
    ax2.set_xlabel('$a_0$')
    ax2.set_ylabel('$R^2$')
    fig2.tight_layout()
    
    # plot linear fit using a0 estimate to find w1
    fig3, ax3 = plt.subplots(figsize=(8,4.5))
    a0 = 1.736
    y = np.log(1 - avg_heights/(a0*L))
    ax3.plot(x, y, 'x', label='Measured values')  # Don't include L=4
    FIT, COV = np.polyfit(x, y, 1, cov=True)
    fit = np.poly1d(FIT)    
    ax3.plot(x, fit(x), label=('fit: ' + r'$\log(1- <h(t;L)> / a_0 L)$' 
                    +' = %5.3f log(L) %5.3f' % tuple(FIT)))
    ax3.set_xlabel('log(L)')
    ax3.set_ylabel(r'$\log(1- <h(t;L)> / a_0 L)$')
    ax3.legend()
    fig3.tight_layout()
    print('w1 = ', -FIT[0], u'\u00B1', np.sqrt(np.diag(COV)[0]))
    
    
    # find and plot std dev scaling for h
    fig4, ax4 = plt.subplots(figsize=(8,4.5))
    ax4.plot(np.log(L), np.log(std_devs), 'x', label=('Measured $\sigma_h(L)$'))
    FIT2, COV2 = np.polyfit(np.log(L), np.log(std_devs), 1, cov=True)
    fit2 = np.poly1d(FIT2)    
    ax4.plot(np.log(L), fit2(np.log(L)), label=('fit: ' + r'$\log( \sigma_h(L))$' 
                    +' = %5.3f log(L) %5.3f' % tuple(FIT2)))
    ax4.set_xlabel('log(L)')
    ax4.set_ylabel(r'$log(\sigma_h(L))$')
    ax4.legend()
    fig4.tight_layout()
    print('sigma h power = ', FIT2[0], u'\u00B1', np.sqrt(np.diag(COV2)[0]))
    
    # find and plot std dev scaling for z
    fig5, ax5 = plt.subplots(figsize=(8,4.5))
    ax5.plot(np.log(L), np.log(std_devs/L), 'x', label=('Measured $\sigma_z(L)/L$'))
    FIT3, COV3 = np.polyfit(np.log(L), np.log(std_devs/L), 1, cov=True)
    fit3 = np.poly1d(FIT3)     
    ax5.plot(np.log(L), fit3(np.log(L)), label=('fit: ' + r'$\log( \sigma_h(L))$' 
                    +' = %5.3f log(L) %5.3f' % tuple(FIT3)))
    ax5.set_xlabel('log(L)')
    ax5.set_ylabel(r'$log(\sigma_h(L))$')
    ax5.legend()
    fig5.tight_layout()
    print('sigma z power = ', FIT3[0], u'\u00B1', np.sqrt(np.diag(COV3)[0]))
    
    
    # probability function
    fig6, ax6 = plt.subplots(figsize=(8,4.5))
    fig7, ax7 = plt.subplots(figsize=(8,4.5))
    for i in range(len(probs)):
        label = ('L = ' + str(int(L[i])))
        h = np.arange(maxh[i] + 1)
        ax6.plot(h, probs[i], label=label)
        ax6.legend()
        ax6.set_xlabel('h')
        ax6.set_ylabel('P(h;L)')
        fig6.tight_layout()
        
        arg = (h - avg_heights[i])/std_devs[i]
        y = probs[i]*np.sqrt(2*np.pi)*std_devs[i]*np.exp(1/2*(h - avg_heights[i])**2/std_devs[i]**2)
        ax7.plot(arg, y, '-o', label=label)
        ax7.legend()
        ax7.set_xlim(-3,3)
        ax7.set_ylim(0,2)
        ax7.axhline(1, linewidth=0.7, color='black')
        ax7.axvline(0, linewidth=0.7, color='black')
        ax7.set_xlabel(r'$(h - <h>) / \sigma}$')
        ax7.set_ylabel(r'$\sqrt{2 \pi}\sigma \, P(h;L) \, \exp{\frac{(h - <h>)^2}{2 \sigma^2}}$')        
        fig7.tight_layout()
    
    
def run(N, L, p):
    """
    Main function, that runs the system and prints data at the end.
    Input: N = iterations, L = lattice size, p = probability
    """
    model = om.OsloModel(L, N)
    model.Iteration(N, p)
    print("Average heights ", np.mean(model.heights))
    print('checks', model.count)

    
L = [4, 8, 16, 32, 64, 128, 256, 512]
tc = [14.7, 55.73, 215.1, 854.83, 3438.3, 13953.93, 56211.4, 225265.25]

#timeseries_plot(L)
#smooth_data(20, 500000, mode='generate')
#smooth_data(20, 500000)
#cross_over_plot(L, tc)
#data_collapse(L)
#avg_height(8, 10000, 55)
height_scaling(500000)





