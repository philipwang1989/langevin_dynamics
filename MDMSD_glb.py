# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:44:51 2018

@author: Philip
"""

# import sys
import numpy as np
# from time import clock
from numba import jit
# import scipy.io as sio

def main(y, dy, Ly, ind_big):
    N_dt = len(y) # number of time delays, also need info of the true time delay
    
    output = []    
    output_l = []
    output_s = []    
    
    for i in range(0, N_dt):   
        y_input = np.transpose(np.asarray(y[i])) # Is the final output bug due to the transpose?
        # dy_input = np.transpose(np.asarray(dy[i]))
        # dy_input = np.nanmean(dy_input, axis = 0)
        # print(dy_input.shape)
        
        if y_input.shape[0] is not 0:
            [N, T] = y_input.shape
            # print(N, T)
            # print(y_input[0])
            MSD_all = []
            MSD_l_all = []
            MSD_s_all = []
            for j in range(0, N):
                y_temp = y_input[j]
                MSD = interproc_MSD_r(y_temp, Ly)
                # MSD, MSD_corrected = interproc_MSD_lm_corrected_r(y_temp, dy_input, Ly)
                MSD_all.append(MSD)
                if ind_big[j]:
                    MSD_l_all.append(MSD)
                else:
                    MSD_s_all.append(MSD)
            # print(np.asarray(MSD_all).shape)
            # print(np.asarray(MSD_corrected_all).shape)
            output.append(MSD_all)
            output_l.append(MSD_l_all)
            output_s.append(MSD_s_all)
        else:
            output.append([])
            output_l.append([])
            output_s.append([])
        print(i, j)
    
    print('Done appending!')
    
    return output, output_l, output_s

@jit
def interproc_MSD_lm_corrected_r(y, dist_y, Ly):
    # local mean velocity corrected MSD
    # Correction method see: Fan, Yi, et al. 
    # "Modelling size segregation of granular materials: the roles of segregation, advection and diffusion." 
    # Journal of Fluid Mechanics 741 (2014): 252-279. - Apeendix A. 3. Diffusion coefficient

    counts = np.zeros(len(y))
    MSDy = np.zeros(len(y))
    MSDy_corrected = np.zeros(len(y))
    for t1 in range(0, len(y) - 1):
        for t2 in range(t1 + 1, len(y)):
            y1 = y[t1]
            y2 = y[t2]
            dt_index = t2 - t1 - 1 # first index is 0
            dy = y2 - y1
            dy = dy - round(dy / Ly) * Ly
            dy2 = dy * dy
            MSDy[dt_index] += dy2
            MSDy_corrected[dt_index] += (dy2 - interproc_sum(dist_y[t1:t2]))
            counts[dt_index] = counts[dt_index] + 1
    ind = counts > 0
    
    MSDy[ind] = MSDy[ind] / counts[ind]
    MSDy_corrected[ind] = MSDy_corrected[ind] / counts[ind]
    
    MSDy = MSDy[0:-1]
    MSDy_corrected = MSDy_corrected[0:-1]
    
    return MSDy, MSDy_corrected

@jit
def interproc_sum(y):
    total = 0
    for i in range(0, len(y)):
        total = total + y[i]
        
    return total

@jit
def interproc_MSD_r(y, Ly):
    counts = np.zeros(len(y))
    MSDy = np.zeros(len(y))
    for t1 in range(0, len(y) - 1):
        for t2 in range(t1 + 1, len(y)):
            y1 = y[t1]
            y2 = y[t2]
            dt_index = t2 - t1 - 1 # first index is 0
            dy = y2 - y1
            dy = dy - round(dy / Ly) * Ly
            MSDy[dt_index] = MSDy[dt_index] + dy * dy
            counts[dt_index] = counts[dt_index] + 1
    ind = counts > 0
    
    MSDy[ind] = MSDy[ind] / counts[ind]
    
    MSDy = MSDy[0:-1]
    
    return MSDy
