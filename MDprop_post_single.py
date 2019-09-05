# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:37:36 2018

@author: Philip
"""

import numpy as np
import scipy.special as spe
from numba import jit
from numpy import linalg as LA
from scipy.spatial import Voronoi, ConvexHull

@jit
def preproc_2D_NDensity(x, y, bnlmt, proc_bw, ind_big, Lx, Ly):
    # Step 1: find the furthest distance the single large particle traveled
    ind_big = np.asarray(ind_big, dtype=bool)
    ind_big_single = np.argwhere(ind_big==1)[0]
    lower_bound = Ly
    if Lx > Ly:
        lower_bound = Lx
    upper_bound = 0

    T = np.size(y, 0) # sampled in time
    N = np.size(y, 1) # length of data (number of particles)
    
    for j in range(0, T):
        if 
    
    binsize = bnlmt // proc_bw
    binvol = proc_bw * Lx
    
    yall = np.zeros((T, binsize))
    ydata0 = np.zeros((T, binsize))
    ydata1 = np.zeros((T, binsize))    
    
    for j in range (0, T): # evolve in time
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            if (ind >= 0):
                yall[j][ind] = yall[j][ind] + 1
                if (ind_big[i]): # large particle
                    ydata0[j][ind] = ydata0[j][ind] + 1
                if (~ind_big[i]): # small particle
                    ydata1[j][ind] = ydata1[j][ind] + 1
    
    yall = np.transpose(yall)
    ydata0 = np.transpose(ydata0)
    ydata1 = np.transpose(ydata1)
    
    return yall, ydata0, ydata1

