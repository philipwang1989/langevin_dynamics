
# coding: utf-8

# In[ ]:

import numpy as np
import scipy.special as spe
from numba import jit
from numpy import linalg as LA
from scipy.spatial import Voronoi, ConvexHull
import matplotlib.pyplot as plt
import matplotlib.collections
# from numba import cuda

def initial_preparation_RD(N, G, D, R_hyd, HfillN, flag1):
    extra = 0
    flag = False
    while (flag == False):
        Lx = N * np.mean(2 * R_hyd) / HfillN
        Ly = (N / np.round(Lx / (2 * G * D)) + extra) * (G * D)
        x, y = np.mgrid[G:(Lx):G, G:(Ly - G):G]
        if (x.size >= N):
            flag = True
            ii = np.argsort(np.random.random(x.size))
            x = np.transpose(x)
            y = np.transpose(y)
            x = x.flatten()
            y = y.flatten()
            if (flag1 == 1):
                x = x[ii[0:N]]
                y = y[ii[0:N]]
            x = x[0:N]
            y = y[0:N]
        else:
            extra = extra + 1
    
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    return x, y, Lx, Ly

def initial_preparation_ST(N, Nb, Ns, G, D, R_hyd, HfillN):
    extra = 0
    flag = False
    while (flag == False):
        Lx = N * np.mean(2 * R_hyd) / HfillN
        Ly = (N / np.round(Lx / (2 * G * D)) + extra) * (G * D)
        xb, yb = np.mgrid[G:(Lx - G):G, G:(Ly / 2 - G):G]
        xb = np.transpose(xb)
        yb = np.transpose(yb)
        xb = xb.flatten()
        yb = yb.flatten()
        ii = np.argsort(np.random.random(xb.size))
        xb = xb[ii[0:Nb]]
        yb = yb[ii[0:Nb]]
        xs, ys = np.mgrid[G:(Lx - G):G, (Ly / 2 + G):(Ly - G):G]
        xs = np.transpose(xs)
        ys = np.transpose(ys)
        xs = xs.flatten()
        ys = ys.flatten()
        ii = np.argsort(np.random.random(xs.size))
        xs = xs[ii[0:Ns]]
        ys = ys[ii[0:Ns]]
        x = np.concatenate((xs, xb), axis = 0)
        y = np.concatenate((ys, yb), axis = 0)
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        if (x.size >= N):
            flag = True
        else:
            extra = extra + 1

    return x, y, Lx, Ly

def initial_preparation_SB(N, Nb, Ns, G, D, R_hyd, HfillN):
    extra = 0
    flag = False
    while (flag == False):
        Lx = N * np.mean(2 * R_hyd) / HfillN
        Ly = (N / np.round(Lx / (2 * G * D)) + extra) * (G * D)        
        xb, yb = np.mgrid[G:(Lx - G):G, (Ly / 2 + G):(Ly - G):G]
        xb = np.transpose(xb)
        yb = np.transpose(yb)
        xb = xb.flatten()
        yb = yb.flatten()
        ii = np.argsort(np.random.random(xb.size))
        xb = xb[ii[0:Nb]]
        yb = yb[ii[0:Nb]]
        xs, ys = np.mgrid[G:(Lx - G):G, G:(Ly / 2 - G):G]
        xs = np.transpose(xs)
        ys = np.transpose(ys)
        xs = xs.flatten()
        ys = ys.flatten()
        ii = np.argsort(np.random.random(xs.size))
        xs = xs[ii[0:Ns]]
        ys = ys[ii[0:Ns]]
        x = np.concatenate((xs, xb), axis = 0)
        y = np.concatenate((ys, yb), axis = 0)
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        if (x.size >= N):
            flag = True
        else:
            extra = extra + 1

    return x, y, Lx, Ly

def initial_preparation_VAR_ST(N, Nb, Ns, G, D, R_hyd, StreamN, HfillN, sigma_erf, mbig, msmall):
    # Deal with erf first
    mu_erf = 0.5
    ph_rcp = 0.85
    HfN = int(HfillN)
    x_erf = np.linspace(0, 1, HfillN)
    y_erf = (0.5) * (1 + spe.erf((x_erf - mu_erf) / (sigma_erf * np.sqrt(2))))
    Nsmall_erf = np.ceil((StreamN * y_erf / msmall) * ph_rcp)
    Nbig_erf = np.ceil((StreamN * (1 - y_erf) / mbig) * ph_rcp)
    #Nsmall_erf_total = np.sum(Nsmall_erf)
    #Nbig_erf_total = np.sum(Nbig_erf)
    
    # Add/Remove additional particles due to ceiling
    if (np.sum(Nsmall_erf) != Ns or np.sum(Nbig_erf) != Nb):
        dNsmall = int(Ns - np.sum(Nsmall_erf))
        dNbig = int(Nb - np.sum(Nbig_erf))
        for i in range (0, abs(dNsmall)):
            small_randlist = np.asarray(np.nonzero(Nsmall_erf >= 1))
            small_randlist = small_randlist[0][:]
            if (dNsmall > 0): # add particle
                # np.floor(2 * HfillN / 3)
                ind_rand = np.random.randint(np.floor(0.5 * HfillN), HfillN) # low (inclusive), high (exclusive)
            else:
                ind_rand = np.random.choice(small_randlist)
            Nsmall_erf[ind_rand] = Nsmall_erf[ind_rand] + np.sign(dNsmall) # if>0, add, else remove
        for i in range (0, abs(dNbig)):
            big_randlist = np.asarray(np.nonzero(Nbig_erf >= 1))
            big_randlist = big_randlist[0][:]
            if (dNbig > 0): # add particle
                ind_rand = np.random.randint(0, np.floor(HfillN / 2)) # low (inclusive), high (exclusive)
            else:
                ind_rand = np.random.choice(big_randlist)
            Nbig_erf[ind_rand] = Nbig_erf[ind_rand] + np.sign(dNbig)
    ind_small_erf = np.sort(np.cumsum(Nsmall_erf))
    
    # Artificially remove small particles from lower layer
    for layer in range (0, 2): # 2 layers (0 and 1)
        while (ind_small_erf[layer] > 0):
            ind_rand = HfN - 1 
            #ind_rand = np.random.randint(np.ceil(1 * HfillN / 2), HfillN)
            ind_small_erf[ind_rand] = ind_small_erf[ind_rand] + 1
            ind_small_erf[layer] = ind_small_erf[layer] - 1
        
    # Artificially remove big particles from upper most layer
    if (sigma_erf == 1 or sigma_erf == 0.5):
        tail = 3
    else:
        tail = 2
    ind_temp = int(np.amax(np.nonzero(Nbig_erf)))
    value_temp = np.ceil(Nbig_erf[ind_temp] / tail)
    while (Nbig_erf[ind_temp] >= value_temp):
        ind_rand = np.random.randint(1, np.ceil(1 * HfillN / 4))
        Nbig_erf[ind_rand] = Nbig_erf[ind_rand] + 1
        Nbig_erf[ind_temp] = Nbig_erf[ind_temp] - 1
    
    ind_big_erf = np.sort(np.cumsum(Nbig_erf))
    ind_small_erf = np.sort(ind_small_erf)
    ind_small_erf[ind_small_erf > Ns] = Ns
    ind_big_erf[ind_big_erf > Nb] = Nb
    ind_small_erf = ind_small_erf.astype(int) # increment # of particles
    ind_big_erf = ind_big_erf.astype(int)
    
    # Final number of particles in "each" layer
    Nsmall_erf = np.diff(ind_small_erf)
    Nsmall_erf = np.append(np.asarray(ind_small_erf[0]), Nsmall_erf) # numpy issue, single-element array treated as a scalar
    Nbig_erf = np.diff(ind_big_erf)
    Nbig_erf = np.append(np.asarray(ind_big_erf[0]), Nbig_erf)
    Nsum_erf = Nsmall_erf + Nbig_erf
    
    # Now the usual initial state preparation
    extra = 0
    flag = False
    while (flag == False):
        Lx = N * np.mean(2 * R_hyd) / HfillN
        Ly = (N / np.round(Lx / (2 * G * D)) + extra) * (G * D)
        # bug in here: need Dfluc correction for G (spacing)...maybe not?
        x_list, y_list = np.mgrid[G:(Lx):G, G:(Ly - G):G] # mgrid stop is not inclusive
        x_list = np.transpose(x_list)
        y_list = np.transpose(y_list)
        x_length = np.unique(x_list)
        x_length = x_length.size
        y_length = np.unique(y_list)
        y_length = y_length.size
        num_layers = np.ceil(Nsum_erf / x_length)
        cum_sum_layers = np.cumsum(num_layers)
        cum_sum_layers = cum_sum_layers.astype(int)
        
        rand_list_x = np.zeros(num_layers.size, dtype = object)
        rand_list_y = np.zeros(num_layers.size, dtype = object)
        if (np.sum(num_layers) <= y_length):
            flag = True
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range (1, num_layers.size): # loop through each layer
                j = i - 1
                x_temp = x_list[cum_sum_layers[j]:cum_sum_layers[i]][:] # j=0 results in csl=1 -> accessing 2nd elem. x_list
                x_temp = x_temp.flatten()
                y_temp = y_list[cum_sum_layers[j]:cum_sum_layers[i]][:]
                y_temp = y_temp.flatten()
                rand_list_x[i] = x_temp
                rand_list_y[i] = y_temp           
            temp_list_x = x_list[0:cum_sum_layers[0]][:]            
            temp_list_x = temp_list_x.flatten()            
            rand_list_x[0] = temp_list_x            
            temp_list_y = y_list[0:cum_sum_layers[0]][:]
            temp_list_y = temp_list_y.flatten()
            rand_list_y[0] = temp_list_y               
            for i in range (1, num_layers.size):    
                length_rand_list = np.arange(rand_list_x[i].size)
                length_rand_list = length_rand_list.astype(int)
                ind_rand = np.random.choice(length_rand_list, size = int(Nsum_erf[i]), replace = False) # bug here
                count = 0
                temp_rand_list_x = rand_list_x[i]
                temp_rand_list_y = rand_list_y[i]
                for j in range (0, int(Nsmall_erf[i])):
                    x[ind_small_erf[i - 1] + j] = temp_rand_list_x[ind_rand[count]]
                    y[ind_small_erf[i - 1] + j] = temp_rand_list_y[ind_rand[count]]
                    count = count + 1
                for j in range (0, int(Nbig_erf[i])):
                    x[Ns + ind_big_erf[i - 1] + j] = temp_rand_list_x[ind_rand[count]]
                    y[Ns + ind_big_erf[i - 1] + j] = temp_rand_list_y[ind_rand[count]]
                    count = count + 1
            length_rand_list = np.arange(rand_list_x[0].size)
            length_rand_list = length_rand_list.astype(int)
            ind_rand = np.random.choice(length_rand_list, size = int(Nsum_erf[0]), replace = False) # bug here
            count = 0
            temp_rand_list_x = rand_list_x[0]
            temp_rand_list_y = rand_list_y[0]  
            for j in range (0, int(Nsmall_erf[0])):
                x[j] = temp_rand_list_x[ind_rand[count]]
                y[j] = temp_rand_list_y[ind_rand[count]]
                count = count + 1
            for j in range (0, int(Nbig_erf[0])):
                x[Ns + j] = temp_rand_list_x[ind_rand[count]]
                y[Ns + j] = temp_rand_list_y[ind_rand[count]]
                count = count + 1  
        else:
            extra = extra + 1
    return x, y, Lx, Ly

def initial_preparation_VAR_SB(N, Nb, Ns, G, D, R_hyd, StreamN, HfillN, sigma_erf, mbig, msmall):
    # Deal with erf first
    mu_erf = 0.5
    ph_rcp = 0.85
    HfN = int(HfillN)
    x_erf = np.linspace(0, 1, HfillN)
    y_erf = (0.5) * (1 + spe.erf((x_erf - mu_erf) / (sigma_erf * np.sqrt(2))))
    y_erf = np.flipud(y_erf)
    Nsmall_erf = np.ceil((StreamN * y_erf / msmall) * ph_rcp)
    Nbig_erf = np.ceil((StreamN * (1 - y_erf) / mbig) * ph_rcp)
    #Nsmall_erf_total = np.sum(Nsmall_erf)
    #Nbig_erf_total = np.sum(Nbig_erf)
    
    # Add/Remove additional particles due to ceiling
    if (np.sum(Nsmall_erf) != Ns or np.sum(Nbig_erf) != Nb):
        dNsmall = int(Ns - np.sum(Nsmall_erf))
        dNbig = int(Nb - np.sum(Nbig_erf))
        for i in range (0, abs(dNsmall)):
            small_randlist = np.asarray(np.nonzero(Nsmall_erf >= 1))
            small_randlist = small_randlist[0][:]
            if (dNsmall > 0): # add particle
                # np.floor(2 * HfillN / 3)
                ind_rand = np.random.randint(np.floor(0.5 * HfillN), HfillN) # low (inclusive), high (exclusive)
            else:
                ind_rand = np.random.choice(small_randlist)
            Nsmall_erf[ind_rand] = Nsmall_erf[ind_rand] + np.sign(dNsmall)
        for i in range (0, abs(dNbig)):
            big_randlist = np.asarray(np.nonzero(Nbig_erf >= 1))
            big_randlist = big_randlist[0][:]
            if (dNbig > 0): # add particle
                ind_rand = np.random.randint(0, np.floor(HfillN / 2)) # low (inclusive), high (exclusive)
            else:
                ind_rand = np.random.choice(big_randlist)
            Nbig_erf[ind_rand] = Nbig_erf[ind_rand] + np.sign(dNbig)
    ind_small_erf = np.sort(np.cumsum(Nsmall_erf))
    
    # Artificially remove small particles from lower layer
    #for layer in range (0, 2): # 2 layers (0 and 1)
    #    while (ind_small_erf[layer] > 0):
    #        ind_rand = HfN - 1 
    #        ind_small_erf[ind_rand] = ind_small_erf[ind_rand] + 1
    #        ind_small_erf[layer] = ind_small_erf[layer] - 1
        
    # Artificially remove big particles from upper most layer
    #if (sigma_erf == 1 or sigma_erf == 0.5):
    #    tail = 3
    #else:
    #    tail = 2
    #ind_temp = int(np.amax(np.nonzero(Nbig_erf)))
    #value_temp = np.ceil(Nbig_erf[ind_temp] / tail)
    #while (Nbig_erf[ind_temp] >= value_temp):
    #    ind_rand = np.random.randint(1, np.ceil(1 * HfillN / 4))
    #    Nbig_erf[ind_rand] = Nbig_erf[ind_rand] + 1
    #    Nbig_erf[ind_temp] = Nbig_erf[ind_temp] - 1
    
    ind_big_erf = np.sort(np.cumsum(Nbig_erf))
    ind_small_erf = np.sort(ind_small_erf)
    ind_small_erf[ind_small_erf > Ns] = Ns
    ind_big_erf[ind_big_erf > Nb] = Nb
    ind_small_erf = ind_small_erf.astype(int) # increment # of particles
    ind_big_erf = ind_big_erf.astype(int)
    
    # Final number of particles in "each" layer
    Nsmall_erf = np.diff(ind_small_erf)
    Nsmall_erf = np.append(np.asarray(ind_small_erf[0]), Nsmall_erf) # numpy issue, single-element array treated as a scalar
    Nbig_erf = np.diff(ind_big_erf)
    Nbig_erf = np.append(np.asarray(ind_big_erf[0]), Nbig_erf)
    Nsum_erf = Nsmall_erf + Nbig_erf

    # Now the usual initial state preparation
    extra = 0
    flag = False
    while (flag == False):
        Lx = N * np.mean(2 * R_hyd) / HfillN
        Ly = (N / np.round(Lx / (2 * G * D)) + extra) * (G * D)
        # bug in here: need Dfluc correction for G (spacing)
        x_list, y_list = np.mgrid[G:(Lx):G, G:(Ly - G):G] # mgrid stop is not inclusive
        x_list = np.transpose(x_list)
        y_list = np.transpose(y_list)
        x_length = np.unique(x_list)
        x_length = x_length.size
        y_length = np.unique(y_list)
        y_length = y_length.size
        num_layers = np.ceil(Nsum_erf / x_length)
        cum_sum_layers = np.cumsum(num_layers)
        cum_sum_layers = cum_sum_layers.astype(int)
        rand_list_x = np.zeros(num_layers.size, dtype = object)
        rand_list_y = np.zeros(num_layers.size, dtype = object)
        if (np.sum(num_layers) <= y_length):
            flag = True
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range (1, num_layers.size): # loop through each layer
                j = i - 1
                x_temp = x_list[cum_sum_layers[j]:cum_sum_layers[i]][:] # j=0 results in csl=1 -> accessing 2nd elem. x_list
                x_temp = x_temp.flatten()
                y_temp = y_list[cum_sum_layers[j]:cum_sum_layers[i]][:]
                y_temp = y_temp.flatten()
                rand_list_x[i] = x_temp
                rand_list_y[i] = y_temp           
            # Need to flatten temp_list_x and temp_list_y here:
            temp_list_x = x_list[0:cum_sum_layers[0]][:]            
            temp_list_x = temp_list_x.flatten()            
            rand_list_x[0] = temp_list_x            
            temp_list_y = y_list[0:cum_sum_layers[0]][:]
            temp_list_y = temp_list_y.flatten()
            rand_list_y[0] = temp_list_y
            for i in range (1, num_layers.size):    
                length_rand_list = np.arange(rand_list_x[i].size)
                length_rand_list = length_rand_list.astype(int)
                ind_rand = np.random.choice(length_rand_list, size = int(Nsum_erf[i]), replace = False)
                count = 0
                temp_rand_list_x = rand_list_x[i]
                temp_rand_list_y = rand_list_y[i]
                for j in range (0, int(Nsmall_erf[i])):
                    x[ind_small_erf[i - 1] + j] = temp_rand_list_x[ind_rand[count]]
                    y[ind_small_erf[i - 1] + j] = temp_rand_list_y[ind_rand[count]]
                    count = count + 1
                for j in range (0, int(Nbig_erf[i])):
                    x[Ns + ind_big_erf[i - 1] + j] = temp_rand_list_x[ind_rand[count]]
                    y[Ns + ind_big_erf[i - 1] + j] = temp_rand_list_y[ind_rand[count]]
                    count = count + 1            
            length_rand_list = np.arange(rand_list_x[0].size)
            length_rand_list = length_rand_list.astype(int)
            ind_rand = np.random.choice(length_rand_list, size = int(Nsum_erf[0]), replace = False)
            count = 0
            temp_rand_list_x = rand_list_x[0]
            temp_rand_list_y = rand_list_y[0]  
            for j in range (0, int(Nsmall_erf[0])):
                x[j] = temp_rand_list_x[ind_rand[count]]
                y[j] = temp_rand_list_y[ind_rand[count]]
                count = count + 1
            for j in range (0, int(Nbig_erf[0])):
                x[Ns + j] = temp_rand_list_x[ind_rand[count]]
                y[Ns + j] = temp_rand_list_y[ind_rand[count]]
                count = count + 1  
        else:
            extra = extra + 1
    return x, y, Lx, Ly

@jit
def preproc_2D_AccurateDensity(m, Dn, z, bnlmt, proc_bw, ind_big, rho_l, rho_s, Lx): # m is the volume
    binsize = bnlmt // proc_bw
    binvol = proc_bw * Lx
    binbound = np.arange(0, bnlmt + proc_bw, proc_bw)    
    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)
    
    yvol = np.zeros((T, binsize)) # Sum of large and small volume
    ymass = np.zeros((T, binsize)) # Sum of large and small volume
    yvol_l = np.zeros((T, binsize)) # Sum of large volume
    yvol_s = np.zeros((T, binsize)) # Sum of small volume
    
    for j in range (0, T): # evolve in time
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1 # indicating which layer
            if (ind >= 0):
                R = Dn[i] / 2
                lower_bound_check = abs(z[j][i] - binbound[ind])
                upper_bound_check = abs(z[j][i] - binbound[ind + 1])
                if (lower_bound_check < R and upper_bound_check > R): # overlaps the lower bound                    
                    R2 = R * R
                    delta_y = z[j][i] - binbound[ind]
                    if (R2 - delta_y * delta_y < 0):
                        print(delta_y, R)
                    A_major = (0.5 * np.pi * R2) + (delta_y * np.sqrt(R2 - delta_y * delta_y)) + (R * np.arcsin(delta_y / R))
                    A_minor = np.pi * R2 - A_major
                    if ind == 0: # lowest layer
                        yvol[j][ind] = yvol[j][ind] + A_major
                        if ind_big[i]:
                            yvol_l[j][ind] = yvol_l[j][ind] + A_major
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_l
                        else:
                            yvol_s[j][ind] = yvol_s[j][ind] + A_major
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_s
                    else:
                        yvol[j][ind] = yvol[j][ind] + A_major
                        yvol[j][ind - 1] = yvol[j][ind - 1] + A_minor
                        if ind_big[i]:
                            yvol_l[j][ind] = yvol_l[j][ind] + A_major
                            yvol_l[j][ind - 1] = yvol_l[j][ind - 1] + A_minor
                            
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_l
                            ymass[j][ind - 1] = ymass[j][ind - 1] + A_minor * rho_l
                        else:
                            yvol_s[j][ind] = yvol_s[j][ind] + A_major
                            yvol_s[j][ind - 1] = yvol_s[j][ind - 1] + A_minor
                            
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_s
                            ymass[j][ind - 1] = ymass[j][ind - 1] + A_minor * rho_s
                    
                elif (lower_bound_check > R and upper_bound_check < R): # overlaps the upper bound
                    R = Dn[i] / 2
                    R2 = R * R
                    delta_y = binbound[ind + 1] - z[j][i]
                    A_major = (0.5 * np.pi * R2) + (delta_y * np.sqrt(R2 - delta_y * delta_y)) + (R * np.arcsin(delta_y / R))
                    A_minor = np.pi * R2 - A_major
                    if ind == (binsize - 1): # highest layer
                        yvol[j][ind] = yvol[j][ind] + A_major
                        if ind_big[i]:
                            yvol_l[j][ind] = yvol_l[j][ind] + A_major
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_l
                        else:
                            yvol_s[j][ind] = yvol_s[j][ind] + A_major
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_s
                    else:
                        if ind > (binsize - 1):
                            print('Shit')                            
                        yvol[j][ind] = yvol[j][ind] + A_major
                        yvol[j][ind + 1] = yvol[j][ind + 1] + A_minor
                        if ind_big[i]:
                            yvol_l[j][ind] = yvol_l[j][ind] + A_major
                            yvol_l[j][ind + 1] = yvol_l[j][ind + 1] + A_minor
                            
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_l
                            ymass[j][ind + 1] = ymass[j][ind + 1] + A_minor * rho_l
                        else:
                            yvol_s[j][ind] = yvol_s[j][ind] + A_major
                            yvol_s[j][ind + 1] = yvol_s[j][ind + 1] + A_minor
                            
                            ymass[j][ind] = ymass[j][ind] + A_major * rho_s
                            ymass[j][ind + 1] = ymass[j][ind + 1] + A_minor * rho_s
                    
                elif (lower_bound_check < R and upper_bound_check < R): # overlpas both bounds (very rare)
                    print('Pending!')
                    return None
                else: # Not overlaping any bound
                    yvol[j][ind] = yvol[j][ind] + m[i]
                    if (ind_big[i]): # large particle
                        yvol_l[j][ind] = yvol_l[j][ind] + m[i]
                        ymass[j][ind] = ymass[j][ind] + m[i] * rho_l
                    else:
                        yvol_s[j][ind] = yvol_s[j][ind] + m[i]
                        ymass[j][ind] = ymass[j][ind] + m[i] * rho_s
    
    ymass_l = yvol_l * rho_l
    ymass_s = yvol_s * rho_s
    
    yvol = np.transpose(yvol)
    yvol_l = np.transpose(yvol_l)
    yvol_s = np.transpose(yvol_s)
    
    ymass = np.transpose(ymass)
    ymass_l = np.transpose(ymass_l)
    ymass_s = np.transpose(ymass_s)
    
    ph = yvol / binvol
    
    return yvol, yvol_l, yvol_s, ymass, ymass_l, ymass_s, ph

@jit
def preproc_2D_NDensity(m, z, bnlmt, proc_bw, ind_big, Lx):
    binsize = bnlmt // proc_bw
    binvol = proc_bw * Lx
    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)
    
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

@jit
def preproc_2D_MDensity(m, z, bnlmt, proc_bw, ind_big, Lx):
    binsize = bnlmt // proc_bw
    binvol = proc_bw * Lx
    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)
    
    yall = np.zeros((T, binsize))
    ydata0 = np.zeros((T, binsize))
    ydata1 = np.zeros((T, binsize))    
    
    for j in range (0, T): # evolve in time
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            if (ind >= 0):
                yall[j][ind] = yall[j][ind] + m[i]
                if (ind_big[i]): # large particle
                    ydata0[j][ind] = ydata0[j][ind] + m[i]
                if (~ind_big[i]): # small particle
                    ydata1[j][ind] = ydata1[j][ind] + m[i]
    
    yall = np.transpose(yall)
    ydata0 = np.transpose(ydata0)
    ydata1 = np.transpose(ydata1)
    
    return yall, ydata0, ydata1

@jit
def preproc_2D_VolDensity_w_ph(m, z, bnlmt, proc_bw, ind_big, Lx):
    binsize = bnlmt // proc_bw
    binvol = proc_bw * Lx
    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)
    
    yall = np.zeros((T, binsize))
    ydata0 = np.zeros((T, binsize))
    ydata1 = np.zeros((T, binsize))    
    
    for j in range (0, T): # evolve in time
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            if (ind >= 0):
                yall[j][ind] = yall[j][ind] + m[i]
                if (ind_big[i]): # large particle
                    ydata0[j][ind] = ydata0[j][ind] + m[i]
                if (~ind_big[i]): # small particle
                    ydata1[j][ind] = ydata1[j][ind] + m[i]
    
    yall = np.transpose(yall)
    ydata0 = np.transpose(ydata0)
    ydata1 = np.transpose(ydata1)
    
    ph = yall / binvol
    
    return yall, ydata0, ydata1, ph

@jit
def preproc_2D_vs_simple(vy, m, z, bnlmt, proc_bw): # get average velocity
    if proc_bw != 0:
        binsize = bnlmt // proc_bw    
        T = np.size(z, 0) # sampled in time
        N = np.size(z, 1) # length of data (number of particles)    
        
        vy_all = np.zeros((T, binsize))
        
        for j in range (0, T):
            count_all = np.zeros(binsize)
            for i in range (0, N):
                ind = int(np.ceil(z[j][i] / proc_bw)) - 1
                count_all[ind] = count_all[ind] + 1
                vy_all[j][ind] = vy_all[j][ind] + vy[j][i]

            vy_all[j][:] = vy_all[j][:] / count_all # division by zero!

        vy_all = np.transpose(vy_all)
    else:
        T = np.size(z, 0) # sampled in time
        vy_all = np.zeros(T)
        for j in range(0, T):
            vy_all[j] = np.mean(vy[j])
    
    return vy_all

@jit
def preproc_2D_vs(vy, m, ind_big, z, bnlmt, proc_bw): # get average velocity and fluctuation components
    binsize = bnlmt // proc_bw    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)    

    vy_big = np.zeros((T, binsize))
    vy_small = np.zeros((T, binsize))
    vy_all = np.zeros((T, binsize))
    vy_prime = np.zeros((T, N))
    vy_prime_species = np.zeros((T, N))
  
    for j in range (0, T):
        count_big = np.zeros(binsize)
        count_small = np.zeros(binsize)
        count_all = np.zeros(binsize)
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            # print(j,i,ind)
            if (ind_big[i]): # large particle
                vy_big[j][ind] = vy_big[j][ind] + vy[j][i]
                count_big[ind] = count_big[ind] + 1
            if (~ind_big[i]): # small particle
                vy_small[j][ind] = vy_small[j][ind] + vy[j][i]
                count_small[ind] = count_small[ind] + 1
                
            count_all[ind] = count_all[ind] + 1
            vy_all[j][ind] = vy_all[j][ind] + vy[j][i]
            
        vy_big[j][:] = vy_big[j][:] / count_big # division by zero!
        vy_small[j][:] = vy_small[j][:] / count_small # division by zero!
        vy_all[j][:] = vy_all[j][:] / count_all # division by zero!
        
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            vy_prime[j][i] = vy[j][i] - vy_all[j][ind]
            if (ind_big[i]): # large particle
                vy_prime_species[j][i] = vy[j][i] - vy_big[j][ind]
            if (~ind_big[i]): # small particle
                vy_prime_species[j][i] = vy[j][i] - vy_small[j][ind]

    vy_big = np.transpose(vy_big)
    vy_small = np.transpose(vy_small)
    vy_all = np.transpose(vy_all)
    
    return vy_big, vy_small, vy_all, vy_prime, vy_prime_species

@jit
def preproc_2D_T(vy, vy_species, m, Vol, ind_big, z, bnlmt, proc_bw):
    binsize = bnlmt // proc_bw    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)
    
    Ek_big = np.zeros((T, binsize))
    Ek_small = np.zeros((T, binsize))  
    Ek_all = np.zeros((T, binsize))  
    T_big = np.zeros((T, binsize))
    T_small = np.zeros((T, binsize))
    T_all = np.zeros((T, binsize))  
  
    for j in range (0, T):
        count_big = np.zeros(binsize)
        count_small = np.zeros(binsize)
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            Ek_all[j][ind] = Ek_all[j][ind] + 0.5 * m[i] * vy[j][i] * vy[j][i]
            if (ind_big[i]): # large particle
                Ek_big[j][ind] = Ek_big[j][ind] + 0.5 * m[i] * vy_species[j][i] * vy_species[j][i]
                count_big[ind] = count_big[ind] + 1
            if (~ind_big[i]): # small particle
                Ek_small[j][ind] = Ek_small[j][ind] + 0.5 * m[i] * vy_species[j][i] * vy_species[j][i]
                count_small[ind] = count_small[ind] + 1
        
        T_big[j][:] = Ek_big[j][:] / count_big # division by zero!
        T_small[j][:] = Ek_small[j][:] / count_small
        T_all[j][:] = Ek_all[j][:] / (count_big + count_small)

    Ek_all = np.transpose(Ek_all)
    Ek_big = np.transpose(Ek_big)
    Ek_small = np.transpose(Ek_small)
    T_all = np.transpose(T_all)
    T_big = np.transpose(T_big)
    T_small = np.transpose(T_small)
        
    return Ek_all, Ek_big, Ek_small, T_all, T_big, T_small

@jit
def preproc_2D_Txy(vx, vx_species, vy, vy_species, m, Vol, ind_big, z, bnlmt, proc_bw):
    binsize = bnlmt // proc_bw    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # length of data (number of particles)
    Ek_big = np.zeros((T, binsize))
    Ek_small = np.zeros((T, binsize))  
    Ek_all = np.zeros((T, binsize))  
    T_big = np.zeros((T, binsize))
    T_small = np.zeros((T, binsize))
    T_all = np.zeros((T, binsize))  
  
    for j in range (0, T):
        count_big = np.zeros(binsize)
        count_small = np.zeros(binsize)
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            Ek_all[j][ind] = Ek_all[j][ind] + 0.5 * m[i] * vx[j][i] * vy[j][i]
            if (ind_big[i]): # large particle
                Ek_big[j][ind] = Ek_big[j][ind] + 0.5 * m[i] * vx_species[j][i] * vy_species[j][i]
                count_big[ind] = count_big[ind] + 1
            if (~ind_big[i]): # small particle
                Ek_small[j][ind] = Ek_small[j][ind] + 0.5 * m[i] * vx_species[j][i] * vy_species[j][i]
                count_small[ind] = count_small[ind] + 1
        
        T_big[j][:] = Ek_big[j][:] / count_big # division by zero!
        T_small[j][:] = Ek_small[j][:] / count_small
        T_all[j][:] = Ek_all[j][:] / (count_big + count_small)

    Ek_all = np.transpose(Ek_all)
    Ek_big = np.transpose(Ek_big)
    Ek_small = np.transpose(Ek_small)
    T_all = np.transpose(T_all)
    T_big = np.transpose(T_big)
    T_small = np.transpose(T_small)
        
    return Ek_all, Ek_big, Ek_small, T_all, T_big, T_small    

def pour_particle(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, gv, a_thresh, v_thresh, plotit, BottomWallTag):
    g = 0.1
    nt = 0
    flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt * dt) / 2 
        y = y + vy * dt + ay_old * (dt * dt) / 2
        
        x = np.mod(x, Lx)
        y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        if (gv.size <= 1 and gv == 0):
            Fx, Fy = force(Fx, Fy, N, x, y, Lx, Ly, K, R_eff)
        else:
            Fx, Fy = force_rest(Fx, Fy, N, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)                
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
        
        # left wall
        #iil = (x < Dn/2)
        #dw = x[iil] - Dn[iil] / 2
        #Fx[iil] = Fx[iil] - K * dw
        
        # right wall
        #iir = (x > (Lx - Dn/2))
        #dw = x[iir] - (Lx - (Dn[iir] / 2))
        #Fx[iir] = Fx[iir] - K * dw
    
        #############
    
        Fx = Fx - B_pour * vx
        Fy = Fy - B_pour * vy
        
        if (flag == False):
            ax = Fx / np.mean(m)
            ay = Fy / np.mean(m) - g
        else:
            ax = Fx / m
            ay = Fy / m - g
    
        ax[iib] = 0.0
        if (BottomWallTag):
            ay[N_org:N] = 0.0 # fixed the bottom wall particle
    
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2
        
        vx[iib] = 0.0
        if (BottomWallTag):
            vy[N_org:N] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay
        
        if (nt % 5000 == 0 and plotit):
            d_acc = np.sqrt(np.amax(ax * ax + ay * ay)) - a_thresh
            d_vel = np.sqrt(np.amax(vx * vx + vy * vy)) - v_thresh
            print(d_acc, d_vel)
        
        nt = nt + 1
        
        if (np.sqrt(np.amax(ax * ax + ay * ay)) < (a_thresh) and np.sqrt(np.amax(vx * vx + vy * vy)) < (v_thresh)):
            if (flag == False):
                flag = True
            else:
                #print('gravity MS reached')
                break
            
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old

def drop_wall(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, Nc_org, R_eff, K, B_pour, gv, a_thresh, v_thresh, plotit):
    g = 0.1
    nt = 0
    flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt * dt) / 2 
        y = y + vy * dt + ay_old * (dt * dt) / 2
        
        x = np.mod(x, Lx)
        y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        # ph = np.pi * R2n
        
        if (gv.size <= 1 and gv == 0):
            Fx, Fy = force(Fx, Fy, Nc_org, x, y, Lx, Ly, K, R_eff)
        else:
            Fx, Fy = force_rest(Fx, Fy, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)  
        
        # ph = ph / (R * R) / np.pi
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
        
        # top wall
        #iit = (y[0:Nc_org] + Dn[0:Nc_org]/2) > y[-1]
        iit = (y + Dn/2) > y[-1]
        iit[-1] = 0
        dw = y[iit] + Dn[iit]/2 - y[-1]
        Fy[iit] = Fy[iit] - K * dw
        Fy[-1] = np.sum(K * dw)
        
        # left wall
        #iil = (x < Dn/2)
        #dw = x[iil] - Dn[iil] / 2
        #Fx[iil] = Fx[iil] - K * dw
        
        # right wall
        #iir = (x > (Lx - Dn/2))
        #dw = x[iir] - (Lx - (Dn[iir] / 2))
        #Fx[iir] = Fx[iir] - K * dw
    
        #############
    
        Fx = Fx - B_pour * vx
        Fy = Fy - B_pour * vy
        
        if (flag == False):
            ax = Fx / np.mean(m)
            ay = Fy / np.mean(m) - g
        else:
            ax = Fx / m
            ay = Fy / m - g
    
        ax[iib] = 0.0
        #if (BottomWallTag):
        #    ay[N_org:N] = 0.0 # fixed the bottom wall particle
    
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2
        
        vx[iib] = 0.0
        #if (BottomWallTag):
        #    vy[N_org:N] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay
        
        if (nt % 5000 == 0 and plotit):
            d_acc = np.sqrt(np.amax(ax * ax + ay * ay)) - a_thresh
            d_vel = np.sqrt(np.amax(vx * vx + vy * vy)) - v_thresh
            print(d_acc, d_vel)
        
        nt = nt + 1
        
        if (np.sqrt(np.amax(ax * ax + ay * ay)) < (a_thresh) and np.sqrt(np.amax(vx * vx + vy * vy)) < (v_thresh)):
            if (flag == False):
                flag = True
            else:
                #print('gravity MS reached')
                break
            
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old

def drop_wall_w_bottom_flat(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_wall, N_org, R_eff, K, B_pour, a_thresh, v_thresh, plotit, BottomWallTag):
    g = 0.1
    nt = 0
    flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt * dt) / 2 
        y = y + vy * dt + ay_old * (dt * dt) / 2
        
        x = np.mod(x, Lx)
        y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        # ph = np.pi * R2n
        
        Fx, Fy = force(Fx, Fy, N_wall, x, y, Lx, Ly, K, R_eff)
        
        # ph = ph / (R * R) / np.pi
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
        
        # top wall
        #iit = (y[0:Nc_org] + Dn[0:Nc_org]/2) > y[-1]
        iit = (y + Dn/2) > y[-1]
        iit[-1] = 0
        dw = y[iit] + Dn[iit]/2 - y[-1]
        Fy[iit] = Fy[iit] - K * dw
                
        # left wall
        #iil = (x < Dn/2)
        #dw = x[iil] - Dn[iil] / 2
        #Fx[iil] = Fx[iil] - K * dw
        
        # right wall
        #iir = (x > (Lx - Dn/2))
        #dw = x[iir] - (Lx - (Dn[iir] / 2))
        #Fx[iir] = Fx[iir] - K * dw
    
        #############
    
        Fx = Fx - B_pour * vx
        Fy = Fy - B_pour * vy
        
        Fy[-1] = np.sum(K * dw)
        
        if (flag == False):
            ax = Fx / np.mean(m)
            ay = Fy / np.mean(m) - g
        else:
            ax = Fx / m
            ay = Fy / m - g
    
        ax[iib] = 0.0
        if (BottomWallTag):
            ay[N_org:N_wall] = 0.0 # fixed the bottom wall particle
    
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2
        
        vx[iib] = 0.0
        if (BottomWallTag):
            vy[N_org:N_wall] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay
        
        if (nt % 5000 == 0 and plotit):
            d_acc = np.sqrt(np.amax(ax * ax + ay * ay)) - a_thresh
            d_vel = np.sqrt(np.amax(vx * vx + vy * vy)) - v_thresh
            print(d_acc, d_vel)
        
        nt = nt + 1
        
        if (np.sqrt(np.amax(ax * ax + ay * ay)) < (a_thresh) and np.sqrt(np.amax(vx * vx + vy * vy)) < (v_thresh)):
            if (flag == False):
                flag = True
            else:
                #print('gravity MS reached')
                break
            
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old

def drop_wall_w_bottom(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_bottom, N_top, N_org, R_eff, K, B_pour, gv, a_thresh, v_thresh, plotit):
    g = 0.1
    nt = 0

    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt * dt) / 2 
        y = y + vy * dt + ay_old * (dt * dt) / 2
        
        x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        if (gv.size <= 1 and gv == 0):
            Fx, Fy = force(Fx, Fy, N_top, x, y, Lx, Ly, K, R_eff)
        else:
            Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
    
        #############
    
        Fx = Fx - B_pour * vx
        Fy = Fy - B_pour * vy
        
        #if y[-1] < Ly / 2:
        #    Fy[N_bottom:-1] = Fy[N_bottom:-1] - B_pour * 10 * vy[N_bottom:-1]
        
        ax = Fx / m
        ay = Fy / m - g
    
        ax[iib] = 0.0
        
        # Bottom wall
        ax[N_org:N_bottom] = 0.0 # fixed the bottom wall particle
        ay[N_org:N_bottom] = 0.0 # fixed the bottom wall particle
        
        # Top wall
        ax[N_bottom:N] = np.mean(ax[N_bottom:N])
        ay[N_bottom:N] = np.mean(ay[N_bottom:N])
    
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2
        
        vx[iib] = 0.0
        vx[N_org:N_bottom] = 0.0 # fixed the bottom wall particle
        vy[N_org:N_bottom] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay
        
        if (nt % 5000 == 0 and plotit):
            d_acc = np.sqrt(np.amax(ax * ax + ay * ay)) - a_thresh
            d_vel = np.sqrt(np.amax(vx * vx + vy * vy)) - v_thresh
            print(d_acc, d_vel)
        
        nt = nt + 1
        
        if (np.sqrt(np.amax(ax * ax + ay * ay)) < (a_thresh) and np.sqrt(np.amax(vx * vx + vy * vy)) < (v_thresh)):                
            break
            
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old

def shear_plate_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_bottom, N_top, Nc_org, R_eff, K, B, vf, gv, a_thresh, v_thresh, plotit):
    # Velocity Verlet
    ind_big = np.asarray(ind_big, dtype=bool)
    Vol = Dn * Dn * np.pi / 4
    Nbig = sum(ind_big)
    Nsmall = Nc_org - Nbig
    dt_half = np.sqrt(dt)
    dt2 = dt ** 2
    g = 0.1
    nt = 0
    saveskip = 2000 # corresponds to 100 tau_c, dt = tau_c/20
    saveskip_pos = 100 # corresponds to 2 tau_c
    saveskip_MSD = 20 # corresponds to 1 tau_c
    
    count = 0
    #count1 = 0
    datalength = int(Nt / saveskip)
    datalength_pos = int(Nt / saveskip_pos)
    acc_list = np.zeros(datalength)
    CM_big = np.zeros(datalength)
    CM_small = np.zeros(datalength)
    ph_list = np.zeros(datalength)
    T_list = np.zeros(datalength)   
    Tx_list = np.zeros(datalength)   
    Ty_list = np.zeros(datalength)
    T_big = np.zeros(datalength)    
    T_small = np.zeros(datalength) 
    Ek_list = np.zeros(datalength)   
    Ekx_list = np.zeros(datalength)   
    Eky_list = np.zeros(datalength)
    Ek_big = np.zeros(datalength)    
    Ek_small = np.zeros(datalength) 
    Tx_big_list = np.zeros(datalength) 
    Ty_big_list = np.zeros(datalength) 
    Tx_small_list = np.zeros(datalength) 
    Ty_small_list = np.zeros(datalength) 
    
    y_wall_data = np.zeros(datalength)
    
    x_data = np.zeros((datalength, Nc_org))
    y_data = np.zeros((datalength, Nc_org))
    vx_data = np.zeros((datalength, Nc_org))
    vy_data = np.zeros((datalength, Nc_org))
    
    ######################################
    #xdata_MSD = []
    #ydata_MSD = []        
    
    vx = vx - np.mean(vx)
    vy = vy - np.mean(vy)
    
    flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt2) / 2 
        y = y + vy * dt + ay_old * (dt2) / 2
        
        # x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        if (gv.size <= 2 and gv == 0):
            Fx, Fy = force(Fx, Fy, N_top, x, y, Lx, Ly, K, R_eff)
        else:
            Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw      
        
        #############
        ax = Fx / m
        ay = Fy / m - g
        
        ax[iib] = 0.0
        
        # Bottom wall
        ax[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        ay[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle            
        
        # Top wall
        ax[N_bottom:N] = np.mean(ax[N_bottom:N])
        ay[N_bottom:N] = np.mean(ay[N_bottom:N])                
        
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2

        vx[iib] = 0.0
        vx[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        vy[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        
        vx[N_bottom:N] = vf
        
        ax_old = ax
        ay_old = ay

        if (nt % saveskip == 0):
            y_CM = y[0:Nc_org]   
            m_CM = m[0:Nc_org]
            m_temp = m[0:Nc_org]
            
            vx_temp = vx[0:Nc_org] 
            vy_temp = vy[0:Nc_org] 
            
            d_acc = np.sqrt(np.amax(ax[0:Nc_org] * ax[0:Nc_org] + ay[0:Nc_org] * ay[0:Nc_org]))
            
            Ek_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2 + (vy_temp - np.mean(vy_temp)) ** 2))
            Ekx_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2))
            Eky_temp = (0.5 * m_temp * ((vy_temp - np.mean(vy_temp)) ** 2))
            Ek_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 + (vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Ek_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 + (vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
            Ekx_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 ))
            Ekx_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 ))
            Eky_big_temp = (0.5 * m_temp[ind_big] * ((vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Eky_small_temp = (0.5 * m_temp[~ind_big] * ((vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
                        
            T_temp = sum(Ek_temp) / Nc_org
            Tx_temp = sum(Ekx_temp) / Nc_org
            Ty_temp = sum(Eky_temp) / Nc_org
            T_big_temp = sum(Ek_big_temp) / Nbig
            T_small_temp = sum(Ek_small_temp) / Nsmall            
            Tx_big_temp = sum(Ekx_big_temp) / Nbig
            Tx_small_temp = sum(Ekx_small_temp) / Nsmall
            Ty_big_temp = sum(Eky_big_temp) / Nbig            
            Ty_small_temp = sum(Eky_small_temp) / Nsmall
            
            T_list[count] = T_temp
            Tx_list[count] = Tx_temp
            Ty_list[count] = Ty_temp
            T_big[count] = T_big_temp
            T_small[count] = T_small_temp
            Tx_big_list[count] = Tx_big_temp
            Ty_big_list[count] = Ty_big_temp
            Tx_small_list[count] = Tx_small_temp
            Ty_small_list[count] = Ty_small_temp
            
            Ek_list[count] = sum(Ek_temp)
            Ekx_list[count] = sum(Ekx_temp)
            Eky_list[count] = sum(Eky_temp)            
            Ek_big[count] = sum(Ek_big_temp)
            Ek_small[count] = sum(Ek_small_temp)
            
            CM_big[count] = np.mean(y_CM[ind_big])
            CM_small[count] = np.mean(y_CM[~ind_big])
            
            ph_list[count] = sum(Vol[0:Nc_org]) / (Lx * y[-1] - sum(Vol[Nc_org:N] / 2))
            
            y_wall_data[count] = y[-1]
            
            x_data[count] = x[0:Nc_org] 
            y_data[count] = y[0:Nc_org] 
            vx_data[count] = vx[0:Nc_org] 
            vy_data[count] = vy[0:Nc_org] 
            
            if (nt % (saveskip * 100) == 0 and plotit):
                print(nt, T_temp, T_big[count] / T_small[count], Tx_temp / Ty_temp)

            count = count + 1
            
        nt = nt + 1
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list

def couette_flow_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_bottom, N_top, Nc_org, R_eff, K, B, gamma_dot, P_plate, gv, a_thresh, v_thresh, N_per_coll, save_reduction, plotit):
    # Velocity Verlet
    ind_big = np.asarray(ind_big, dtype=bool)
    Vol = Dn * Dn * np.pi / 4
    Vol_temp = Vol[0:Nc_org]
    Vol_temp_sum = sum(Vol_temp)
    Vol_wall_sum = sum(Vol[Nc_org:N] / 2)
    m_temp = m[0:Nc_org]
    # m_temp_sum = sum(m_temp)
    Nbig = sum(ind_big)
    Nsmall = Nc_org - Nbig
    # dt_half = np.sqrt(dt)
    dt2 = dt * dt
    g = 0.1
    nt = 0
    settling_time = np.sqrt(Dn / g)
    saveskip = N_per_coll * 100 # corresponds to 100 tau_c, dt = tau_c/20
    # saveskip_pos = N_per_coll * 5 # corresponds to 5 tau_c    
    # save_reduction = 0.50 # saving only last 50% of the data
    count = 0
    count1 = 0
    datalength = int((1 - save_reduction) * Nt / saveskip)
    datalength_CM = int(Nt / saveskip)
    #acc_list = np.zeros(datalength)
    CM_big = np.zeros(datalength_CM)
    CM_small = np.zeros(datalength_CM)
    ph_list = np.zeros(datalength)
    T_list = np.zeros(datalength)   
    Tx_list = np.zeros(datalength)   
    Ty_list = np.zeros(datalength)
    T_big = np.zeros(datalength)    
    T_small = np.zeros(datalength) 
    Ek_list = np.zeros(datalength)   
    Ekx_list = np.zeros(datalength)   
    Eky_list = np.zeros(datalength)
    Ek_big = np.zeros(datalength)    
    Ek_small = np.zeros(datalength) 
    Tx_big_list = np.zeros(datalength) 
    Ty_big_list = np.zeros(datalength) 
    Tx_small_list = np.zeros(datalength) 
    Ty_small_list = np.zeros(datalength) 
    
    y_wall_data = np.zeros(datalength)
    
    x_data = np.zeros((datalength, Nc_org))
    y_data = np.zeros((datalength, Nc_org))
    vx_data = np.zeros((datalength, Nc_org))
    vy_data = np.zeros((datalength, Nc_org))
    ax_data = np.zeros((datalength, Nc_org))
    ay_data = np.zeros((datalength, Nc_org))
    
    # P_data = np.zeros((datalength, Nc_org))
    # tau_data = np.zeros((datalength, Nc_org))
    # P_Vor_data = np.zeros((datalength, Nc_org))
    # tau_Vor_data = np.zeros((datalength, Nc_org))
    Sxx_data = np.zeros((datalength, Nc_org))
    Sxy_data = np.zeros((datalength, Nc_org))
    Syy_data = np.zeros((datalength, Nc_org))
    area_Voronoi = np.zeros((datalength, Nc_org))
    
    # x_data_plot = np.zeros((datalength_CM, Nc_org))
    # y_data_plot = np.zeros((datalength_CM, Nc_org))
    
    ##### get log extraction set up ######
    N_dt = 6
    log_list = [-1,0,1,2,3,4]
    y_data_proc = []
    dy_data_proc = []
    # x_data_proc = []
    for i in range(0,N_dt):
        y_data_proc.append([0])
        dy_data_proc.append([0])
    
    datalength_vy_temp_MSD = int(N_per_coll * (10 ** log_list[-1]))
    # vy_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    # y_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    count_save_MSD = 0   
    # count_proc_sample = 0
    N_proc_target = 10000
    proc_bw = 2
    ######################################
    
    # clear_flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt2) / 2 
        y = y + vy * dt + ay_old * (dt2) / 2
        
        # x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        # if (gv.size <= 2 and gv == 0):
        #     Fx, Fy = force(Fx, Fy, N_top, x, y, Lx, Ly, K, R_eff)
        # else:
        #     Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            # Fx, Fy = force_rest_print(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            # Fx, Fy, Sxx, Syy, Sxy, P_particle, tau_particle, P_Vor_particle, tau_Vor_particle, A_Vor_particle = force_rest_w_stress(Fx, Fy, N, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            Fx, Fy, Sxx, Syy, Sxy, A_Vor_particle = force_rest_w_stress(Fx, Fy, N, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
        else:
            Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
        #    Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            
        # shear - Couette
        #if (nt % (saveskip * 100) == 0 and plotit):
        #    print((P_plate + g * y))
        Fx = Fx - ((P_plate + g * y) * settling_time * (vx - (gamma_dot * y)))        
        
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw      
        
        #############
        ax = Fx / m
        ay = Fy / m - g
        
        ax[iib] = 0.0
        
        # Bottom wall
        ax[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        ay[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle            
        
        # Top wall
        ax[N_bottom:N] = np.mean(ax[N_bottom:N])
        ay[N_bottom:N] = np.mean(ay[N_bottom:N])                
        
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2

        vx[iib] = 0.0
        vx[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        vy[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        
        ax_old = ax
        ay_old = ay
        
        if (nt > 0.80 * Nt): # take only the last 20% of data
            # vy_data_temp[count_save_MSD] = vy[0:Nc_org] # interval between data is exactly dt
            # y_data_temp[count_save_MSD] = y[0:Nc_org]
            count_save_MSD = count_save_MSD + 1
            for i in range(0, N_dt): # total number of exponents is N_dt
                if (nt % (N_per_coll * (10 ** log_list[i])) == 0):
                    if (len(y_data_proc[i]) - 1 < N_proc_target + 1): # -1 is to count the first 0, +1 is to get N_proc_target 
                        # accessing this section indicates that save data is executed
                        y_data_proc[i].append(y[0:Nc_org])
                        # x_data_proc[i].append(x[0:Nc_org])
                        # Need better way to get average data on the fly
                        # When calculating the average, take (20 * (10 ** log_list[i])) + 1 
                        # Step 1: get all the data points - DATA[count_save_MSD - (20 * (10 ** log_list[i])) : count_save_MSD - 1]
                        #bnlmt = int(np.amax(y_wall_data) * 1.1)
                        #if (np.mod(bnlmt, 2) == 1):
                        #    bnlmt = bnlmt + 1 # to ensure even number
                        #range_high = int(count_save_MSD)
                        #range_low = int(count_save_MSD - (N_per_coll * (10 ** log_list[i])))  
                        #dy_mean_temp = preproc_2D_vs_simple(vy_data_temp[range_low:range_high], m, y_data_temp[range_low:range_high], bnlmt, proc_bw)
                        #dy_mean_temp = np.trapz(dy_mean_temp, x = None, dx = dt)
                        # Step 2: save the binned average velocity that will be used later in the correction of Diff.
                        #dy_data_proc[i].append(dy_mean_temp)
                        #if (i == (N_dt - 1)): # largest set has been calculated                            
                        #    count_save_MSD = 0
        if (nt % saveskip == 0):
            y_CM = y[0:Nc_org]
            CM_big[count1] = np.mean(y_CM[ind_big])
            CM_small[count1] = np.mean(y_CM[~ind_big])
            
            # x_data_plot[count1] = x[0:Nc_org] 
            # y_data_plot[count1] = y[0:Nc_org] 
            
            count1 = count1 + 1
        
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            y_CM = y[0:Nc_org]
            # m_CM = m[0:Nc_org]
            # m_temp = m[0:Nc_org]
            
            vx_temp = vx[0:Nc_org] 
            vy_temp = vy[0:Nc_org] 
            
            # d_acc = np.sqrt(np.amax(ax[0:Nc_org] * ax[0:Nc_org] + ay[0:Nc_org] * ay[0:Nc_org]))
            
            Ek_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2 + (vy_temp - np.mean(vy_temp)) ** 2))
            Ekx_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2))
            Eky_temp = (0.5 * m_temp * ((vy_temp - np.mean(vy_temp)) ** 2))
            Ek_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 + (vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Ek_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 + (vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
            Ekx_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 ))
            Ekx_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 ))
            Eky_big_temp = (0.5 * m_temp[ind_big] * ((vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Eky_small_temp = (0.5 * m_temp[~ind_big] * ((vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
                        
            T_temp = sum(Ek_temp) / Nc_org
            Tx_temp = sum(Ekx_temp) / Nc_org
            Ty_temp = sum(Eky_temp) / Nc_org
            T_big_temp = sum(Ek_big_temp) / Nbig
            T_small_temp = sum(Ek_small_temp) / Nsmall            
            Tx_big_temp = sum(Ekx_big_temp) / Nbig
            Tx_small_temp = sum(Ekx_small_temp) / Nsmall
            Ty_big_temp = sum(Eky_big_temp) / Nbig            
            Ty_small_temp = sum(Eky_small_temp) / Nsmall
            
            T_list[count] = T_temp
            Tx_list[count] = Tx_temp
            Ty_list[count] = Ty_temp
            T_big[count] = T_big_temp
            T_small[count] = T_small_temp
            Tx_big_list[count] = Tx_big_temp
            Ty_big_list[count] = Ty_big_temp
            Tx_small_list[count] = Tx_small_temp
            Ty_small_list[count] = Ty_small_temp
            
            Ek_list[count] = sum(Ek_temp)
            Ekx_list[count] = sum(Ekx_temp)
            Eky_list[count] = sum(Eky_temp)            
            Ek_big[count] = sum(Ek_big_temp)
            Ek_small[count] = sum(Ek_small_temp)
            
            ph_list[count] = Vol_temp_sum / (Lx * y[-1] - Vol_wall_sum)
            
            y_wall_data[count] = y[-1]
            
            x_data[count] = x[0:Nc_org] 
            y_data[count] = y_CM
            vx_data[count] = vx_temp
            vy_data[count] = vy_temp
            ax_data[count] = ax[0:Nc_org]
            ay_data[count] = ay[0:Nc_org]
            
            # P_data[count] = P_particle
            # tau_data[count] = tau_particle
            # P_Vor_data[count] = P_Vor_particle
            # tau_Vor_data[count] = tau_Vor_particle
            Sxx_data[count] = Sxx
            Sxy_data[count] = Sxy
            Syy_data[count] = Syy
            area_Voronoi[count] = A_Vor_particle
            
            if (nt % (saveskip * 100) == 0 and plotit):
                print(nt, T_temp, T_big[count] / T_small[count], Tx_temp / Ty_temp)
            if (nt % (saveskip * 100) == 0):
                print(nt / Nt)

            count = count + 1
            
        nt = nt + 1
    
    for i in range(0, N_dt):
        del y_data_proc[i][0]
        del dy_data_proc[i][0]
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, ax_data, ay_data, dy_data_proc, y_data_proc, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list, Sxx_data, Sxy_data, Syy_data, area_Voronoi
    # return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, dy_data_proc, y_data_proc, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list, x_data_plot, y_data_plot

def couette_flow_w_spring_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_bottom, N_top, Nc_org, R_eff, K, B, gamma_dot, P_plate, gv, a_thresh, v_thresh, N_per_coll, save_reduction, y0, plotit, Vor_flag):
    # Velocity Verlet
    ind_big = np.asarray(ind_big, dtype=bool)
    ind_big_single = np.argwhere(ind_big==1)[0]
    ind_spring = np.append(ind_big, np.zeros(N - Nc_org, dtype=bool))
    Vol = Dn * Dn * np.pi / 4
    Vol_temp = Vol[0:Nc_org]
    Vol_temp_sum = sum(Vol_temp)
    Vol_wall_sum = sum(Vol[Nc_org:N] / 2)
    m_temp = m[0:Nc_org]
    # m_temp_sum = sum(m_temp)
    Nbig = sum(ind_big)
    Nsmall = Nc_org - Nbig
    # dt_half = np.sqrt(dt)
    dt2 = dt * dt
    g = 0.1
    nt = 0
    settling_time = np.sqrt(Dn / g)
    K_spring = K / 10000
    saveskip = N_per_coll * 100 # corresponds to 100 tau_c, dt = tau_c/20
    # saveskip_pos = N_per_coll * 5 # corresponds to 5 tau_c    
    # save_reduction = 0.50 # saving only last 50% of the data
    count = 0
    count1 = 0
    datalength = int((1 - save_reduction) * Nt / saveskip)
    datalength_CM = int(Nt / saveskip)
    #acc_list = np.zeros(datalength)
    CM_big = np.zeros(datalength_CM)
    CM_small = np.zeros(datalength_CM)
    ph_list = np.zeros(datalength)
    T_list = np.zeros(datalength)   
    Tx_list = np.zeros(datalength)   
    Ty_list = np.zeros(datalength)
    T_big = np.zeros(datalength)    
    T_small = np.zeros(datalength) 
    Ek_list = np.zeros(datalength)   
    Ekx_list = np.zeros(datalength)   
    Eky_list = np.zeros(datalength)
    Ek_big = np.zeros(datalength)    
    Ek_small = np.zeros(datalength) 
    Tx_big_list = np.zeros(datalength) 
    Ty_big_list = np.zeros(datalength) 
    Tx_small_list = np.zeros(datalength) 
    Ty_small_list = np.zeros(datalength) 
    
    y_wall_data = np.zeros(datalength)
    
    x_data = np.zeros((datalength, Nc_org))
    y_data = np.zeros((datalength, Nc_org))
    vx_data = np.zeros((datalength, Nc_org))
    vy_data = np.zeros((datalength, Nc_org))
    ax_data = np.zeros((datalength))
    ay_data = np.zeros((datalength))
    
    # P_data = np.zeros((datalength, Nc_org))
    # tau_data = np.zeros((datalength, Nc_org))
    # P_Vor_data = np.zeros((datalength, Nc_org))
    # tau_Vor_data = np.zeros((datalength, Nc_org))
    Sxx_data = np.zeros((datalength, Nc_org))
    Sxy_data = np.zeros((datalength, Nc_org))
    Syy_data = np.zeros((datalength, Nc_org))
    area_Voronoi = np.zeros((datalength, Nc_org))
    
    # x_data_plot = np.zeros((datalength_CM, Nc_org))
    # y_data_plot = np.zeros((datalength_CM, Nc_org))
    
    ##### get log extraction set up ######
    N_dt = 6
    log_list = [-1,0,1,2,3,4]
    y_data_proc = []
    dy_data_proc = []
    # x_data_proc = []
    for i in range(0,N_dt):
        y_data_proc.append([0])
        dy_data_proc.append([0])
    
    datalength_vy_temp_MSD = int(N_per_coll * (10 ** log_list[-1]))
    # vy_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    # y_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    count_save_MSD = 0   
    # count_proc_sample = 0
    N_proc_target = 10000
    proc_bw = 2
    ######################################
    
    # clear_flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt2) / 2 
        y = y + vy * dt + ay_old * (dt2) / 2
        
        # x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        # if (gv.size <= 2 and gv == 0):
        #     Fx, Fy = force(Fx, Fy, N_top, x, y, Lx, Ly, K, R_eff)
        # else:
        #     Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            # Fx, Fy = force_rest_print(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            # Fx, Fy, Sxx, Syy, Sxy, P_particle, tau_particle, P_Vor_particle, tau_Vor_particle, A_Vor_particle = force_rest_w_stress(Fx, Fy, N, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            Fx, Fy, Sxx, Syy, Sxy, A_Vor_particle = force_rest_w_stress(Fx, Fy, N, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv, Vor_flag)
        else:
            Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
        #    Fx, Fy = force_rest(Fx, Fy, N_top, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            
        # shear - Couette
        #if (nt % (saveskip * 100) == 0 and plotit):
        #    print((P_plate + g * y))
        Fx = Fx - ((P_plate + g * y) * settling_time * (vx - (gamma_dot * y)))
        
        Fx_seg = Fx[ind_big_single]
        Fy_seg = Fy[ind_big_single]
        
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw      
        
        # Fake spring attaching to the single large particle        
        dw = y - y0
        Fy_fake_spring = K_spring * dw
        Fy[ind_spring] = Fy[ind_spring] - Fy_fake_spring[ind_spring]
        
        #############
        ax = Fx / m
        ay = Fy / m - g
        
        ax[iib] = 0.0
        
        # Bottom wall
        ax[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        ay[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle            
        
        # Top wall
        ax[N_bottom:N] = np.mean(ax[N_bottom:N])
        ay[N_bottom:N] = np.mean(ay[N_bottom:N])                
        
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2

        vx[iib] = 0.0
        vx[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        vy[Nc_org:N_bottom] = 0.0 # fixed the bottom wall particle
        
        ax_old = ax
        ay_old = ay
        
        if (nt > 0.80 * Nt): # take only the last 20% of data
            # vy_data_temp[count_save_MSD] = vy[0:Nc_org] # interval between data is exactly dt
            # y_data_temp[count_save_MSD] = y[0:Nc_org]
            count_save_MSD = count_save_MSD + 1
            for i in range(0, N_dt): # total number of exponents is N_dt
                if (nt % (N_per_coll * (10 ** log_list[i])) == 0):
                    if (len(y_data_proc[i]) - 1 < N_proc_target + 1): # -1 is to count the first 0, +1 is to get N_proc_target 
                        # accessing this section indicates that save data is executed
                        y_data_proc[i].append(y[0:Nc_org])
                        # x_data_proc[i].append(x[0:Nc_org])
                        # Need better way to get average data on the fly
                        # When calculating the average, take (20 * (10 ** log_list[i])) + 1 
                        # Step 1: get all the data points - DATA[count_save_MSD - (20 * (10 ** log_list[i])) : count_save_MSD - 1]
                        #bnlmt = int(np.amax(y_wall_data) * 1.1)
                        #if (np.mod(bnlmt, 2) == 1):
                        #    bnlmt = bnlmt + 1 # to ensure even number
                        #range_high = int(count_save_MSD)
                        #range_low = int(count_save_MSD - (N_per_coll * (10 ** log_list[i])))  
                        #dy_mean_temp = preproc_2D_vs_simple(vy_data_temp[range_low:range_high], m, y_data_temp[range_low:range_high], bnlmt, proc_bw)
                        #dy_mean_temp = np.trapz(dy_mean_temp, x = None, dx = dt)
                        # Step 2: save the binned average velocity that will be used later in the correction of Diff.
                        #dy_data_proc[i].append(dy_mean_temp)
                        #if (i == (N_dt - 1)): # largest set has been calculated                            
                        #    count_save_MSD = 0
        if (nt % saveskip == 0):
            y_CM = y[0:Nc_org]
            CM_big[count1] = np.mean(y_CM[ind_big])
            CM_small[count1] = np.mean(y_CM[~ind_big])
            
            # x_data_plot[count1] = x[0:Nc_org] 
            # y_data_plot[count1] = y[0:Nc_org] 
            
            count1 = count1 + 1
        
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            y_CM = y[0:Nc_org]
            # m_CM = m[0:Nc_org]
            # m_temp = m[0:Nc_org]
            
            vx_temp = vx[0:Nc_org] 
            vy_temp = vy[0:Nc_org] 
            
            # d_acc = np.sqrt(np.amax(ax[0:Nc_org] * ax[0:Nc_org] + ay[0:Nc_org] * ay[0:Nc_org]))
            
            Ek_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2 + (vy_temp - np.mean(vy_temp)) ** 2))
            Ekx_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2))
            Eky_temp = (0.5 * m_temp * ((vy_temp - np.mean(vy_temp)) ** 2))
            Ek_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 + (vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Ek_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 + (vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
            Ekx_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 ))
            Ekx_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 ))
            Eky_big_temp = (0.5 * m_temp[ind_big] * ((vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Eky_small_temp = (0.5 * m_temp[~ind_big] * ((vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
                        
            T_temp = sum(Ek_temp) / Nc_org
            Tx_temp = sum(Ekx_temp) / Nc_org
            Ty_temp = sum(Eky_temp) / Nc_org
            T_big_temp = sum(Ek_big_temp) / Nbig
            T_small_temp = sum(Ek_small_temp) / Nsmall            
            Tx_big_temp = sum(Ekx_big_temp) / Nbig
            Tx_small_temp = sum(Ekx_small_temp) / Nsmall
            Ty_big_temp = sum(Eky_big_temp) / Nbig            
            Ty_small_temp = sum(Eky_small_temp) / Nsmall
            
            T_list[count] = T_temp
            Tx_list[count] = Tx_temp
            Ty_list[count] = Ty_temp
            T_big[count] = T_big_temp
            T_small[count] = T_small_temp
            Tx_big_list[count] = Tx_big_temp
            Ty_big_list[count] = Ty_big_temp
            Tx_small_list[count] = Tx_small_temp
            Ty_small_list[count] = Ty_small_temp
            
            Ek_list[count] = sum(Ek_temp)
            Ekx_list[count] = sum(Ekx_temp)
            Eky_list[count] = sum(Eky_temp)            
            Ek_big[count] = sum(Ek_big_temp)
            Ek_small[count] = sum(Ek_small_temp)
            
            ph_list[count] = Vol_temp_sum / (Lx * y[-1] - Vol_wall_sum)
            
            y_wall_data[count] = y[-1]
            
            x_data[count] = x[0:Nc_org] 
            y_data[count] = y_CM
            vx_data[count] = vx_temp
            vy_data[count] = vy_temp
            ax_data[count] = Fx_seg / m[ind_big_single]
            ay_data[count] = Fy_seg / m[ind_big_single]
            
            # P_data[count] = P_particle
            # tau_data[count] = tau_particle
            # P_Vor_data[count] = P_Vor_particle
            # tau_Vor_data[count] = tau_Vor_particle
            Sxx_data[count] = Sxx
            Sxy_data[count] = Sxy
            Syy_data[count] = Syy
            area_Voronoi[count] = A_Vor_particle
            
            if (nt % (saveskip * 100) == 0 and plotit):
                print(nt, T_temp, T_big[count] / T_small[count], Tx_temp / Ty_temp)
            if (nt % (saveskip * 100) == 0):
                print(nt / Nt)

            count = count + 1
            
        nt = nt + 1
    
    for i in range(0, N_dt):
        del y_data_proc[i][0]
        del dy_data_proc[i][0]
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, ax_data, ay_data, dy_data_proc, y_data_proc, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list, Sxx_data, Sxy_data, Syy_data, area_Voronoi
    # return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, dy_data_proc, y_data_proc, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list, x_data_plot, y_data_plot


#@jit
def langevin_thermostat_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, Nc_org, R_eff, K, B, Tamp, gv, a_thresh, v_thresh, Tswitch, Theta, N_per_coll, plotit):
    # Velocity Verlet
    # Tswitch=0, balanced. Tswitch=1, large in x+y
    # Tswitch=2, large in x, Tswitch=3, large in y
    Vol = Dn * Dn * np.pi / 4
    ind_big = np.asarray(ind_big, dtype=bool)
    Nbig = sum(ind_big)
    Nsmall = Nc_org - Nbig
    
    ##### Make T0 list #####
    T0 = Tamp
    Tamp = np.zeros(Nc_org, dtype=np.float64)    
    if (Tswitch == 0):
        for i in range (0, Nc_org):
            Tamp[i] = T0
    elif (Tswitch != 0): # kick only the large particles, whether it's x and/or y
        for i in range (0, Nc_org):
            if ind_big[i]:
                Tamp[i] = T0
            else:
                Tamp[i] = 0
    if (Tswitch == 4):
        Tl = Theta * T0
        Ts = T0
        for i in range (0, Nc_org):
            if ind_big[i]:
                Tamp[i] = Tl
            else:
                Tamp[i] = Ts
        
    Tamp = np.append(Tamp, 0)
    T0 = Tamp
    #######################
    
    epsilon = B / m # unit: 1/time
    dt_half = np.sqrt(dt)
    dt2 = dt * dt
    g = 0.1
    nt = 0
    saveskip = N_per_coll * 20 # corresponds to 20 tau_c, dt = tau_c/50
    # saveskip_pos = N_per_coll * 2 # corresponds to 2 tau_c
    save_reduction = 0.50 # saving only last 50% of the data
    count = 0
    count1 = 0
    #datalength = int(Nt / saveskip)
    datalength_CM = int(Nt / saveskip)
    datalength = int(save_reduction * Nt / saveskip) - 1
    # datalength_pos = int(save_reduction * Nt / saveskip_pos)
    # acc_list = np.zeros(datalength)
    CM_big = np.zeros(datalength_CM)
    CM_small = np.zeros(datalength_CM)
    Ek_big = np.zeros(datalength)
    Ek_small = np.zeros(datalength)
    ph_list = np.zeros(datalength)
    T_list = np.zeros(datalength)   
    Tx_list = np.zeros(datalength)   
    Ty_list = np.zeros(datalength)
    T_big = np.zeros(datalength)    
    T_small = np.zeros(datalength) 
    Ek_list = np.zeros(datalength)   
    Ekx_list = np.zeros(datalength)   
    Eky_list = np.zeros(datalength)
    Ek_big = np.zeros(datalength)    
    Ek_small = np.zeros(datalength) 
    Tx_big_list = np.zeros(datalength) 
    Ty_big_list = np.zeros(datalength) 
    Tx_small_list = np.zeros(datalength) 
    Ty_small_list = np.zeros(datalength) 
    
    y_wall_data = np.zeros(datalength)
    
    x_data = np.zeros((datalength, Nc_org))
    y_data = np.zeros((datalength, Nc_org))
    vx_data = np.zeros((datalength, Nc_org))
    vy_data = np.zeros((datalength, Nc_org))    
    ay_data = np.zeros((datalength, Nc_org))
    
    x_data_plot = np.zeros((datalength_CM, Nc_org))
    y_data_plot = np.zeros((datalength_CM, Nc_org))
    
    ##### get log extraction set up ######
    N_dt = 6
    log_list = [-1,0,1,2,3,4]
    y_data_proc = []
    dy_data_proc = []
    # x_data_proc = []
    for i in range(0,N_dt):
        y_data_proc.append([0])
        dy_data_proc.append([0])
    datalength_vy_temp_MSD = int(N_per_coll * (10 ** log_list[-1]))
    #vy_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    #y_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    #count_save_MSD = 0           
    # count_proc_sample = 0
    N_proc_target = 10000
    proc_bw = 2
    ######################################
    #xdata_MSD = []
    #ydata_MSD = []        
    
    vx = vx - np.mean(vx)
    vy = vy - np.mean(vy)
    
    m_temp = m[0:Nc_org]
    # m_temp_sum = sum(m_temp)
    Vol_temp = Vol[0:Nc_org]
    Vol_temp_sum = sum(Vol_temp)
    # flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt2) / 2 
        y = y + vy * dt + ay_old * (dt2) / 2
        
        # x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        if (gv == 0):
            Fx, Fy = force(Fx, Fy, Nc_org, x, y, Lx, Ly, K, R_eff)
        else:
            Fx, Fy = force_rest(Fx, Fy, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
        
        # top wall
        iit = (y + Dn/2) > y[-1]
        iit[-1] = 0
        dw = y[iit] + Dn[iit]/2 - y[-1]
        Fy[iit] = Fy[iit] - K * dw
        Fy[-1] = np.sum(K * dw)
        
        # top wall
        #iit = (y > Ly - Dn/2)
        #dw = y[iit] - (Ly - Dn[iit] / 2)
        #Fy[iit] = Fy[iit] - K * dw
        
        # left wall
        #iil = (x < Dn/2)
        #dw = x[iil] - Dn[iil] / 2
        #Fx[iil] = Fx[iil] - K * dw
        
        # right wall
        #iir = (x > (Lx - Dn/2))
        #dw = x[iir] - (Lx - (Dn[iir] / 2))
        #Fx[iir] = Fx[iir] - K * dw        
        
        #############
        if (Tswitch == 0 or Tswitch == 1 or Tswitch == 4):
            sigma = np.sqrt(2 * T0 * epsilon / m)
            ax = (Fx / m - epsilon * (vx - ax_old *(dt/2)) + sigma * np.random.randn(N) * (1/dt_half)) / (1 + epsilon * (dt/2))
            ay = (Fy / m - epsilon * (vy - ay_old *(dt/2)) + sigma * np.random.randn(N) * (1/dt_half) - g) / (1 + epsilon * (dt/2))
        elif (Tswitch == 2):
            sigmax = np.sqrt(2 * T0 * epsilon / m)
            sigmay = np.sqrt(2 * 0 * epsilon / m)
            ax = (Fx / m - epsilon * (vx - ax_old *(dt/2)) + sigmax * np.random.randn(N) * (1/dt_half)) / (1 + epsilon * (dt/2))
            ay = (Fy / m - epsilon * (vy - ay_old *(dt/2)) + sigmay * np.random.randn(N) * (1/dt_half) - g) / (1 + epsilon * (dt/2))
        elif (Tswitch == 3):
            sigmax = np.sqrt(2 * 0 * epsilon / m)
            sigmay = np.sqrt(2 * T0 * epsilon / m)
            ax = (Fx / m - epsilon * (vx - ax_old *(dt/2)) + sigmax * np.random.randn(N) * (1/dt_half)) / (1 + epsilon * (dt/2))
            ay = (Fy / m - epsilon * (vy - ay_old *(dt/2)) + sigmay * np.random.randn(N) * (1/dt_half) - g) / (1 + epsilon * (dt/2))

        
        ay[-1] = Fy[-1] / m[-1] - g # this is fine, no double count of forces on the top wall
        
        ax[-1] = 0.0
        
        #ax[iib] = 0.0
        #if (BottomWallTag):
        #    ax[N_org:N] = 0.0 # fixed the bottom wall particle
        #    ay[N_org:N] = 0.0 # fixed the bottom wall particle            
        
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2
        
        vx[-1] = 0.0
        
        #vx[iib] = 0.0
        #if (BottomWallTag):
        #    vx[N_org:N] = 0.0 # fixed the bottom wall particle
        #    vy[N_org:N] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay

        if (nt > 0.70 * Nt): # take only the last 30% of data
            for i in range(0, N_dt): # total number of exponents is N_dt
                if (nt % (N_per_coll * (10 ** log_list[i])) == 0):
                    if (len(y_data_proc[i]) - 1 < N_proc_target + 1): # -1 is to count the first 0, +1 is to get N_proc_target                    
                        y_data_proc[i].append(y[0:Nc_org])
                        # x_data_proc[i].append(x[0:Nc_org])
                        #bnlmt = int(np.amax(y_wall_data) * 1.1)
                        #if (np.mod(bnlmt, 2) == 1):
                        #    bnlmt = bnlmt + 1 # to ensure even number
                        #range_high = int(count_save_MSD)
                        #range_low = int(count_save_MSD - (N_per_coll * (10 ** log_list[i])))                        
                        #dy_mean_temp = preproc_2D_vs_simple(vy_data_temp[range_low:range_high], m, y_data_temp[range_low:range_high], bnlmt, proc_bw)
                        #dy_mean_temp = np.trapz(dy_mean_temp, x = None, dx = dt)                        
                        # Step 2: save the binned average velocity that will be used later in the correction of Diff.
                        #dy_data_proc[i].append(dy_mean_temp)
                        #if (i == (N_dt - 1)): # largest set has been calculated                            
                        #    count_save_MSD = 0

        if (nt % saveskip == 0):
            y_CM = y[0:Nc_org]
            CM_big[count1] = np.mean(y_CM[ind_big])
            CM_small[count1] = np.mean(y_CM[~ind_big])
            
            x_data_plot[count1] = x[0:Nc_org] 
            y_data_plot[count1] = y[0:Nc_org] 
            
            count1 = count1 + 1
                        
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            y_CM = y[0:Nc_org]   
            # m_CM = m[0:Nc_org]
            # m_temp = m[0:Nc_org]
            
            vx_temp = vx[0:Nc_org] 
            vy_temp = vy[0:Nc_org] 
            
            # d_acc = np.sqrt(np.amax(ax[0:Nc_org] * ax[0:Nc_org] + ay[0:Nc_org] * ay[0:Nc_org]))
            
            Ek_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2 + (vy_temp - np.mean(vy_temp)) ** 2))
            Ekx_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2))
            Eky_temp = (0.5 * m_temp * ((vy_temp - np.mean(vy_temp)) ** 2))
            Ek_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 + (vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Ek_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 + (vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
            Ekx_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 ))
            Ekx_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 ))
            Eky_big_temp = (0.5 * m_temp[ind_big] * ((vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Eky_small_temp = (0.5 * m_temp[~ind_big] * ((vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
                        
            T_temp = sum(Ek_temp) / Nc_org
            Tx_temp = sum(Ekx_temp) / Nc_org
            Ty_temp = sum(Eky_temp) / Nc_org
            T_big_temp = sum(Ek_big_temp) / Nbig
            T_small_temp = sum(Ek_small_temp) / Nsmall            
            Tx_big_temp = sum(Ekx_big_temp) / Nbig
            Tx_small_temp = sum(Ekx_small_temp) / Nsmall
            Ty_big_temp = sum(Eky_big_temp) / Nbig            
            Ty_small_temp = sum(Eky_small_temp) / Nsmall
            
            T_list[count] = T_temp
            Tx_list[count] = Tx_temp
            Ty_list[count] = Ty_temp
            T_big[count] = T_big_temp
            T_small[count] = T_small_temp
            Tx_big_list[count] = Tx_big_temp
            Ty_big_list[count] = Ty_big_temp
            Tx_small_list[count] = Tx_small_temp
            Ty_small_list[count] = Ty_small_temp
            
            Ek_list[count] = sum(Ek_temp)
            Ekx_list[count] = sum(Ekx_temp)
            Eky_list[count] = sum(Eky_temp)            
            Ek_big[count] = sum(Ek_big_temp)
            Ek_small[count] = sum(Ek_small_temp)
            
            ph_list[count] = Vol_temp_sum / (Lx * y[-1])
            
            x_data[count] = x[0:Nc_org] 
            y_data[count] = y_CM
            vx_data[count] = vx_temp
            vy_data[count] = vy_temp
            ay_data[count] = ay[0:Nc_org]
            
            y_wall_data[count] = y[-1]
            
            if (nt % (saveskip * 100) == 0 and plotit):
                print(nt, T_temp, T_big[count] / T_small[count], Tx_temp / Ty_temp)
                #acc_list, T_list = make_data_plot(fig, ax, acc_list, T_list, d_acc, T_temp, count) # does not work
            count = count + 1
            
        nt = nt + 1
        
        #if (np.sqrt(np.amax(ax * ax + ay * ay)) < (a_thresh) and np.sqrt(np.amax(vx * vx + vy * vy)) < (v_thresh)):
        #    if (flag == False):
        #        flag = True
        #    else:
        #        #print('gravity MS reached')
        #        break
            
        
    #xdata_MSD = np.asarray(xdata_MSD)
    #ydata_MSD = np.asarray(ydata_MSD)    
    #xdata_MSD = np.transpose(xdata_MSD)
    #ydata_MSD = np.transpose(ydata_MSD)
    
    #MSD_x, MSD_y = get_MSD(xdata_MSD,ydata_MSD)
    
    for i in range(0,N_dt):
        del y_data_proc[i][0]
        del dy_data_proc[i][0]
        
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, dy_data_proc, y_data_proc, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list, x_data_plot, y_data_plot

def langevin_thermostat_w_spring_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, Nc_org, R_eff, K, B, Tamp, gv, a_thresh, v_thresh, Tswitch, Theta, N_per_coll, save_reduction, y0, plotit, Vor_flag):
    # Velocity Verlet
    # Tswitch=0, balanced. Tswitch=1, large in x+y
    # Tswitch=2, large in x, Tswitch=3, large in y
    Vol = Dn * Dn * np.pi / 4
    ind_big = np.asarray(ind_big, dtype=bool)
    ind_big_single = np.argwhere(ind_big==1)[0]
    ind_spring = np.append(ind_big, np.zeros(N - Nc_org, dtype=bool))
    Nbig = sum(ind_big)
    Nsmall = Nc_org - Nbig
    
    ##### Make T0 list #####
    T0 = Tamp
    Tamp = np.zeros(Nc_org, dtype=np.float64)    
    if (Tswitch == 0):
        for i in range (0, Nc_org):
            Tamp[i] = T0
    elif (Tswitch != 0): # kick only the large particles, whether it's x and/or y
        for i in range (0, Nc_org):
            if ind_big[i]:
                Tamp[i] = T0
            else:
                Tamp[i] = 0
    if (Tswitch == 4):
        Tl = Theta * T0
        Ts = T0
        for i in range (0, Nc_org):
            if ind_big[i]:
                Tamp[i] = Tl
            else:
                Tamp[i] = Ts
        
    Tamp = np.append(Tamp, 0)
    T0 = Tamp
    #######################
    
    epsilon = B / m # unit: 1/time
    dt_half = np.sqrt(dt)
    dt2 = dt * dt
    g = 0.1
    nt = 0
    saveskip = N_per_coll * 20 # corresponds to 20 tau_c, dt = tau_c/50
    K_spring = K / 10000
    # saveskip_pos = N_per_coll * 2 # corresponds to 2 tau_c
    # save_reduction = 0.50 # saving only last 50% of the data    
    count = 0
    count1 = 0
    #datalength = int(Nt / saveskip)
    datalength_CM = int(Nt / saveskip)
    datalength = int((1 - save_reduction) * Nt / saveskip)
    # datalength_pos = int(save_reduction * Nt / saveskip_pos)
    # acc_list = np.zeros(datalength)
    CM_big = np.zeros(datalength_CM)
    CM_small = np.zeros(datalength_CM)
    Ek_big = np.zeros(datalength)
    Ek_small = np.zeros(datalength)
    ph_list = np.zeros(datalength)
    T_list = np.zeros(datalength)   
    Tx_list = np.zeros(datalength)   
    Ty_list = np.zeros(datalength)
    T_big = np.zeros(datalength)    
    T_small = np.zeros(datalength) 
    Ek_list = np.zeros(datalength)   
    Ekx_list = np.zeros(datalength)   
    Eky_list = np.zeros(datalength)
    Ek_big = np.zeros(datalength)    
    Ek_small = np.zeros(datalength) 
    Tx_big_list = np.zeros(datalength) 
    Ty_big_list = np.zeros(datalength) 
    Tx_small_list = np.zeros(datalength) 
    Ty_small_list = np.zeros(datalength) 
    
    y_wall_data = np.zeros(datalength)
    
    x_data = np.zeros((datalength, Nc_org))
    y_data = np.zeros((datalength, Nc_org))
    vx_data = np.zeros((datalength, Nc_org))
    vy_data = np.zeros((datalength, Nc_org))    
    ax_data = np.zeros((datalength))
    ay_data = np.zeros((datalength))
    
    # x_data_plot = np.zeros((datalength_CM, Nc_org))
    # y_data_plot = np.zeros((datalength_CM, Nc_org))
    
    Sxx_data = np.zeros((datalength, Nc_org))
    Sxy_data = np.zeros((datalength, Nc_org))
    Syy_data = np.zeros((datalength, Nc_org))
    area_Voronoi = np.zeros((datalength, Nc_org))
    
    ##### get log extraction set up ######
    N_dt = 6
    log_list = [-1,0,1,2,3,4]
    y_data_proc = []
    dy_data_proc = []
    # x_data_proc = []
    for i in range(0,N_dt):
        y_data_proc.append([0])
        dy_data_proc.append([0])
    datalength_vy_temp_MSD = int(N_per_coll * (10 ** log_list[-1]))
    #vy_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    #y_data_temp = np.zeros((datalength_vy_temp_MSD, Nc_org))
    #count_save_MSD = 0           
    # count_proc_sample = 0
    N_proc_target = 10000
    proc_bw = 2
    ######################################
    #xdata_MSD = []
    #ydata_MSD = []        
    
    vx = vx - np.mean(vx)
    vy = vy - np.mean(vy)
    
    m_temp = m[0:Nc_org]
    # m_temp_sum = sum(m_temp)
    Vol_temp = Vol[0:Nc_org]
    Vol_temp_sum = sum(Vol_temp)
    # flag = False
    while (nt < Nt):
        
        x = x + vx * dt + ax_old * (dt2) / 2 
        y = y + vy * dt + ay_old * (dt2) / 2
        
        # x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        #if (gv == 0):
        #    Fx, Fy = force(Fx, Fy, Nc_org, x, y, Lx, Ly, K, R_eff)
        #else:
        #    Fx, Fy = force_rest(Fx, Fy, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
                    
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            # Fx, Fy, Sxx, Syy, Sxy, P_particle, tau_particle, P_Vor_particle, tau_Vor_particle, A_Vor_particle = force_rest_w_stress(Fx, Fy, N, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
            Fx, Fy, Sxx, Syy, Sxy, A_Vor_particle = force_w_stress(Fx, Fy, N, Nc_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, Vor_flag)
        else:
            Fx, Fy = force(Fx, Fy, Nc_org, x, y, Lx, Ly, K, R_eff)
        
        Fx_seg = Fx[ind_big_single]
        Fy_seg = Fy[ind_big_single]
        
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
        
        # top wall
        iit = (y + Dn/2) > y[-1]
        iit[-1] = 0
        dw = y[iit] + Dn[iit]/2 - y[-1]
        Fy[iit] = Fy[iit] - K * dw
        Fy[-1] = np.sum(K * dw)
        
        # Fake spring attaching to the single large particle        
        dw = y - y0
        Fy_fake_spring = K_spring * dw
        Fy[ind_spring] = Fy[ind_spring] - Fy_fake_spring[ind_spring]
        
        # top wall
        #iit = (y > Ly - Dn/2)
        #dw = y[iit] - (Ly - Dn[iit] / 2)
        #Fy[iit] = Fy[iit] - K * dw
        
        # left wall
        #iil = (x < Dn/2)
        #dw = x[iil] - Dn[iil] / 2
        #Fx[iil] = Fx[iil] - K * dw
        
        # right wall
        #iir = (x > (Lx - Dn/2))
        #dw = x[iir] - (Lx - (Dn[iir] / 2))
        #Fx[iir] = Fx[iir] - K * dw        
        
        #############
        if (Tswitch == 0 or Tswitch == 1 or Tswitch == 4):
            sigma = np.sqrt(2 * T0 * epsilon / m)
            ax = (Fx / m - epsilon * (vx - ax_old *(dt/2)) + sigma * np.random.randn(N) * (1/dt_half)) / (1 + epsilon * (dt/2))
            ay = (Fy / m - epsilon * (vy - ay_old *(dt/2)) + sigma * np.random.randn(N) * (1/dt_half) - g) / (1 + epsilon * (dt/2))
        elif (Tswitch == 2):
            sigmax = np.sqrt(2 * T0 * epsilon / m)
            sigmay = np.sqrt(2 * 0 * epsilon / m)
            ax = (Fx / m - epsilon * (vx - ax_old *(dt/2)) + sigmax * np.random.randn(N) * (1/dt_half)) / (1 + epsilon * (dt/2))
            ay = (Fy / m - epsilon * (vy - ay_old *(dt/2)) + sigmay * np.random.randn(N) * (1/dt_half) - g) / (1 + epsilon * (dt/2))
        elif (Tswitch == 3):
            sigmax = np.sqrt(2 * 0 * epsilon / m)
            sigmay = np.sqrt(2 * T0 * epsilon / m)
            ax = (Fx / m - epsilon * (vx - ax_old *(dt/2)) + sigmax * np.random.randn(N) * (1/dt_half)) / (1 + epsilon * (dt/2))
            ay = (Fy / m - epsilon * (vy - ay_old *(dt/2)) + sigmay * np.random.randn(N) * (1/dt_half) - g) / (1 + epsilon * (dt/2))

        
        ay[-1] = Fy[-1] / m[-1] - g # this is fine, no double count of forces on the top wall
        
        ax[-1] = 0.0
        
        #ax[iib] = 0.0
        #if (BottomWallTag):
        #    ax[N_org:N] = 0.0 # fixed the bottom wall particle
        #    ay[N_org:N] = 0.0 # fixed the bottom wall particle            
        
        vx = vx + (ax_old + ax) * dt / 2
        vy = vy + (ay_old + ay) * dt / 2
        
        vx[-1] = 0.0
        
        #vx[iib] = 0.0
        #if (BottomWallTag):
        #    vx[N_org:N] = 0.0 # fixed the bottom wall particle
        #    vy[N_org:N] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay

        if (nt > 0.70 * Nt): # take only the last 30% of data
            for i in range(0, N_dt): # total number of exponents is N_dt
                if (nt % (N_per_coll * (10 ** log_list[i])) == 0):
                    if (len(y_data_proc[i]) - 1 < N_proc_target + 1): # -1 is to count the first 0, +1 is to get N_proc_target                    
                        y_data_proc[i].append(y[0:Nc_org])
                        # x_data_proc[i].append(x[0:Nc_org])
                        #bnlmt = int(np.amax(y_wall_data) * 1.1)
                        #if (np.mod(bnlmt, 2) == 1):
                        #    bnlmt = bnlmt + 1 # to ensure even number
                        #range_high = int(count_save_MSD)
                        #range_low = int(count_save_MSD - (N_per_coll * (10 ** log_list[i])))                        
                        #dy_mean_temp = preproc_2D_vs_simple(vy_data_temp[range_low:range_high], m, y_data_temp[range_low:range_high], bnlmt, proc_bw)
                        #dy_mean_temp = np.trapz(dy_mean_temp, x = None, dx = dt)                        
                        # Step 2: save the binned average velocity that will be used later in the correction of Diff.
                        #dy_data_proc[i].append(dy_mean_temp)
                        #if (i == (N_dt - 1)): # largest set has been calculated                            
                        #    count_save_MSD = 0

        if (nt % saveskip == 0):
            y_CM = y[0:Nc_org]
            CM_big[count1] = np.mean(y_CM[ind_big])
            CM_small[count1] = np.mean(y_CM[~ind_big])
            
            # x_data_plot[count1] = x[0:Nc_org] 
            # y_data_plot[count1] = y[0:Nc_org] 
            
            count1 = count1 + 1
                        
        if (nt > save_reduction * Nt and nt % saveskip == 0):
            y_CM = y[0:Nc_org]   
            # m_CM = m[0:Nc_org]
            # m_temp = m[0:Nc_org]
            
            vx_temp = vx[0:Nc_org] 
            vy_temp = vy[0:Nc_org] 
            
            # d_acc = np.sqrt(np.amax(ax[0:Nc_org] * ax[0:Nc_org] + ay[0:Nc_org] * ay[0:Nc_org]))
            
            Ek_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2 + (vy_temp - np.mean(vy_temp)) ** 2))
            Ekx_temp = (0.5 * m_temp * ((vx_temp - np.mean(vx_temp)) ** 2))
            Eky_temp = (0.5 * m_temp * ((vy_temp - np.mean(vy_temp)) ** 2))
            Ek_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 + (vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Ek_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 + (vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
            Ekx_big_temp = (0.5 * m_temp[ind_big] * ((vx_temp[ind_big] - np.mean(vx_temp[ind_big])) ** 2 ))
            Ekx_small_temp = (0.5 * m_temp[~ind_big] * ((vx_temp[~ind_big] - np.mean(vx_temp[~ind_big])) ** 2 ))
            Eky_big_temp = (0.5 * m_temp[ind_big] * ((vy_temp[ind_big] - np.mean(vy_temp[ind_big])) ** 2))
            Eky_small_temp = (0.5 * m_temp[~ind_big] * ((vy_temp[~ind_big] - np.mean(vy_temp[~ind_big])) ** 2))
                        
            T_temp = sum(Ek_temp) / Nc_org
            Tx_temp = sum(Ekx_temp) / Nc_org
            Ty_temp = sum(Eky_temp) / Nc_org
            T_big_temp = sum(Ek_big_temp) / Nbig
            T_small_temp = sum(Ek_small_temp) / Nsmall            
            Tx_big_temp = sum(Ekx_big_temp) / Nbig
            Tx_small_temp = sum(Ekx_small_temp) / Nsmall
            Ty_big_temp = sum(Eky_big_temp) / Nbig            
            Ty_small_temp = sum(Eky_small_temp) / Nsmall
            
            T_list[count] = T_temp
            Tx_list[count] = Tx_temp
            Ty_list[count] = Ty_temp
            T_big[count] = T_big_temp
            T_small[count] = T_small_temp
            Tx_big_list[count] = Tx_big_temp
            Ty_big_list[count] = Ty_big_temp
            Tx_small_list[count] = Tx_small_temp
            Ty_small_list[count] = Ty_small_temp
            
            Ek_list[count] = sum(Ek_temp)
            Ekx_list[count] = sum(Ekx_temp)
            Eky_list[count] = sum(Eky_temp)            
            Ek_big[count] = sum(Ek_big_temp)
            Ek_small[count] = sum(Ek_small_temp)
            
            ph_list[count] = Vol_temp_sum / (Lx * y[-1])
            
            x_data[count] = x[0:Nc_org] 
            y_data[count] = y_CM
            vx_data[count] = vx_temp
            vy_data[count] = vy_temp
            ax_data[count] = Fx_seg / m[ind_big_single]
            ay_data[count] = Fy_seg / m[ind_big_single]
            
            Sxx_data[count] = Sxx
            Sxy_data[count] = Sxy
            Syy_data[count] = Syy
            area_Voronoi[count] = A_Vor_particle
            
            y_wall_data[count] = y[-1]
            
            if (nt % (saveskip * 500) == 0 and plotit):
                print(nt / Nt, T_temp, T_big[count] / T_small[count], Tx_temp / Ty_temp)
                # draw_particle(x[0:Nc_org], y[0:Nc_org], Lx, Nc_org, R_eff[0:Nc_org])
                #acc_list, T_list = make_data_plot(fig, ax, acc_list, T_list, d_acc, T_temp, count) # does not work
            count = count + 1
            
        nt = nt + 1
        
        #if (np.sqrt(np.amax(ax * ax + ay * ay)) < (a_thresh) and np.sqrt(np.amax(vx * vx + vy * vy)) < (v_thresh)):
        #    if (flag == False):
        #        flag = True
        #    else:
        #        #print('gravity MS reached')
        #        break
            
        
    #xdata_MSD = np.asarray(xdata_MSD)
    #ydata_MSD = np.asarray(ydata_MSD)    
    #xdata_MSD = np.transpose(xdata_MSD)
    #ydata_MSD = np.transpose(ydata_MSD)
    
    #MSD_x, MSD_y = get_MSD(xdata_MSD,ydata_MSD)
    
    for i in range(0,N_dt):
        del y_data_proc[i][0]
        del dy_data_proc[i][0]
        
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, ax_data, ay_data, dy_data_proc, y_data_proc, y_wall_data, ph_list, T_list, Tx_list, Ty_list, T_big, T_small, Ek_list, Ek_big, Ek_small, Ekx_list, Eky_list, Tx_big_list, Ty_big_list, Tx_small_list, Ty_small_list, Sxx_data, Sxy_data, Syy_data, area_Voronoi

@jit
def force(Fx, Fy, N, x, y, Lx, Ly, K, R_eff):
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx            
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                dnm2 = (dx * dx) + (dy * dy)
                dnm = np.sqrt(dnm2)
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)
                    Fx[nn] = Fx[nn] + F * dx
                    Fx[mm] = Fx[mm] - F * dx
                    Fy[nn] = Fy[nn] + F * dy
                    Fy[mm] = Fy[mm] - F * dy
    return Fx, Fy

@jit
def force_w_stress(Fx, Fy, N, N_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, Vor_flag):
    Sxx = np.zeros(N_org)
    Sxy = np.zeros(N_org)
    Syy = np.zeros(N_org)
    area_Voronoi = []
    if Vor_flag:
        area_Voronoi = get_Voronoi_area(x, y, Lx, Ly, N)
        area_Voronoi = area_Voronoi[0:N_org]
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx            
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                dnm2 = (dx * dx) + (dy * dy)
                dnm = np.sqrt(dnm2)
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)                  
                    Fx[nn] = Fx[nn] + F * dx
                    Fx[mm] = Fx[mm] - F * dx
                    Fy[nn] = Fy[nn] + F * dy
                    Fy[mm] = Fy[mm] - F * dy                 
                    # contact streess partitioning is applied following the 
                    # mixture theory of Thornton et. al. and many other groups.
                    partnn = R_eff[mm] / (R_eff[nn] + R_eff[mm])
                    partmm = R_eff[nn] / (R_eff[nn] + R_eff[mm])
                    if nn < N_org:
                        Sxx[nn] = Sxx[nn] - partnn * (F * dx * dx)
                        Syy[nn] = Syy[nn] - partnn * (F * dy * dy)
                        Sxy[nn] = Sxy[nn] - partnn * (F * dx * dy)
                    if mm < N_org:
                        Sxx[mm] = Sxx[mm] - partmm * (F * dx * dx)
                        Syy[mm] = Syy[mm] - partmm * (F * dy * dy)
                        Sxy[mm] = Sxy[mm] - partmm * (F * dx * dy)
    if not Vor_flag:
        area_Voronoi = np.zeros(N_org)
    return Fx, Fy, Sxx, Syy, Sxy, area_Voronoi

#@cuda.jit
@jit
def force_rest(Fx, Fy, N, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv):
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx
            
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                dnm2 = (dx * dx) + (dy * dy)
                dnm = np.sqrt(dnm2)
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)
                    m_red = m[nn] * m[mm] / (m[nn] + m[mm])
                    v_dot_r = ((vx[nn]-vx[mm]) * dx + (vy[nn]-vy[mm]) * dy)
                    Nx = gv * m_red * v_dot_r * dx / dnm2
                    Ny = gv * m_red * v_dot_r * dy / dnm2                    
                    Fx[nn] = Fx[nn] + F * dx - Nx
                    Fx[mm] = Fx[mm] - F * dx + Nx
                    Fy[nn] = Fy[nn] + F * dy - Ny
                    Fy[mm] = Fy[mm] - F * dy + Ny
    return Fx, Fy

@jit
def force_rest_print(Fx, Fy, N, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv):
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx
            
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                dnm2 = (dx * dx) + (dy * dy)
                dnm = np.sqrt(dnm2)
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)
                    m_red = m[nn] * m[mm] / (m[nn] + m[mm])
                    v_dot_r = ((vx[nn]-vx[mm]) * dx + (vy[nn]-vy[mm]) * dy)
                    Nx = gv * m_red * v_dot_r * dx / dnm2
                    Ny = gv * m_red * v_dot_r * dy / dnm2
                    print(F * dx, Nx, F * dy, Ny)
                    Fx[nn] = Fx[nn] + F * dx - Nx
                    Fx[mm] = Fx[mm] - F * dx + Nx
                    Fy[nn] = Fy[nn] + F * dy - Ny
                    Fy[mm] = Fy[mm] - F * dy + Ny
    return Fx, Fy

@jit
def force_rest_w_stress(Fx, Fy, N, N_org, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv, Vor_flag):
    Sxx = np.zeros(N_org)
    Sxy = np.zeros(N_org)
    Syy = np.zeros(N_org)
    area_Voronoi = []
    if Vor_flag:
        area_Voronoi = get_Voronoi_area(x, y, Lx, Ly, N)
        area_Voronoi = area_Voronoi[0:N_org]
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx
            
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                dnm2 = (dx * dx) + (dy * dy)
                dnm = np.sqrt(dnm2)
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)
                    m_red = m[nn] * m[mm] / (m[nn] + m[mm])
                    v_dot_r = ((vx[nn]-vx[mm]) * dx + (vy[nn]-vy[mm]) * dy)
                    Nx = gv * m_red * v_dot_r * dx / dnm2
                    Ny = gv * m_red * v_dot_r * dy / dnm2                    
                    Fx[nn] = Fx[nn] + F * dx - Nx
                    Fx[mm] = Fx[mm] - F * dx + Nx
                    Fy[nn] = Fy[nn] + F * dy - Ny
                    Fy[mm] = Fy[mm] - F * dy + Ny                    
                    # contact streess partitioning is applied following the 
                    # mixture theory of Thornton et. al. and many other groups.
                    partnn = R_eff[mm] / (R_eff[nn] + R_eff[mm])
                    partmm = R_eff[nn] / (R_eff[nn] + R_eff[mm])
                    if nn < N_org:
                        Sxx[nn] = Sxx[nn] - partnn * (F * dx * dx)
                        Syy[nn] = Syy[nn] - partnn * (F * dy * dy)
                        Sxy[nn] = Sxy[nn] - partnn * (F * dx * dy)
                    if mm < N_org:
                        Sxx[mm] = Sxx[mm] - partmm * (F * dx * dx)
                        Syy[mm] = Syy[mm] - partmm * (F * dy * dy)
                        Sxy[mm] = Sxy[mm] - partmm * (F * dx * dy)
    if not Vor_flag:
        area_Voronoi = np.zeros(N_org)
    return Fx, Fy, Sxx, Syy, Sxy, area_Voronoi

@jit
def get_Voronoi_area(x, y, Lx, Ly, N):
    x = np.mod(x, Lx)
    y = np.mod(y, Ly)
    x_input = x
    y_input = y
    x = np.append(x_input, x_input - Lx)
    x = np.append(x, x_input + Lx)
    y = np.append(y_input, y_input)
    y = np.append(y, y_input)
    pair = list(zip(x,y))
    vor = Voronoi(pair)
    vertices = vor.vertices
    regions = vor.regions
    point_region = vor.point_region
    area = np.zeros(N);
    for i in range (0, N):
        ind = point_region[i]
        pair_0 = vertices[regions[ind]]
        hull = ConvexHull(pair_0)
        area[i] = hull.area
    return area

def draw_particle(x, y, Lx, N, R_eff):
    x_disp = np.mod(x, Lx)
    
    xy = np.zeros((N, 2))
    for i in range(N):
        xy[i][0] = x_disp[i]
        xy[i][1] = y[i]
    
    patches = [plt.Circle(center, size) for center, size in zip(xy, R_eff)]
    fig, ax = plt.subplots()
    coll = matplotlib.collections.PatchCollection(patches, edgecolors='black')
    ax.add_collection(coll)

    ax.margins(0.01)
    plt.axis('equal')
    plt.show()
    return