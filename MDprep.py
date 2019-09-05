# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:47:14 2018

@author: Philip
"""

import numpy as np

def particle_create(N, phi, G, gamma, ic, Dfluc, bimodal_flag):
    ##### Make particles #####
    Nsmall = N / ((phi / (1 - phi)) * (1 / (G * G)) + 1)
    Nsmall = int(Nsmall)    
    Nbig = N - Nsmall
    Nbig = int(Nbig)    
    # Nbig and Nsmall are INTEGER!!! #
    
    if phi == 0.0: # only one big particle
        Nbig = 1
        Nsmall = N - Nbig
        Nbig = int(Nbig)
        Nsmall = int(Nsmall)
    
    if phi == 1.0: # only one small particle
        Nsmall = 1
        Nbig = N - Nsmall
        Nbig = int(Nbig)
        Nsmall = int(Nsmall)
        
    
    Dsmall = N / (Nsmall + (Nbig * G))
    Dbig = G * Dsmall
    
    rho_s = 1 / (1 + (gamma - 1) * phi)
    rho_l = gamma * rho_s
    
    Dn = np.random.random(N)
    
    # Dfluc = 0.2
    
    if bimodal_flag:
        Dnsmall = (Dsmall * (1 + Dfluc) - Dsmall * (1 - Dfluc)) * np.random.random(Nsmall) + Dsmall * (1 - Dfluc)
        Dnbig = (Dbig * (1 + Dfluc) - Dbig * (1 - Dfluc)) * np.random.random(Nbig) + Dbig * (1 - Dfluc)
    else:
        Dnsmall = Dsmall * np.ones(Nsmall) # no bimodal distribution
        Dnbig = Dbig * np.ones(Nbig)
    
    if phi == 0.0:
        Dnbig = [Dbig]
    if phi == 1.0:
        Dnsmall = [Dsmall]
    
    rho_ns = rho_s * np.ones(Nsmall)
    rho_nl = rho_l * np.ones(Nbig)
    
    m = np.zeros(N, dtype=np.float64)
    rho = np.zeros(N, dtype=np.float64)
    R_eff = np.zeros(N, dtype=np.float64)
    R_hyd = np.zeros(N, dtype=np.float64)
    R2n = np.zeros(N, dtype=np.float64)
    
    ind_big = np.zeros(N, dtype=np.float64)
    ind_small = np.zeros(N, dtype=np.float64)
    
    if (ic == 0): # SquareRD
        i = np.argsort(Dn)
        for k in range (0, Nsmall):
            Dn[i[k]] = Dnsmall[k]
            rho[i[k]] = rho_ns[k]
            ind_small[i[k]] = 1
        for k in range (0, Nbig):
            Dn[i[k + Nsmall]] = Dnbig[k]
            rho[i[k + Nsmall]] = rho_nl[k]
            ind_big[i[k + Nsmall]] = 1
    elif (ic != 0): # SquareST and SquareVAR
        for k in range (0, Nsmall):
            Dn[k] = Dnsmall[k]
            rho[k] = rho_ns[k]
            ind_small[k] = 1
        for k in range (0, Nbig):
            Dn[k + Nsmall] = Dnbig[k] 
            rho[k + Nsmall] = rho_nl[k]
            ind_big[k + Nsmall] = 1
    
    Vol = np.pi * np.power(Dn, 2) / 4

    m = Vol * rho
    
    R_eff = Dn / 2 
    R_hyd = np.sqrt((Vol / np.pi), dtype=np.float64)
    R2n = np.power(R_hyd , 2)
    
    ind_big = np.asarray(ind_big, dtype=bool)
    ind_small = np.asarray(ind_small, dtype=bool)
    
    return Nbig, Nsmall, Dbig, Dsmall, rho_s, rho_l, Dn, rho, Vol, m, R_eff, R_hyd, R2n, ind_big, ind_small