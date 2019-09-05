# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:30:02 2018

@author: Philip
"""
from numba import jit
import numpy as np
from numpy import linalg as LA

@jit
def preproc_2D_contact(contactlist, L, Dn, m, m_sep, z, HfillN, bnlmt, bw):
    #nbins = bnlmt / bw
    nbins = np.linspace(0, bnlmt, (bnlmt // bw) + 1)
    # Step 1: find location of contacts, find type of contacts
    indbig = m > m_sep   
    flag1 = np.zeros(len(contactlist))
    flag2 = np.zeros(len(contactlist))
    dz = np.zeros(len(contactlist))
    z_mean = np.zeros(len(contactlist))
    z_data = np.zeros(len(contactlist))    
    for i in range (0, len(contactlist)):
        contact = contactlist[i]
        flag1[i] = indbig[contact[0]]
        flag2[i] = indbig[contact[1]]
        dz[i] = z[contact[0]] - z[contact[0]]
        z_mean[i] = (z[contact[0]] + z[contact[0]]) / 2
        if (dz[i] > 0): # particle 1 is higher
            z_data[i] = z[contact[0]] - abs(dz[i]) * (Dn[contact[0]] / 2 / L[i])
        else:
            z_data[i] = z[contact[1]] - abs(dz[i]) * (Dn[contact[1]] / 2 / L[i])
    
    # Step 2: sort out three types of contacts
    bigbig = [0]
    smallsmall = [0]
    bigsmall = [0]
    for i in range (0, len(contactlist)):
        ind = flag1[i] + flag2[i]
        if (ind == 2): # big-big contact
            bigbig.append(z_data[i])
        elif (ind == 0): # small-small contact
            smallsmall.append(z_data[i])
        elif (ind == 1):
            bigsmall.append(z_data[i])
    del bigbig[0]
    del smallsmall[0]
    del bigsmall[0]
    
    # Step 3: histogram of the contacts
    yall, edge = np.histogram(z_data, bins = nbins)
    y0, dump = np.histogram(smallsmall, bins = nbins)
    y1, dump = np.histogram(bigsmall, bins = nbins)
    y2, dump = np.histogram(bigbig, bins = nbins)
    
    return edge, yall, y0, y1, y2

def preproc_2D_stress(N, x, y, Lx, Ly, K, R_eff, y_wall):
    stress = np.zeros((2,2))
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx
            dnm2 = (dx * dx) + (dy * dy)
            dnm = np.sqrt(dnm2)
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)
                    stress[0][0] = stress[0][0] - F * dx * dx
                    stress[0][1] = stress[0][1] - F * dx * dy
                    stress[1][0] = stress[1][0] - F * dy * dx
                    stress[1][1] = stress[1][1] - F * dy * dy
    stress = stress / (Lx * y_wall)
    eigval, eigvec = LA.eig(stress)
    return stress, eigval, eigvec

#@jit
def preproc_2D_ContactStressOrganizeLayer(z, Vol, Sxx_data, Sxy_data, Syy_data, bnlmt, proc_bw, ind_big):
    binsize = bnlmt // proc_bw
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # sampled in time
    
    Sxx_all = np.zeros((T, binsize))
    Syy_all = np.zeros((T, binsize))
    Sxy_all = np.zeros((T, binsize))
    Sxx_big = np.zeros((T, binsize))
    Syy_big = np.zeros((T, binsize))
    Sxy_big = np.zeros((T, binsize))
    Sxx_small = np.zeros((T, binsize))
    Syy_small = np.zeros((T, binsize))
    Sxy_small = np.zeros((T, binsize))
    
    Vol_dim = Vol.ndim
    
    for j in range (0, T):
        count_big = np.zeros(binsize)
        count_small = np.zeros(binsize)
        Vol_big = np.zeros(binsize)
        Vol_small = np.zeros(binsize)
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            Sxx_all[j][ind] = Sxx_all[j][ind] + Sxx_data[j][i]
            Sxy_all[j][ind] = Sxy_all[j][ind] + Sxy_data[j][i]
            Syy_all[j][ind] = Syy_all[j][ind] + Syy_data[j][i]        
            if (ind_big[i]): # large particle
                Sxx_big[j][ind] = Sxx_big[j][ind] + Sxx_data[j][i]
                Sxy_big[j][ind] = Sxy_big[j][ind] + Sxy_data[j][i]
                Syy_big[j][ind] = Syy_big[j][ind] + Syy_data[j][i]
                count_big[ind] = count_big[ind] + 1
                if Vol_dim < 2:
                    Vol_big[ind] = Vol_big[ind] + Vol[i]
                else:
                    Vol_big[ind] = Vol_big[ind] + Vol[j][i]
            if (~ind_big[i]): # small particle
                Sxx_small[j][ind] = Sxx_small[j][ind] + Sxx_data[j][i]
                Sxy_small[j][ind] = Sxy_small[j][ind] + Sxy_data[j][i]
                Syy_small[j][ind] = Syy_small[j][ind] + Syy_data[j][i]
                count_small[ind] = count_small[ind] + 1
                if Vol_dim < 2:
                    Vol_small[ind] = Vol_small[ind] + Vol[i]
                else:
                    Vol_small[ind] = Vol_small[ind] + Vol[j][i]
        
        Sxx_all[j][:] = Sxx_all[j][:] / (Vol_big + Vol_small)
        Sxy_all[j][:] = Sxy_all[j][:] / (Vol_big + Vol_small)
        Syy_all[j][:] = Syy_all[j][:] / (Vol_big + Vol_small)
        Sxx_big[j][:] = Sxx_big[j][:] / (Vol_big)
        Sxy_big[j][:] = Sxy_big[j][:] / (Vol_big)
        Syy_big[j][:] = Syy_big[j][:] / (Vol_big)
        Sxx_small[j][:] = Sxx_small[j][:] / (Vol_small)
        Sxy_small[j][:] = Sxy_small[j][:] / (Vol_small)
        Syy_small[j][:] = Syy_small[j][:] / (Vol_small)         
        
    Sxx_all = np.transpose(Sxx_all)
    Sxy_all = np.transpose(Sxy_all)
    Syy_all = np.transpose(Syy_all)
    Sxx_big = np.transpose(Sxx_big)
    Sxy_big = np.transpose(Sxy_big)
    Syy_big = np.transpose(Syy_big)
    Sxx_small = np.transpose(Sxx_small)
    Sxy_small = np.transpose(Sxy_small)
    Syy_small = np.transpose(Syy_small)
    
    return Sxx_all, Sxy_all, Syy_all, Sxx_big, Sxy_big, Syy_big, Sxx_small, Sxy_small, Syy_small

@jit
def preproc_2D_P_tau_from_Sij(Sxx_all, Sxy_all, Syy_all, Sxx_big, Sxy_big, Syy_big, Sxx_small, Sxy_small, Syy_small):
    # Step 1: tranpose to save development time...
    Sxx_all = np.transpose(Sxx_all)
    Sxy_all = np.transpose(Sxy_all)
    Syy_all = np.transpose(Syy_all)
    Sxx_big = np.transpose(Sxx_big)
    Sxy_big = np.transpose(Sxy_big)
    Syy_big = np.transpose(Syy_big)
    Sxx_small = np.transpose(Sxx_small)
    Sxy_small = np.transpose(Sxy_small)
    Syy_small = np.transpose(Syy_small)
    
    T = np.size(Sxx_all, 0) # sampled in time
    N_layers = np.size(Sxx_all, 1) # sampled in layers
    
    P_all = np.zeros((T, N_layers))
    P_big = np.zeros((T, N_layers))
    P_small = np.zeros((T, N_layers))
    tau_all = np.zeros((T, N_layers))
    tau_big = np.zeros((T, N_layers))
    tau_small = np.zeros((T, N_layers))
    
    for j in range (0, T):
        for ind in range (0, N_layers):
            if (~np.isnan(Sxx_all[j][ind])):
                P_all[j][ind], tau_all[j][ind] = preproc_2D_get_P_tau(Sxx_all[j][ind], Sxy_all[j][ind], Syy_all[j][ind])
            if (~np.isnan(Sxx_big[j][ind])):
                P_big[j][ind], tau_big[j][ind] = preproc_2D_get_P_tau(Sxx_big[j][ind], Sxy_big[j][ind], Syy_big[j][ind])
            if (~np.isnan(Sxx_small[j][ind])):
                P_small[j][ind], tau_small[j][ind] = preproc_2D_get_P_tau(Sxx_small[j][ind], Sxy_small[j][ind], Syy_small[j][ind])     
    
    P_all = np.transpose(P_all)
    P_big = np.transpose(P_big)
    P_small = np.transpose(P_small)
    tau_all = np.transpose(tau_all)
    tau_big = np.transpose(tau_big)
    tau_small = np.transpose(tau_small)
    
    return P_all, P_big, P_small, tau_all, tau_big, tau_small

@jit
def preproc_2D_P_tau_from_Sij_per_particle(z, Vol, Sxx_data, Sxy_data, Syy_data, ind_big):    
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # sampled in time    
    
    P_all = np.zeros((T, N))
    tau_all = np.zeros((T, N))
    
    # Get P and tau for each particle
    for j in range (0, T):
        for ind in range (0, N):
            P_all[j][ind], tau_all[j][ind] = preproc_2D_get_P_tau(Sxx_data[j][ind], Sxy_data[j][ind], Syy_data[j][ind])
            P_all[j][ind] = P_all[j][ind] / Vol[ind]
            tau_all[j][ind] = tau_all[j][ind] / Vol[ind]
    
    P_all = np.transpose(P_all)
    tau_all = np.transpose(tau_all)
    
    return P_all, tau_all

@jit
def preproc_2D_movmean(x, val, w):
    x = np.reshape(x, (1, -1))
    val = np.reshape(val, (1, -1))
    k = np.ones(w) / w
    x_mean = np.convolve(x, k)
    val_mean = np.convolve(val, k)
    return x_mean, val_mean

@jit
def preproc_2D_get_P_tau(Sxx, Sxy, Syy):
    stress_local = np.zeros((2,2))
    stress_local[0][0] = Sxx
    stress_local[0][1] = Sxy
    stress_local[1][0] = Sxy
    stress_local[1][1] = Syy
    eigval, eigvec = LA.eig(stress_local)
    P = np.sum(eigval) / 2
    tau = np.abs(np.diff(eigval)) / 2
    return P, tau

@jit
def preproc_2D_Ek2KineticStress(Eky_all, Eky_big, Eky_small, Ekx_all, Ekx_big, Ekx_small, Ekxy_all, Ekxy_big, Ekxy_small, Lx, proc_bw):
    convertion = 2 / (Lx * proc_bw) # converts Ek to kinetic stress by multiplying by 2 (0.5mv2 to mv2), divide by Lx*bw (the volume)
    Eky_all = Eky_all * convertion
    Eky_big = Eky_big * convertion
    Eky_small = Eky_small * convertion
    Ekx_all = Ekx_all * convertion
    Ekx_big = Ekx_big * convertion
    Ekx_small = Ekx_small * convertion
    Ekxy_all = Ekxy_all * convertion
    Ekxy_big = Ekxy_big * convertion
    Ekxy_small = Ekxy_small * convertion
    return Eky_all, Eky_big, Eky_small, Ekx_all, Ekx_big, Ekx_small, Ekxy_all, Ekxy_big, Ekxy_small

@jit
def preproc_2D_ContactStressOrganize_v0(z, Dn, P_data, tau_data, Sxx_data, Sxy_data, Syy_data, bnlmt, proc_bw, ind_big):
    Vol = np.pi * Dn * Dn / 4
    binsize = bnlmt // proc_bw
    T = np.size(z, 0) # sampled in time
    N = np.size(z, 1) # sampled in time
    Sxx_all = np.zeros((T, binsize))
    Syy_all = np.zeros((T, binsize))
    Sxy_all = np.zeros((T, binsize))
    Sxx_big = np.zeros((T, binsize))
    Syy_big = np.zeros((T, binsize))
    Sxy_big = np.zeros((T, binsize))
    Sxx_small = np.zeros((T, binsize))
    Syy_small = np.zeros((T, binsize))
    Sxy_small = np.zeros((T, binsize))
    
    P_all = np.zeros((T, binsize))
    P_big = np.zeros((T, binsize))
    P_small = np.zeros((T, binsize))
    tau_all = np.zeros((T, binsize))
    tau_big = np.zeros((T, binsize))
    tau_small = np.zeros((T, binsize))
    
    for j in range (0, T):
        count_big = np.zeros(binsize)
        count_small = np.zeros(binsize)
        Vol_big = np.zeros(binsize)
        Vol_small = np.zeros(binsize)
        for i in range (0, N):
            ind = int(np.ceil(z[j][i] / proc_bw)) - 1
            Sxx_all[j][ind] = Sxx_all[j][ind] + Sxx_data[j][i]
            Sxy_all[j][ind] = Sxy_all[j][ind] + Sxy_data[j][i]
            Syy_all[j][ind] = Syy_all[j][ind] + Syy_data[j][i]
            P_all[j][ind] = P_all[j][ind] + P_data[j][i]
            tau_all[j][ind] = tau_all[j][ind] + tau_data[j][i]          
            if (ind_big[i]): # large particle
                Sxx_big[j][ind] = Sxx_big[j][ind] + Sxx_data[j][i]
                Sxy_big[j][ind] = Sxy_big[j][ind] + Sxy_data[j][i]
                Syy_big[j][ind] = Syy_big[j][ind] + Syy_data[j][i]
                P_big[j][ind] = P_big[j][ind] + P_data[j][i]
                tau_big[j][ind] = tau_big[j][ind] + tau_data[j][i]
                count_big[ind] = count_big[ind] + 1
                Vol_big[ind] = Vol_big[ind] + Vol[i]
            if (~ind_big[i]): # small particle
                Sxx_small[j][ind] = Sxx_small[j][ind] + Sxx_data[j][i]
                Sxy_small[j][ind] = Sxy_small[j][ind] + Sxy_data[j][i]
                Syy_small[j][ind] = Syy_small[j][ind] + Syy_data[j][i]
                P_small[j][ind] = P_small[j][ind] + P_data[j][i]
                tau_small[j][ind] = tau_small[j][ind] + tau_data[j][i]
                count_small[ind] = count_small[ind] + 1
                Vol_small[ind] = Vol_small[ind] + Vol[i]
        
        Sxx_all[j][:] = Sxx_all[j][:] / (Vol_big + Vol_small)
        Sxy_all[j][:] = Sxy_all[j][:] / (Vol_big + Vol_small)
        Syy_all[j][:] = Syy_all[j][:] / (Vol_big + Vol_small)
        Sxx_big[j][:] = Sxx_big[j][:] / (Vol_big)
        Sxy_big[j][:] = Sxy_big[j][:] / (Vol_big)
        Syy_big[j][:] = Syy_big[j][:] / (Vol_big)
        Sxx_small[j][:] = Sxx_small[j][:] / (Vol_small)
        Sxy_small[j][:] = Sxy_small[j][:] / (Vol_small)
        Syy_small[j][:] = Syy_small[j][:] / (Vol_small)
        
        P_all[j][:] = P_all[j][:] / (count_big + count_small)
        tau_all[j][:] = tau_all[j][:] / (count_big + count_small)
        P_big[j][:] = P_big[j][:] / count_big
        tau_big[j][:] = tau_big[j][:] / count_big
        P_small[j][:] = P_small[j][:] / count_small
        tau_small[j][:] = tau_small[j][:] / count_small
        
    Sxx_all = np.transpose(Sxx_all)
    Sxy_all = np.transpose(Sxy_all)
    Syy_all = np.transpose(Syy_all)
    Sxx_big = np.transpose(Sxx_big)
    Sxy_big = np.transpose(Sxy_big)
    Syy_big = np.transpose(Syy_big)
    Sxx_small = np.transpose(Sxx_small)
    Sxy_small = np.transpose(Sxy_small)
    Syy_small = np.transpose(Syy_small)
    P_all = np.transpose(P_all)
    P_big = np.transpose(P_big)
    P_small = np.transpose(P_small)
    tau_all = np.transpose(tau_all)
    tau_big = np.transpose(tau_big)
    tau_small = np.transpose(tau_small)
    
    return Sxx_all, Sxy_all, Syy_all, Sxx_big, Sxy_big, Syy_big, Sxx_small, Sxy_small, Syy_small, P_all, P_big, P_small, tau_all, tau_big, tau_small



@jit
def preproc_2D_ContactStress(N, N_org, x, y, Lx, Ly, K, R_eff):
    # This code returns the contact stress per particle
    stress_local = np.zeros((2,2))
    Sxx = np.zeros(N_org)
    Sxy = np.zeros(N_org)
    Syy = np.zeros(N_org)
    P = np.zeros(N_org)
    tau = np.zeros(N_org)
    for nn in range (0, N):
        for mm in range (nn + 1, N):
            dy = y[mm] - y[nn]
            dy = dy - np.around(dy / Ly) * Ly
            dx = x[mm] - x[nn]
            dx = dx - np.around(dx / Lx) * Lx
            dnm2 = (dx * dx) + (dy * dy)
            dnm = np.sqrt(dnm2)
            Dnm = R_eff[nn] + R_eff[mm]
            if (abs(dy) < Dnm):
                if (dnm < Dnm):
                    F = -K * (Dnm / dnm - 1)                    
                    # contact stress partitioning is applied following the 
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
                    
    for nn in range (0, N_org):
        stress_local[0][0] = Sxx[nn]
        stress_local[0][1] = Sxy[nn]
        stress_local[1][0] = Sxy[nn]
        stress_local[1][1] = Syy[nn]
                    
        eigval, eigvec = LA.eig(stress_local)
        P[nn] = np.sum(eigval) / 2
        tau[nn] = np.abs(np.diff(eigval)) / 2
            
    return Sxx, Syy, Sxy, P, tau
    
    
    
    
    
    
    
    
    
    
    
    
    