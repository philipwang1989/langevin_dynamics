
# coding: utf-8

# In[ ]:

##### Thermal model - Langevin #####

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
from time import clock
import scipy.io as sio
import scipy.special as spe
import math
import MDfun as MDF
import importlib
importlib.reload(MDF)

#def main(G, phi, psi, Tamp, ReynoldsSett, en, HfillN, NondimStiff, sigma_erf, ic, seed):
def main():    
    #DebugTag = True
    DebugTag = False
    
    #BottomWallTag = True
    BottomWallTag = False
    
    HalfWidthTag = True
    #HalfWidthTag = False
    
    if (DebugTag == False):
        G = sys.argv[1]
        phi = sys.argv[2]
        psi = sys.argv[3]
        Tamp = sys.argv[4]
        ReynoldsSett = (sys.argv[5])
        en = sys.argv[6]
        HfillN = sys.argv[7]
        NondimStiff = sys.argv[8]
        sigma_erf = (sys.argv[9])
        ic = sys.argv[10]
        seed = sys.argv[11]
        N = int(HfillN) * int(HfillN)
        if (HalfWidthTag):
            N = int(HfillN) * int(HfillN) // 2
        Nstr = str(N)        
        
        # gamma = rho_l/rho_s = (m_l/m_s)/(V_l/V_s)
        
        gamma = float(psi) / (float(G) ** 2) # psi if the ratio of m_{l} / m_{s}
        gamma = f"{gamma:.2f}"
        
        filename = 'output_2D_LT_VV_T0_' + Tamp + '_G_' + G + '_phi_' + phi + '_psi_' + psi + '_ReynoldsSett_' + ReynoldsSett + '_Nc_' + Nstr + '_HfillN_' + HfillN + '_sigma_erf_' + sigma_erf + '_ic_' + ic + '_seed_' + seed + '.mat'
        
        G = float(G)
        phi = float(phi)
        gamma = float(psi) / (float(G) ** 2)
        Tamp = float(Tamp)
        ReynoldsSett = float(ReynoldsSett)
        en = float(en)
        HfillN = float(HfillN)
        NondimStiff = float(NondimStiff)
        sigma_erf = float(sigma_erf)
        ic = int(ic)
        seed = int(seed)
        
        ##***## CHECK SAVEPATH ##***##
        if (ic == 3):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_SB/'
        elif (ic == 1):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_ST/'
        elif (ic == 0):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_RD/'
        elif (ic == 4):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_VAR_SB/'
        elif (ic == 2):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_VAR_ST/'
        
        # savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT/'
        
    else:        
        gamma = float(psi) / (float(G) ** 2) # psi is the ratio of m_{l} / m_{s}
        
        N = int(HfillN) * int(HfillN) # // 2
        if (HalfWidthTag):
            N = int(HfillN) * int(HfillN) // 2
        filename = 'temp_' + str(Tamp) + '.mat'
        savepath = 'E:/Matlab_output/local/'            
    
    destination = savepath + filename
    
    plotit = DebugTag
    wtime1 = clock()  
    
    timestamp()
    
    np.random.seed(seed)
    
    ##### Experimental parameters #####
    D = 1
    g = 0.1
    
    Nt = 2e7
    
    #if (Tamp < 0.30): # if low temperature
    #    Nt = 1e7
    #else: # else is high temperature
    #    Nt = 5e7
    
    #if (G < 2.1):
    #    Nt = 1e7 # Number of time step
    #else:
    #    Nt = 5e6
    
    if (DebugTag == True):
        Nt = 1e5
    
    K = NondimStiff * g * np.pi * (D * D / 4) / D
    B = np.sqrt(g / D, dtype=np.float64) * np.pi / 4 * D * D * ReynoldsSett
    B_pour = np.sqrt(g / D) * np.pi / 4 * D * D * 0.5
    
    ##### Simulation parameters #####
    N_per_coll = 50
    dt = 1 / N_per_coll * 2 * np.pi * np.sqrt(np.pi * D * D / 4 / K, dtype=np.float64)
    
    tau_c = np.pi * np.sqrt(np.pi * D * D / 4 / K)
    gv = -2 * np.log(en) / tau_c
    
    ##### Make particles #####
    Nsmall = N / ((phi / (1 - phi)) * (1 / (G * G)) + 1)
    Nsmall = int(Nsmall)
    Nbig = N - Nsmall
    Nbig = int(Nbig)
    
    Dsmall = N / (Nsmall + (Nbig * G))
    Dbig = G * Dsmall
    
    rho_s = 1 / (1 + (gamma - 1) * phi)
    rho_l = gamma * rho_s
    
    Dn = np.random.random(N)
    
    Dfluc = 0.2
    
    Dnsmall = Dsmall * np.ones(Nsmall) # no bimodal distribution
    Dnbig = Dbig * np.ones(Nbig)
    
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
    
    # m = np.pi * np.power(Dn, 2) / 4
    m = Vol * rho
    # D_ratio = np.mean(Dn)
    
    # Dn = Dn / D_ratio
    # m = m / np.power(D_ratio ,2)
    
    R_eff = Dn / 2 
    R_hyd = np.sqrt((Vol / np.pi), dtype=np.float64)
    R2n = np.power(R_hyd , 2)
    # Hfill = HfillN * np.mean(2 * R_hyd)
    
    Dn_sort = np.sort(Dn)
    Vol_sort = np.sort(Vol)
    m_sort = np.sort(m)
    
    ind_big = np.asarray(ind_big, dtype=bool)
    ind_small = np.asarray(ind_small, dtype=bool)
    
    #Dsmall = np.mean(Dn_sort[1:Nsmall])
    #Dbig = np.mean(Dn_sort[Nsmall + 1:N])      
    Dbig_max = np.amax(Dn[ind_big])
    Dsmall_max = np.amax(Dn[ind_small])
    
    Volsmall = np.mean(Vol[ind_small])
    Volbig = np.mean(Vol[ind_big])
    
    Volsmall_max = np.amax(Vol[ind_small])
    Volbig_min = np.amin(Vol[ind_big])
    Vol_sep = (Volsmall_max + Volbig_min) / 2
    
    msmall = np.mean(m[ind_small])
    mbig = np.mean(m[ind_big])
    
    msmall_max = np.amax(m[ind_small])
    mbig_min = np.amin(m[ind_big])
    m_sep = (msmall_max + mbig_min) / 2
    
    Nb = np.sum(Vol > Vol_sep)
    Ns = np.sum(Vol < Vol_sep)
    
    StreamN = N / HfillN
    
    if (ic == 0): # random preparation
        x, y, Lx, Ly = MDF.initial_preparation_RD(N, G, D, R_hyd, HfillN, 0)
    elif (ic == 1): # fragile
        x, y, Lx, Ly = MDF.initial_preparation_ST(N, Nb, Ns, Dbig_max, D, R_hyd, HfillN)
    elif (ic == 2): # varying initial prep.
        x, y, Lx, Ly = MDF.initial_preparation_VAR_ST(N, Nb, Ns, G, D, R_hyd, StreamN, HfillN, sigma_erf, Volbig, Volsmall)
    elif (ic == 3): # prearmored
        x, y, Lx, Ly = MDF.initial_preparation_SB(N, Nb, Ns, G, Dsmall_max, R_hyd, HfillN)
    elif (ic == 4): # small at the bottom VAR
        x, y, Lx, Ly = MDF.initial_preparation_VAR_SB(N, Nb, Ns, G, D, R_hyd, StreamN, HfillN, sigma_erf, Volbig, Volsmall)    
    
    ##### draw particles #####
    if (plotit):
        draw_particle(x, y, Lx, N, R_eff)    
    ##### end draw #####
    
    ##### Get critical temperature #####
    # mu is the number of layers
    mu_big = float(Nbig) / (float(Lx) / Dbig)
    mu_small = float(Nsmall) / (float(Lx) / Dsmall)
    # pot is the gravitational potential energy 
    pot_big = mbig * g * mu_big * Dbig
    pot_small = msmall * g * mu_small * Dsmall
    mu_0 = 111.5 / 1.15
    Tc_big = pot_big / mu_0
    Tc_small = pot_small / mu_0
    
    ####################################
    
    vx = np.sqrt(0.001, dtype=np.float64) * D * np.random.randn(N)
    vx = vx - np.mean(vx)
    #vx = 0*x
    vy = 0*y
    #if (G == 1.0 and psi == 1.0):
    #    vy = np.sqrt(Tamp, dtype=np.float64) * D * np.random.randn(N)
    #else:
    #    vy = 0*y
    
    ax_old = np.zeros(N, dtype=np.float64)
    ay_old = np.zeros(N, dtype=np.float64)
    
    ax = ax_old.copy()
    ay = ay_old.copy()
    
    a_thresh = 1e-5 * g
    v_thresh = 1e-5 * np.mean(m) * g
    
    ##### Add bottom wall #####
    N_org = N # this makes exactly equal, if the BottomWallTag is triggered, N > N_org
    if (BottomWallTag):       
        #D_wall = np.amin(Dn) # D_wall is the diameter of the wall particles 
        D_wall_min = Dsmall * (1 - Dfluc)
        D_wall_max = Dsmall * (1 + Dfluc)
        #D_spacing_min = 0.1 * np.amin(Dn)
        #D_spacing_max = 0.9 * np.amin(Dn)
        D_spacing_min = 0.1 * D_wall_min
        D_spacing_max = 0.9 * D_wall_min
        spacing = np.random.uniform(low = D_spacing_min, high = D_spacing_max) # spacing is the distance between the wall particles
        x_wall = np.zeros(2)        
        x_wall[0] = 0.0
        x_wall[1] = spacing + np.amin(Dn)
        Dn_wall = np.zeros(2)
        Dn_wall[0] = np.random.uniform(low = D_wall_min, high = D_wall_max)
        Dn_wall[1] = np.random.uniform(low = D_wall_min, high = D_wall_max)
        ind_wall = 1
        
        while (x_wall[ind_wall] < Lx - np.amin(Dn)): # start from the second elements   
            Dn_wall_temp = np.random.uniform(low = D_wall_min, high = D_wall_max)
            spacing = np.random.uniform(low = D_spacing_min, high = D_spacing_max)
            x_temp = x_wall[ind_wall] + spacing + Dn_wall_temp
            x_wall = np.append(x_wall, x_temp)
            Dn_wall = np.append(Dn_wall, Dn_wall_temp)
            ind_wall = ind_wall + 1
        #if (x_wall[-1] - Lx > 0): # two wall particles overlapped
        #    dist = (abs(x_wall[-2] - Lx) + x_wall[1]) - D_wall
        #    dist_flag = dist / D_wall
        #    if (dist_flag > 1 and dist_flag < 2): # only need one in between
        #        x_wall[0] = (x_wall[-1] - Lx) / 2
        #        x_wall = np.delete(x_wall,-1)
        
        #spacing = np.around(np.amin(Dnsmall), decimals = 1)
        
        Nwall = x_wall.size
        #if (Lx - x_wall[Nwall - 1] + x_wall[0] > 2 * np.amin(Dn)):
        #    print('bug')
        #Nwall = int(np.ceil(Lx / (Dsmall + spacing)))
        #delta = Lx / Nwall
        #x_wall = np.arange(Nwall) * delta + delta # offset by one delta
        y_wall = np.zeros(Nwall)
        #Dn_wall = np.amin(Dn) * np.ones(x_wall.size) # correct D_wall
        m_wall = np.amin(m) * np.ones(x_wall.size)
        Vol_wall = np.amin(Vol) * np.ones(x_wall.size)

        N_org = N
        N = N + Nwall
        x = np.append(x, x_wall)
        y = np.append(y, y_wall)
        vx = np.append(vx, np.zeros(x_wall.size))
        vy = np.append(vy, np.zeros(x_wall.size))
        ax_old = np.append(ax_old, np.zeros(x_wall.size))
        ay_old = np.append(ay_old, np.zeros(x_wall.size))
        ax = ax_old
        ay = ay_old
        m = np.append(m, m_wall)
        Vol = np.append(Vol, Vol_wall)
        Dn = np.append(Dn, Dn_wall)
        R_eff = Dn / 2
        R_hyd = np.sqrt(Vol / np.pi)
        R2n = R_hyd * R_hyd    
    
        
    
    ##### Pour particles #####
    nt, x, y, vx, vy, ax, ay, ax_old, ay_old = (
            MDF.pour_particle(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, a_thresh, v_thresh, plotit, BottomWallTag))
    x_start = x[0:N_org]
    y_start = y[0:N_org]
    nt_start = nt
    ##### draw particles #####
    if (plotit):
        #draw_particle(x, y, Lx, N, R_eff)   
        draw_particle_w_wall(x, y, Lx, N, N_org, R_eff)
    ##### end draw #####
    
    ##### add top free wall #####
    #m_wall = np.sum(m)
    #y_wall = np.amax(y_start) + Dbig
    #x_wall = 0.0 # infinitely long wall, for broadcasting consideration
    #D_wall = Dsmall
    
    #vx_wall = 0.0
    #vy_wall = 0.0
    #ax_wall = 0.0
    #ay_wall = 0.0
    
    #x = np.append(x, x_wall)
    #y = np.append(y, y_wall)
    #vx = np.append(vx, vx_wall)
    #vy = np.append(vy, vy_wall)
    #ax = np.append(ax, ax_wall)
    #ay = np.append(ay, ay_wall)
    #ax_old = np.append(ax_old, ax_wall)
    #ay_old = np.append(ay_old, ay_wall)
    #m = np.append(m, m_wall)    
    #Dn = np.append(Dn, D_wall)
    
    Nc_org = N
    #N = N + 1
    
    ##### Drop the top wall #####
    #nt, x, y, vx, vy, ax, ay, ax_old, ay_old = (
    #        MDF.drop_wall(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, a_thresh, v_thresh, plotit))
    
    ##### draw particles #####
    if (plotit):
        #draw_particle(x, y, Lx, N, R_eff)   
        draw_particle_w_wall(x, y, Lx, N, N_org, R_eff)
    ##### end draw #####
    
    ##### Heat the particles #####
    nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, T, Ek, x_data, y_data, vx_data, vy_data, T_big, T_small = (
            MDF.langevin_thermostat_VV_no_wall(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, Nc_org, R_eff, K, B, Tamp, gv, a_thresh, v_thresh, plotit))
    
    print(y_data.shape)
    print(vy_data.shape)
    print(np.size(y_data, 0))
    
    bnlmt = HfillN * 2.0
    bnlmt = int(bnlmt)
    if (np.mod(bnlmt, 2) == 1):
        bnlmt = bnlmt + 1 # to ensure even number
    bw = 2
    
    # Get time evolving layered average velocity
    #vys_big, vys_small, vys_all, vy_prime = MDF.preproc_2D_vs(vy_data, Vol, Vol_sep, y_data, bnlmt, bw)
    #vxs_big, vxs_small, vxs_all, vx_prime = MDF.preproc_2D_vs(vx_data, Vol, Vol_sep, y_data, bnlmt, bw)
    
    m = m[0:N_org]
    
    # Get time evolving layered kinetic energy and temperature
    #Eky_all, Eky_big, Eky_small, Ty_all, Ty_big, Ty_small = MDF.preproc_2D_T(vy_prime, m, Vol, Vol_sep, y_data, bnlmt, bw)        
    #Ekx_all, Ekx_big, Ekx_small, Tx_all, Tx_big, Tx_small = MDF.preproc_2D_T(vx_prime, m, Vol, Vol_sep, y_data, bnlmt, bw)
    
    ##### draw particles #####
    if (plotit):
        #draw_particle(x, y, Lx, N, R_eff)   
        draw_particle_w_wall(x, y, Lx, N, N_org, R_eff)
    ##### end draw #####
    
    x_stop = x[0:N_org]
    y_stop = y[0:N_org]
    nt_stop = nt
    
    static_equilibrium = False
    if (nt < Nt):
        static_equilibrium = True
    
    Dn = Dn[0:N_org]
    
    R_eff = R_eff[0:N_org]
    R_hyd = R_hyd[0:N_org]
    N = N_org
    
    if (DebugTag):
        print(Ly)
        print(rho_l/rho_s)
        print(gamma)
        
        print(mbig/msmall)
        print(psi)
        
        print(np.mean(T))
        #print(np.mean(T_big))
        #print(np.mean(T_small))
        #print(np.mean(ph))
        #print(Volbig/Volsmall)
    
    if (seed != 100):
        (sio.savemat(destination, {'x_start': x_start, 'y_start': y_start, 'nt_start': nt_start
                                   , 'x_stop': x_stop, 'y_stop': y_stop, 'nt_stop': nt_stop, 'N':N
                                   , 'Dn': Dn, 'm': m, 'Vol': Vol, 'R_eff': R_eff, 'R_hyd': R_hyd, 'HfillN': HfillN
                                   , 'G': G, 'NondimStiff': NondimStiff, 'seed': seed
                                   , 'Lx': Lx, 'Ly': Ly, 'en': en, 'tau_c': tau_c, 'gv': gv
                                   , 'gamma': gamma, 'phi': phi
                                   , 'Nsmall': Nsmall, 'Nbig': Nbig, 'Dsmall': Dsmall, 'Dbig': Dbig
                                   , 'rho': rho, 'rho_s': rho_s, 'rho_l': rho_l
                                   , 'CM_big': CM_big, 'CM_small': CM_small, 'T': T, 'Ek': Ek
                                   , 'T_big': T_big, 'T_small': T_small
                                   #, 'Ek_big': Ek_big, 'Ek_small': Ek_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small, 'vxs_big': vxs_big, 'vxs_small': vxs_small
                                   #, 'vys_all': vys_all, 'vxs_all': vxs_all
                                   #, 'Eky_all': Eky_all, 'Eky_big': Eky_big, 'Eky_small': Eky_small, 'Ty_all': Ty_all, 'Ty_big': Ty_big, 'Ty_small': Ty_small
                                   #, 'Ekx_all': Ekx_all, 'Ekx_big': Ekx_big, 'Ekx_small': Ekx_small, 'Tx_all': Tx_all, 'Tx_big': Tx_big, 'Tx_small': Tx_small
                                   , 'dt': dt, 'Nt': Nt, 'B_pour': B_pour, 'B': B, 'K': K, 'static_equilibrium': static_equilibrium
                                   #, 'mean_vxs_big': mean_vxs_big, 'mean_vxs_small': mean_vxs_small, 'mean_vys_big': mean_vys_big, 'mean_vys_small': mean_vys_small
                                   #, 'mean_xs_big': mean_xs_big, 'mean_xs_small': mean_xs_small, 'mean_ys_big': mean_ys_big, 'mean_ys_small': mean_ys_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small
                                   , 'x_data': x_data, 'y_data': y_data
                                   #, 'MSD_x': MSD_x, 'MSD_y': MSD_y                                   
                                   #, 'x_data': x_data, 'y_data': y_data, 'vx_data': vx_data, 'vy_data': vy_data
                                   #, 'ph': ph
                                   #, 'Tc_big': Tc_big, 'Tc_small': Tc_small
                                   , 'ind_big':ind_big, 'ind_small':ind_small}))
                                   #, 'VDen_all': VDen_all, 'VDen_big': VDen_big, 'VDen_small': VDen_small
                                   #, 'NDen_all': NDen_all, 'NDen_big': NDen_big, 'NDen_small': NDen_small}))
    else:
        (sio.savemat(destination, {'x_start': x_start, 'y_start': y_start, 'nt_start': nt_start
                                   , 'x_stop': x_stop, 'y_stop': y_stop, 'nt_stop': nt_stop, 'N':N
                                   , 'Dn': Dn, 'm': m, 'Vol': Vol, 'R_eff': R_eff, 'R_hyd': R_hyd, 'HfillN': HfillN
                                   , 'G': G, 'NondimStiff': NondimStiff, 'seed': seed
                                   , 'Lx': Lx, 'Ly': Ly, 'en': en, 'tau_c': tau_c, 'gv': gv
                                   , 'gamma': gamma, 'phi': phi
                                   , 'Nsmall': Nsmall, 'Nbig': Nbig, 'Dsmall': Dsmall, 'Dbig': Dbig
                                   , 'rho': rho, 'rho_s': rho_s, 'rho_l': rho_l
                                   , 'CM_big': CM_big, 'CM_small': CM_small, 'T': T, 'Ek': Ek
                                   , 'T_big': T_big, 'T_small': T_small
                                   #, 'T_big': T_big, 'T_small': T_small, 'Ek_big': Ek_big, 'Ek_small': Ek_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small, 'vxs_big': vxs_big, 'vxs_small': vxs_small
                                   #, 'vys_all': vys_all, 'vxs_all': vxs_all
                                   #, 'Eky_all': Eky_all, 'Eky_big': Eky_big, 'Eky_small': Eky_small, 'Ty_all': Ty_all, 'Ty_big': Ty_big, 'Ty_small': Ty_small
                                   #, 'Ekx_all': Ekx_all, 'Ekx_big': Ekx_big, 'Ekx_small': Ekx_small, 'Tx_all': Tx_all, 'Tx_big': Tx_big, 'Tx_small': Tx_small
                                   , 'dt': dt, 'Nt': Nt, 'B_pour': B_pour, 'B': B, 'K': K, 'static_equilibrium': static_equilibrium
                                   #, 'mean_vxs_big': mean_vxs_big, 'mean_vxs_small': mean_vxs_small, 'mean_vys_big': mean_vys_big, 'mean_vys_small': mean_vys_small
                                   #, 'mean_xs_big': mean_xs_big, 'mean_xs_small': mean_xs_small, 'mean_ys_big': mean_ys_big, 'mean_ys_small': mean_ys_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small
                                   #, 'x_data': x_data, 'y_data': y_data
                                   #, 'MSD_x': MSD_x, 'MSD_y': MSD_y
                                   #, 'x_data': x_data, 'y_data': y_data, 'vx_data': vx_data, 'vy_data': vy_data
                                   #, 'ph': ph
                                   #, 'Tc_big': Tc_big, 'Tc_small': Tc_small
                                   , 'ind_big':ind_big, 'ind_small':ind_small}))
                                   #, 'VDen_all': VDen_all, 'VDen_big': VDen_big, 'VDen_small': VDen_small
                                   #, 'NDen_all': NDen_all, 'NDen_big': NDen_big, 'NDen_small': NDen_small}))
    
    wtime2 = clock()
    print('')
    print('Elapsed time: %g.' % (wtime2 - wtime1))
    
    return
    

def timestamp():
    import time
    
    t = time.time()
    print(time.ctime(t))
        
    return
    
    

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

def make_data_plot(fig, ax, acc_list, T_list, d_acc, T_temp, count):
    nt_list = range(count)
    nt_list = np.asarray(nt_list)
    acc_list[count] = d_acc
    T_list[count] = T_temp
    
    acc_print = acc_list[0:count] # get rid of zeros
    T_print = T_list[0:count] # get rid of zeros        
        
    line1, = ax.plot(nt_list, acc_print, 'r-')
    line2, = ax.plot(nt_list, T_print, 'b-')
    
    fig.canvas.draw()
    
    return acc_list, T_list

def draw_particle_w_wall(x, y, Lx, N, N_org, R_eff):
    x_disp = np.mod(x, Lx)
    
    xy = np.zeros((N, 2))
    for i in range(N):
        xy[i][0] = x_disp[i]
        xy[i][1] = y[i]
    
    patches = [plt.Circle(center, size) for center, size in zip(xy[0:N_org], R_eff[0:N_org])]
    fig, ax = plt.subplots()
    coll = matplotlib.collections.PatchCollection(patches,edgecolors='black')
    patches = [plt.Circle(center, size) for center, size in zip(xy[N_org:N], R_eff[N_org:N])]
    coll_wall = matplotlib.collections.PatchCollection(patches, facecolor='red',edgecolors='black')
    
    ax.add_collection(coll)
    ax.add_collection(coll_wall)

    ax.margins(0.01)
    plt.axis('equal')
    plt.show()
    return

def get_MSD(x,y):
    [N, T] = x.shape
    xdiff = np.cumsum(np.diff(x, axis=1), axis=1)
    ydiff = np.cumsum(np.diff(y, axis=1), axis=1)   
    xdiff = np.append(np.zeros((N,1)), xdiff, axis=1)
    ydiff = np.append(np.zeros((N,1)), ydiff, axis=1)
    xdiff = np.mean(np.square(xdiff), axis=0)
    ydiff = np.mean(np.square(ydiff), axis=0)
    
    return xdiff, ydiff

def get_ISF(x, y, Lx, Ly):
    [N, T] = x.shape
    for t1 in range (0, N):
        for t2 in range (t1 + 1, N):
            x1 = x[:, t1]
            x2 = x[:, t2]
            y1 = y[:, t1]
            y2 = y[:, t2]
            
            dt_index = t2 - t1;
            dx = x2 - x1
            dx = dx - round(dx / Lx) * Lx
            dy = y2 - y1
            dy = dy - round(dy / Ly) * Ly
            ISF[dt_index] = ISF[dt_index] + sum(np.cos(kx * dx + ky * dy))
            counts[dt_index] = counts[dt_index] + 1
    
    ind = counts > 0
    ISF[ind] = ISF[ind] / counts[ind] / N
    
    
    return ISF

# (G,phi,psi,Tamp,ReSet,en,HfillN,NondimStiff,sigma_erf,ic,seed)
# ic only has 0 (RD), 1 (ST), 3(SB) options
#main(1.4, 0.5, 1.96, 0.10, 2.5, 1, 24, 1000, 0.01, 0, 5) # uncomment to make it a function
#main(1.4, 0.5, 1.0, 0.05, 2.5, 1, 24, 1000, 0.01, 0, 1) # uncomment to make it a function
#main(1.4, 0.5, 1.96, 0.30, 2.5, 1, 24, 1000, 0.01, 0, 1) # uncomment to make it a function
#main(1.4, 0.5, 1.96, 0.20, 2.5, 1, 24, 1000, 0.01, 0, 1) # uncomment to make it a function
#main(1.4, 0.5, 1.96, 0.15, 2.5, 1, 24, 1000, 0.01, 0, 1) # uncomment to make it a function

if __name__ == "__main__": # uncomment this and the next line for cluster usage
    main()
    

