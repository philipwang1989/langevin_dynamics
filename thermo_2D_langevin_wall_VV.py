
# coding: utf-8

# In[1]:

##### Thermal model - Langevin #####
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
from time import clock
import scipy.io as sio
import scipy.special as spe
# import math
import MDfun as MDF
import MDISF as ISF
import MDMSD_glb as MSD_glb
import MDMSD_w_l_s as MSD_local
import MDprep as MDini
import MDstress as MDSTR
import importlib
importlib.reload(MDF)
importlib.reload(ISF)

#def main(G, phi, psi, Tamp, ReynoldsSett, en, HfillN, StreamN, NondimStiff, sigma_erf, PMM, ic, Tswitch, Theta, seed):
def main():
    #DebugTag = True
    DebugTag = False
    
    #BottomWallTag = True
    BottomWallTag = False
    
    # StreamN = 12
    
    if (DebugTag == False):
        G = sys.argv[1]
        phi = sys.argv[2]
        psi = sys.argv[3]
        Tamp = sys.argv[4]
        ReynoldsSett = (sys.argv[5])
        en = sys.argv[6]
        HfillN = sys.argv[7]
        StreamN = sys.argv[8]
        NondimStiff = sys.argv[9]
        sigma_erf = (sys.argv[10])
        PMM = sys.argv[11]
        ic = sys.argv[12]
        Tswitch = sys.argv[13]
        Theta = sys.argv[14]
        seed = sys.argv[15]
        #N = int(HfillN) * int(HfillN)
        ## StreamN = N / HfillN ##
        N = int(HfillN) * int(StreamN)
        Nstr = str(N)        
        
        # gamma = rho_l/rho_s = (m_l/m_s)/(V_l/V_s)
        
        gamma = float(psi) / (float(G) ** 2) # psi if the ratio of m_{l} / m_{s}
        gamma = f"{gamma:.2f}"
        
        if int(Tswitch) == 4:
            filename = 'output_2D_LT_VV_T0_' + Tamp + '_G_' + G + '_phi_' + phi + '_psi_' + psi + '_ReynoldsSett_' + ReynoldsSett + '_Nc_' + Nstr + '_HfillN_' + HfillN + '_StreamN_' + StreamN + '_sigma_erf_' + sigma_erf + '_NDS_' + NondimStiff + '_PMM_' + PMM + '_en_' + en + '_ic_' + ic + '_TS_' + Tswitch + '_Theta_' + Theta +'_seed_' + seed + '.mat'
        elif int(Tswitch) == 1 or int(Tswitch) == 2 or int(Tswitch) == 3:
            filename = 'output_2D_LT_VV_T0_' + Tamp + '_G_' + G + '_phi_' + phi + '_psi_' + psi + '_ReynoldsSett_' + ReynoldsSett + '_Nc_' + Nstr + '_HfillN_' + HfillN + '_StreamN_' + StreamN + '_sigma_erf_' + sigma_erf + '_NDS_' + NondimStiff + '_PMM_' + PMM + '_en_' + en + '_ic_' + ic + '_TS_' + Tswitch + '_seed_' + seed + '.mat'
        elif int(Tswitch) == 0:
            filename = 'output_2D_LT_VV_T0_' + Tamp + '_G_' + G + '_phi_' + phi + '_psi_' + psi + '_ReynoldsSett_' + ReynoldsSett + '_Nc_' + Nstr + '_HfillN_' + HfillN + '_StreamN_' + StreamN + '_sigma_erf_' + sigma_erf + '_NDS_' + NondimStiff + '_PMM_' + PMM + '_en_' + en + '_ic_' + ic + '_seed_' + seed + '.mat'
        
        G = float(G)
        phi = float(phi)
        gamma = float(psi) / (float(G) ** 2)
        Tamp = float(Tamp)
        ReynoldsSett = float(ReynoldsSett)
        en = float(en)
        HfillN = float(HfillN)
        NondimStiff = float(NondimStiff)
        sigma_erf = float(sigma_erf)
        PMM = float(PMM)
        ic = int(ic)
        Tswitch = int(Tswitch)
        Theta = float(Theta)
        seed = int(seed)
        
        ##***## CHECK SAVEPATH ##***##
        if (ic == 3):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_SB/'
        elif (ic == 1):
            if (os.path.exists("/home/fas/ohern/pw374/project")): # Grace
                savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_ST/' # Grace
            else:
                savepath = '/gpfs/ysm/project/pw374/2D_thermo_LT_ST/' # Farnam
        elif (ic == 0):
            if (os.path.exists("/home/fas/ohern/pw374/project")): # Grace
                savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_RD/' # Grace
            else:
                savepath = '/gpfs/ysm/project/pw374/2D_thermo_LT_RD/' # Farnam
            savepath = '/gpfs/ysm/scratch60/pw374/' # Farnam - scratch60
        elif (ic == 4):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_VAR_SB/'
        elif (ic == 2):
            savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT_VAR_ST/'
        
        # savepath = '/home/fas/ohern/pw374/project/2D_thermo_LT/'
        
    else:        
        gamma = float(psi) / (float(G) ** 2) # psi is the ratio of m_{l} / m_{s}

        N = int(HfillN) * int(StreamN)
        filename = 'temp_' + str(Tamp) +  '_HfillN_' + str(HfillN) + '_NDF_' + str(NondimStiff) + '_TS_' + str(Tswitch) + '_Theta_' + str(Theta) + '.mat'
        if (os.path.exists("E:/Matlab_output/local/")): # Windows
            savepath = 'E:/Matlab_output/local/'
        elif (os.path.exists("/Users/philipwang/Desktop/")): # Mac
            savepath = '/Users/philipwang/Desktop/'
        else: # Linux
            savepath = '/home/philip/Desktop/'

            ##### Locate save destination and check existence #####
    destination = savepath + filename        
    if (DebugTag == False and os.path.isfile(destination)):
        print(filename + ' executed and exist!')
        return None
    
    print(filename)
    
    plotit = DebugTag
    wtime1 = clock()  
    
    timestamp()
    
    np.random.seed(seed)
    
    ##### Experimental parameters #####
    D = 1
    g = 0.1
    
    Nt = 1e8
    Nt_prep = Nt / 10
    
    #if (Tamp < 0.100):
    #    Nt = 2.5e7
    #else:
    #    Nt = 2.5e7
    
    if (DebugTag == True):
        Nt = 1e6
        Nt_prep = Nt / 10
    
    Dfluc = 0.2
    
    Nbig, Nsmall, Dbig, Dsmall, rho_s, rho_l, Dn, rho, Vol, m, R_eff, R_hyd, R2n, ind_big, ind_small = (
            MDini.particle_create(N, phi, G, gamma, ic))
    
    ##### Initialization #####
    
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
    
    ##### Simulation parameters #####
    m_sum = np.sum(m)
    m_mean = np.mean(m)
    
    m_plate = m_sum * PMM # Plate Mass Multiplier (PMM)
    
    P_plate = m_plate * g / Lx
    
    # OLD #
    # K = NondimStiff * g * np.pi * (D * D / 4) / D
    # B = np.sqrt(g / D, dtype=np.float64) * np.pi / 4 * D * D * ReynoldsSett
    # B_pour = np.sqrt(g / D) * np.pi / 4 * D * D * 0.5   
    # OLD #
    
    K = NondimStiff * P_plate * np.pi * (D * D / 4) / D
    B = np.sqrt(g / D, dtype=np.float64) * np.pi / 4 * D * D * ReynoldsSett
    B_pour = np.sqrt(g / D) * np.pi / 4 * D * D * 0.5
    
    N_per_coll = 50
    dt = 1 / N_per_coll * 2 * np.pi * np.sqrt(np.pi * D * D / 4 / K, dtype=np.float64)
    
    tau_c = np.pi * np.sqrt(np.pi * D * D / 4 / K)
    gv = -2 * np.log(en) / tau_c
    
    gv_pour = -2 * np.log(0.85) / tau_c
    
    ##### draw particles #####
    if (plotit):
        draw_particle(x, y, Lx, N, R_eff)    
    ##### end draw #####    
    
    ####################################
    
    vx = np.sqrt(0.0001, dtype=np.float64) * D * np.random.randn(N)
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
            MDF.pour_particle(dt, Nt_prep, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, gv_pour, a_thresh, v_thresh, plotit, BottomWallTag))
    x_start = x[0:N_org]
    y_start = y[0:N_org]
    nt_start = nt
    ##### draw particles #####
    if (plotit):
        #draw_particle(x, y, Lx, N, R_eff)   
        draw_particle_w_wall(x, y, Lx, N, N_org, R_eff)
    ##### end draw #####
    
    ##### add top free wall #####
    m_wall = m_plate
    y_wall = np.amax(y_start) + 2 * Dbig
    x_wall = 0.0 # infinitely long wall, for broadcasting consideration
    D_wall = Dsmall
    
    vx_wall = 0.0
    vy_wall = 0.0
    ax_wall = 0.0
    ay_wall = 0.0
    
    x = np.append(x, x_wall)
    y = np.append(y, y_wall)
    vx = np.append(vx, vx_wall)
    vy = np.append(vy, vy_wall)
    ax = np.append(ax, ax_wall)
    ay = np.append(ay, ay_wall)
    ax_old = np.append(ax_old, ax_wall)
    ay_old = np.append(ay_old, ay_wall)
    m = np.append(m, m_wall)    
    Dn = np.append(Dn, D_wall)    
    
    Nc_org = N
    N = N + 1
    
    ##### Drop the top wall #####
    nt, x, y, vx, vy, ax, ay, ax_old, ay_old = (
            MDF.drop_wall(dt, Nt_prep, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, gv_pour, a_thresh, v_thresh, plotit))
    
    if (nt < Nt_prep):
        print('stable')
    else:
        print('bad drop top wall')
    
    ##### draw particles #####
    if (plotit):
        #draw_particle(x, y, Lx, N, R_eff)   
        draw_particle_w_wall(x, y, Lx, N, N_org, R_eff)
    ##### end draw #####
    
    ##### Heat the particles #####
    # If Tswitch = 4, Tamp becomes Tsmall
    nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, x_data, y_data, vx_data, vy_data, dy_data_proc, y_data_proc, y_wall_data, ph, T_glb, Tx_glb, Ty_glb, T_big_glb, T_small_glb, Ek_glb, Ekx_glb, Eky_glb, Ek_big_glb, Ek_small_glb, Tx_big_glb, Ty_big_glb, Tx_small_glb, Ty_small_glb, x_data_plot, y_data_plot = (
            MDF.langevin_thermostat_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, Nc_org, R_eff, K, B, Tamp, gv, a_thresh, v_thresh, Tswitch, Theta, N_per_coll, plotit))
    
    #Ts_Tl = 0.2
    
    #nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, T, Ek, x_data, y_data, vx_data, vy_data, ph, T_big, T_small, y_data_proc = (
    #        MDF.langevin_thermostat_unbalanced_VV(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, Nc_org, R_eff, K, B, Tamp, Ts_Tl, gv, a_thresh, v_thresh, plotit))
    
    bnlmt = np.amax(y_data) * 1.1
    bnlmt = int(bnlmt)
    if (np.mod(bnlmt, 2) == 1):
        bnlmt = bnlmt + 1 # to ensure even number
    bw = 2
    
    # Get time evolving packing fraction
    Vol = Vol[0:N_org]
    VDen_all, VDen_big, VDen_small, ph_all_t = MDF.preproc_2D_VolDensity_w_ph(Vol, y_data, bnlmt, bw, ind_big, Lx)    
    
    # Get time evolving layered average velocity
    vys_big, vys_small, vys_all, vy_prime, vy_prime_species = MDF.preproc_2D_vs(vy_data, Vol, ind_big, y_data, bnlmt, bw)
    vxs_big, vxs_small, vxs_all, vx_prime, vx_prime_species = MDF.preproc_2D_vs(vx_data, Vol, ind_big, y_data, bnlmt, bw)
    
    m = m[0:N_org]
    MDen_all, MDen_big, MDen_small = MDF.preproc_2D_MDensity(m, y_data, bnlmt, bw, ind_big, Lx)
    NDen_all, NDen_big, NDen_small = MDF.preproc_2D_NDensity(m, y_data, bnlmt, bw, ind_big, Lx) 
    
    # Get time evolving layered kinetic energy and temperature
    Eky_all, Eky_big, Eky_small, Ty_all, Ty_big, Ty_small = MDF.preproc_2D_T(vy_prime, vy_prime_species, m, Vol, ind_big, y_data, bnlmt, bw)        
    Ekx_all, Ekx_big, Ekx_small, Tx_all, Tx_big, Tx_small = MDF.preproc_2D_T(vx_prime, vx_prime_species, m, Vol, ind_big, y_data, bnlmt, bw)
    Ekxy_all, Ekxy_big, Ekxy_small, Txy_all, Txy_big, Txy_small = MDF.preproc_2D_Txy(vx_prime, vy_prime, vx_prime_species, vy_prime_species, m, Vol, ind_big, y_data, bnlmt, bw)
    
    # Get time evolving layered kinetic stress
    KSyy_all, KSyy_big, KSyy_small, KSxx_all, KSxx_big, KSxx_small, KSxy_all, KSxy_big, KSxy_small = MDSTR.preproc_2D_Ek2KineticStress(Eky_all, Eky_big, Eky_small, Ekx_all, Ekx_big, Ekx_small, Ekxy_all, Ekxy_big, Ekxy_small, Lx, bw)
    
    # Get time evolving layered contact stress
    Sxx_all, Sxy_all, Syy_all, Sxx_big, Sxy_big, Syy_big, Sxx_small, Sxy_small, Syy_small, P_all, P_big, P_small, tau_all, tau_big, tau_small = (
            MDSTR.preproc_2D_ContactStressMain_no_bottom(N_org, x_data, y_data, Lx, Ly, K, R_eff, y_wall, bnlmt, bw, ind_big))
    
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
    
    # VDen_all, VDen_big, VDen_small, MDen_all, MDen_big, MDen_small, ph_all_t = MDF.preproc_2D_AccurateDensity(Vol, Dn, y_data, bnlmt, bw, ind_big, rho_l, rho_s, Lx)
    
    R_eff = R_eff[0:N_org]
    R_hyd = R_hyd[0:N_org]
    N = N_org
    
    ##### Get ISF #####    
    ISF_y_data = ISF.ISF_main(y_data_proc, Ly, bw)
    bw_ISF = 2 * Dsmall
    ISF_y_data_2Dsmall = ISF.ISF_main(y_data_proc, Ly, bw_ISF)
    bw_ISF = 2 * Dbig
    ISF_y_data_2Dbig = ISF.ISF_main(y_data_proc, Ly, bw_ISF)
    print('ISF done.')
    
    ##### Get MSD #####
    # MSD_y_glb, MSD_l_y_glb, MSD_s_y_glb, MSD_corrected_y_glb, MSD_corrected_l_y_glb, MSD_corrected_s_y_glb = MSD_glb.main(y_data_proc, dy_data_proc, Ly, ind_big)        
    # MSD_y_data, MSD_l_y_data, MSD_s_y_data, MSD_corrected_y_data, MSD_l_corrected_y_data, MSD_s_corrected_y_data = MSD_c.main(y_data_proc, dy_data_proc, Ly, bw, ind_big)
    MSD_y_glb, MSD_l_y_glb, MSD_s_y_glb = MSD_glb.main(y_data_proc, dy_data_proc, Ly, ind_big)
    timestamp()
    print('Gloabl MSD done.')
    bw = 2 * D
    MSD_y_data_2D, MSD_l_y_data_2D, MSD_s_y_data_2D = MSD_local.main(y_data_proc, dy_data_proc, Ly, bw, ind_big)
    timestamp()
    print('Local MSD - 2D done.')
    #bw = 3 * D
    #MSD_y_data_3D, MSD_l_y_data_3D, MSD_s_y_data_3D = MSD_local.main(y_data_proc, dy_data_proc, Ly, bw, ind_big)
    #timestamp()
    #print('Local MSD - 3D done.')
    #bw = 4 * D
    #MSD_y_data_4D, MSD_l_y_data_4D, MSD_s_y_data_4D = MSD_local.main(y_data_proc, dy_data_proc, Ly, bw, ind_big)
    #timestamp()
    #print('Local MSD - 4D done.')
    
    if (DebugTag):
        print(Ly)
        print(rho_l/rho_s)
        print(gamma)
        
        print(mbig/msmall)
        print(psi)
        
        print('Average global T = %2.2f' % np.mean(T_glb))
        print('Average global Tl = %2.2f' % np.mean(T_big_glb))
        print('Average global Ts = %2.2f' % np.mean(T_small_glb))
        
        print('Average wall position = %2.2f' % np.mean(y_wall_data))
        #print(np.mean(ph))
        #print(Volbig/Volsmall)
        #print(y_data_proc[0])
    if (seed != 100):
        (sio.savemat(destination, {'x_start': x_start, 'y_start': y_start, 'nt_start': nt_start, 'y_wall_data': y_wall_data
                                   , 'x_stop': x_stop, 'y_stop': y_stop, 'nt_stop': nt_stop, 'N':N
                                   , 'Dn': Dn, 'm': m, 'Vol': Vol, 'R_eff': R_eff, 'R_hyd': R_hyd, 'HfillN': HfillN
                                   , 'G': G, 'NondimStiff': NondimStiff, 'seed': seed, 'g': g
                                   , 'Lx': Lx, 'Ly': Ly, 'en': en, 'tau_c': tau_c, 'gv': gv
                                   , 'gamma': gamma, 'phi': phi, 'Theta': Theta
                                   , 'Nsmall': Nsmall, 'Nbig': Nbig, 'Dsmall': Dsmall, 'Dbig': Dbig
                                   , 'rho': rho, 'rho_s': rho_s, 'rho_l': rho_l
                                   , 'CM_big': CM_big, 'CM_small': CM_small
                                   , 'T_glb': T_glb, 'Tx_glb': Tx_glb, 'Ty_glb': Ty_glb, 'T_big_glb': T_big_glb, 'T_small_glb': T_small_glb
                                   # , 'Ek_glb': Ek_glb, 'Ekx_glb': Ekx_glb, 'Eky_glb': Eky_glb, 'Ek_big_glb': Ek_big_glb, 'Ek_small_glb': Ek_small_glb
                                   , 'Tx_big_glb': Tx_big_glb, 'Ty_big_glb': Ty_big_glb, 'Tx_small_glb': Tx_small_glb, 'Ty_small_glb': Ty_small_glb
                                   , 'Ty_all': Ty_all, 'Ty_big': Ty_big, 'Ty_small': Ty_small, 'Tx_all': Tx_all, 'Tx_big': Tx_big, 'Tx_small': Tx_small
                                   , 'Txy_all': Txy_all, 'Txy_big': Txy_big, 'Txy_small': Txy_small
                                   #, 'Ek_big': Ek_big, 'Ek_small': Ek_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small, 'vxs_big': vxs_big, 'vxs_small': vxs_small
                                   #, 'vys_all': vys_all, 'vxs_all': vxs_all
                                   ,'ph_all_t': ph_all_t
                                   #, 'Eky_all': Eky_all, 'Eky_big': Eky_big, 'Eky_small': Eky_small, 'Ty_all': Ty_all, 'Ty_big': Ty_big, 'Ty_small': Ty_small
                                   #, 'Ekx_all': Ekx_all, 'Ekx_big': Ekx_big, 'Ekx_small': Ekx_small, 'Tx_all': Tx_all, 'Tx_big': Tx_big, 'Tx_small': Tx_small
                                   , 'dt': dt, 'Nt': Nt, 'B_pour': B_pour, 'B': B, 'K': K, 'static_equilibrium': static_equilibrium
                                   #, 'mean_vxs_big': mean_vxs_big, 'mean_vxs_small': mean_vxs_small, 'mean_vys_big': mean_vys_big, 'mean_vys_small': mean_vys_small
                                   #, 'mean_xs_big': mean_xs_big, 'mean_xs_small': mean_xs_small, 'mean_ys_big': mean_ys_big, 'mean_ys_small': mean_ys_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small
                                   #, 'x_data': x_data, 'y_data': y_data
                                   , 'ISF_y_data': ISF_y_data, 'ISF_y_data_2Dsmall': ISF_y_data_2Dsmall, 'ISF_y_data_2Dbig': ISF_y_data_2Dbig
                                   , 'MSD_y_glb': MSD_y_glb, 'MSD_l_y_glb': MSD_l_y_glb, 'MSD_s_y_glb': MSD_s_y_glb
                                   #, 'MSD_corrected_y_glb': MSD_corrected_y_glb, 'MSD_corrected_l_y_glb': MSD_corrected_l_y_glb, 'MSD_corrected_s_y_glb': MSD_corrected_s_y_glb
                                   #, 'MSD_y_data': MSD_y_data, 'MSD_l_y_data': MSD_l_y_data, 'MSD_s_y_data': MSD_s_y_data                                   
                                   #, 'MSD_corrected_y_data': MSD_corrected_y_data, 'MSD_l_corrected_y_data': MSD_l_corrected_y_data, 'MSD_s_corrected_y_data': MSD_s_corrected_y_data                            
                                   , 'MSD_y_data_2D': MSD_y_data_2D, 'MSD_l_y_data_2D': MSD_l_y_data_2D, 'MSD_s_y_data_2D': MSD_s_y_data_2D
                                   #, 'MSD_y_data_3D': MSD_y_data_3D, 'MSD_l_y_data_3D': MSD_l_y_data_3D, 'MSD_s_y_data_3D': MSD_s_y_data_3D
                                   #, 'MSD_y_data_4D': MSD_y_data_4D, 'MSD_l_y_data_4D': MSD_l_y_data_4D, 'MSD_s_y_data_4D': MSD_s_y_data_4D
                                   , 'KSyy_all': KSyy_all, 'KSyy_big': KSyy_big, 'KSyy_small': KSyy_small
                                   , 'KSxx_all': KSxx_all, 'KSxx_big': KSxx_big, 'KSxx_small': KSxx_small
                                   , 'KSxy_all': KSxy_all, 'KSxy_big': KSxy_big, 'KSxy_small': KSxy_small
                                   , 'Syy_all': Syy_all, 'Syy_big': Syy_big, 'Syy_small': Syy_small
                                   , 'Sxx_all': Sxx_all, 'Sxx_big': Sxx_big, 'Sxx_small': Sxx_small
                                   , 'Sxy_all': Sxy_all, 'Sxy_big': Sxy_big, 'Sxy_small': Sxy_small
                                   , 'P_all': P_all, 'P_big': P_big, 'P_small': P_small
                                   , 'tau_all': tau_all, 'tau_big': tau_big, 'tau_small': tau_small
                                   #, 'vx_data': vx_data, 'vy_data': vy_data
                                   , 'x_data': x_data_plot, 'y_data': y_data_plot
                                   , 'ph': ph, 'P_plate': P_plate, 'PMM': PMM                                   
                                   , 'ind_big':ind_big, 'ind_small':ind_small
                                   , 'MDen_all': MDen_all, 'MDen_big': MDen_big, 'MDen_small': MDen_small
                                   , 'VDen_all': VDen_all, 'VDen_big': VDen_big, 'VDen_small': VDen_small
                                   , 'NDen_all': NDen_all, 'NDen_big': NDen_big, 'NDen_small': NDen_small}))
    
    wtime2 = clock()
    print('')
    print('Elapsed time: %g.' % (wtime2 - wtime1))
    timestamp()
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


#(G,phi,psi,Tamp,ReSet,en,HfillN,StreamN,NondimStiff,sigma_erf,PMM,ic,Tswitch,Theta,seed)
# ic only has 0 (RD), 1 (ST), 3(SB) options
# Tswitch: # Tswitch=0, balanced. Tswitch=1, large in x+y, Tswitch=2, large in x, Tswitch=3, large in y
#main(2.00, 0.50, 4.0, 0.10, 2.5, 1.0, 10, 10, 10000, 0.01, 1, 0, 0, 1.0, 5) # uncomment to make it a function
if __name__ == "__main__": # uncomment this and the next line for cluster usage
    main()
    