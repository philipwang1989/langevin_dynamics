
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
from numba import jit

#def main(G, phi, psi, Tamp, ReynoldsSett, en, HfillN, NondimStiff, sigma_erf, ic, seed):
def main():    
    #DebugTag = True
    DebugTag = False
    
    #BottomWallTag = True
    BottomWallTag = False
    
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
        
        N = int(HfillN) * int(HfillN) // 2
        Nstr = str(N)        
        
        # gamma = rho_l/rho_s = (m_l/m_s)/(V_l/V_s)
        
        gamma = float(psi) / (float(G) ** 2) # psi if the ratio of m_{l} / m_{s}
        gamma = f"{gamma:.2f}"
        
        filename = 'output_2D_LT_T0_' + Tamp + '_G_' + G + '_phi_' + phi + '_psi_' + psi + '_ReynoldsSett_' + ReynoldsSett + '_Nc_' + Nstr + '_HfillN_' + HfillN + '_sigma_erf_' + sigma_erf + '_ic_' + ic + '_seed_' + seed + '.mat'
        
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
        
        N = int(HfillN) * int(HfillN) // 2
        filename = 'temp.mat'
        savepath = 'E:/Matlab_output/local/'            
    
    destination = savepath + filename
    
    plotit = DebugTag
    wtime1 = clock()  
    
    timestamp()
    
    np.random.seed(seed)
    
    ##### Experimental parameters #####
    D = 1
    g = 0.1
    
    Nt = 5e7
    
    #if (Tamp < 0.30): # if low temperature
    #    Nt = 1e7
    #else: # else is high temperature
    #    Nt = 5e7
    
    #if (G < 2.1):
    #    Nt = 1e7 # Number of time step
    #else:
    #    Nt = 5e6
    
    if (DebugTag == True):
        Nt = 5e4
    
    K = NondimStiff * g * np.pi * (D * D / 4) / D
    B = np.sqrt(g / D, dtype=np.float64) * np.pi / 4 * D * D * ReynoldsSett
    B_pour = np.sqrt(g / D) * np.pi / 4 * D * D * 0.5
    
    ##### Simulation parameters #####
    N_per_coll = 20
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
        x, y, Lx, Ly = initial_preparation_RD(N, G, D, R_hyd, HfillN, 0)
    elif (ic == 1): # fragile
        x, y, Lx, Ly = initial_preparation_ST(N, Nb, Ns, Dbig_max, D, R_hyd, HfillN)
    elif (ic == 2): # varying initial prep.
        x, y, Lx, Ly = initial_preparation_VAR_ST(N, Nb, Ns, G, D, R_hyd, StreamN, HfillN, sigma_erf, Volbig, Volsmall)
    elif (ic == 3): # prearmored
        x, y, Lx, Ly = initial_preparation_SB(N, Nb, Ns, G, Dsmall_max, R_hyd, HfillN)
    elif (ic == 4): # small at the bottom VAR
        x, y, Lx, Ly = initial_preparation_VAR_SB(N, Nb, Ns, G, D, R_hyd, StreamN, HfillN, sigma_erf, Volbig, Volsmall)    
    
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
            pour_particle(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, a_thresh, v_thresh, plotit, BottomWallTag))
    x_start = x[0:N_org]
    y_start = y[0:N_org]
    nt_start = nt
    ##### draw particles #####
    if (plotit):
        #draw_particle(x, y, Lx, N, R_eff)   
        draw_particle_w_wall(x, y, Lx, N, N_org, R_eff)
    ##### end draw #####
    
    ##### Heat the particles #####
    nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, T, T_mass, T_big, T_mass_big, T_small, T_mass_small, Ek, Ek_big, Ek_small, x_data, y_data, vx_data, vy_data = (
            langevin_thermostat(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B, Tamp, gv, a_thresh, v_thresh, plotit, BottomWallTag))
    
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
    m = m[0:N_org]
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
        print(np.mean(T_mass))
        print(np.mean(T_big))
        print(np.mean(T_mass_big))
        print(np.mean(T_small))
        print(np.mean(T_mass_small))
        #print(Volbig/Volsmall)
    
    (sio.savemat(destination, {'x_start': x_start, 'y_start': y_start, 'nt_start': nt_start
                                   , 'x_stop': x_stop, 'y_stop': y_stop, 'nt_stop': nt_stop, 'N':N
                                   , 'Dn': Dn, 'm': m, 'Vol': Vol, 'R_eff': R_eff, 'R_hyd': R_hyd, 'HfillN': HfillN
                                   , 'G': G, 'NondimStiff': NondimStiff, 'seed': seed
                                   , 'Lx': Lx, 'Ly': Ly, 'en': en, 'tau_c': tau_c, 'gv': gv
                                   , 'gamma': gamma, 'phi': phi
                                   , 'Nsmall': Nsmall, 'Nbig': Nbig, 'Dsmall': Dsmall, 'Dbig': Dbig
                                   , 'rho': rho, 'rho_s': rho_s, 'rho_l': rho_l
                                   , 'CM_big': CM_big, 'CM_small': CM_small, 'T': T, 'Ek': Ek
                                   , 'T_big': T_big, 'T_small': T_small, 'Ek_big': Ek_big, 'Ek_small': Ek_small
                                   , 'dt': dt, 'Nt': Nt, 'B_pour': B_pour, 'B': B, 'K': K, 'static_equilibrium': static_equilibrium
                                   #, 'mean_vxs_big': mean_vxs_big, 'mean_vxs_small': mean_vxs_small, 'mean_vys_big': mean_vys_big, 'mean_vys_small': mean_vys_small
                                   #, 'mean_xs_big': mean_xs_big, 'mean_xs_small': mean_xs_small, 'mean_ys_big': mean_ys_big, 'mean_ys_small': mean_ys_small
                                   #, 'vys_big': vys_big, 'vys_small': vys_small
                                   #, 'x_data': x_data, 'y_data': y_data
                                   #, 'MSD_x': MSD_x, 'MSD_y': MSD_y
                                   , 'x_data': x_data, 'y_data': y_data, 'vx_data': vx_data, 'vy_data': vy_data
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
        x_list, y_list = np.mgrid[G:Lx:G, G:(Ly - G):G] # mgrid stop is not inclusive
        x_list = np.transpose(x_list)
        y_list = np.transpose(y_list)
        x_length = np.unique(x_list)
        x_length = x_length.size
        y_length = np.unique(y_list)
        y_length = y_length.size
        print(x_length)
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
            rand_list_x[0] = temp_list_x[0]
            temp_list_y = y_list[0:cum_sum_layers[0]][:]
            rand_list_y[0] = temp_list_y[0]            
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
            # bug in here
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

def pour_particle(dt, Nt, m, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B_pour, a_thresh, v_thresh, plotit, BottomWallTag):
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
        
        Fx, Fy = force(Fx, Fy, N, x, y, Lx, Ly, K, R_eff)
        
        # ph = ph / (R * R) / np.pi
    
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

@jit
def langevin_thermostat(dt, Nt, m, ind_big, Dn, x, y, vx, vy, ax, ay, ax_old, ay_old, Lx, Ly, N, N_org, R_eff, K, B, Tamp, gv, a_thresh, v_thresh, plotit, BottomWallTag):
    ind_big = np.asarray(ind_big, dtype=bool)
    Nbig = sum(ind_big)
    Nsmall = N_org - Nbig
    T0 = Tamp
    epsilon = B / m # unit: 1/time
    dt_half = np.sqrt(dt)
    g = 0.1
    nt = 0
    saveskip = 1000 # corresponds to 50 tau_c
    saveskip_MSD = 20 # corresponds to 1 tau_c
    count = 0
    datalength = int(Nt / saveskip)
    acc_list = np.zeros(datalength)
    T_list = np.zeros(datalength)
    T_mass_list = np.zeros(datalength)
    Ek_list = np.zeros(datalength)
    CM_big = np.zeros(datalength)
    CM_small = np.zeros(datalength)
    Ek_big = np.zeros(datalength)
    Ek_small = np.zeros(datalength)
    T_big = np.zeros(datalength)
    T_mass_big = np.zeros(datalength)
    T_small = np.zeros(datalength)
    T_mass_small = np.zeros(datalength)
    
    x_data = np.zeros((datalength, N_org))
    y_data = np.zeros((datalength, N_org))
    vx_data = np.zeros((datalength, N_org))
    vy_data = np.zeros((datalength, N_org))
    
    #xdata_MSD = []
    #ydata_MSD = []
    
    vx = vx - np.mean(vx)
    vy = vy - np.mean(vy)
    
    flag = False
    while (nt < Nt):
        
        x = x + vx * dt
        y = y + vy * dt
        
        # x = np.mod(x, Lx)
        # y = np.mod(y, Ly)
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        
        if (gv == 0):
            Fx, Fy = force(Fx, Fy, N, x, y, Lx, Ly, K, R_eff)
        else:
            Fx, Fy = force_rest(Fx, Fy, N, x, y, vx, vy, Lx, Ly, K, m, R_eff, gv)
    
        # bottom wall
        iib = (y < Dn/2)
        dw = y[iib] - Dn[iib] / 2
        Fy[iib] = Fy[iib] - K * dw
        
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
        ax = Fx / m
        ay = Fy / m - g
        
        #ax[iib] = 0.0
        if (BottomWallTag):
            ax[N_org:N] = 0.0 # fixed the bottom wall particle
            ay[N_org:N] = 0.0 # fixed the bottom wall particle
    
        sigma = np.sqrt(2 * T0 * epsilon / m)
        eta = np.random.randn(N)
        
        vx = vx + ax * dt - epsilon * dt * vx + dt_half * sigma * eta
        vy = vy + ay * dt - epsilon * dt * vy + dt_half * sigma * eta
        
        #vx = vx + (ax_old + ax) * dt / 2
        #vy = vy + (ay_old + ay) * dt / 2
        
        #vx[iib] = 0.0
        if (BottomWallTag):
            vx[N_org:N] = 0.0 # fixed the bottom wall particle
            vy[N_org:N] = 0.0 # fixed the bottom wall particle
    
        ax_old = ax
        ay_old = ay
        
        #if (nt % saveskip_MSD == 0):
        #    xdata_MSD.append(x)
        #    ydata_MSD.append(y)
        
        if (nt % saveskip == 0):
            y_CM = y[0:N_org]   
            m_CM = m[0:N_org]
            
            d_acc = np.sqrt(np.amax(ax * ax + ay * ay))
            vx_mean_mass = (sum(m_CM[ind_big]) / sum(m_CM[0:N_org])) * np.mean(vx[ind_big]) + (sum(m_CM[~ind_big]) / sum(m[0:N_org])) * np.mean(vx[~ind_big])
            vy_mean_mass = (sum(m_CM[ind_big]) / sum(m_CM[0:N_org])) * np.mean(vy[ind_big]) + (sum(m_CM[~ind_big]) / sum(m[0:N_org])) * np.mean(vy[~ind_big])
            Ek_temp = (0.5 * m[0:N_org] * ((vx[0:N_org] - np.mean(vx[0:N_org])) ** 2 + (vy[0:N_org] - np.mean(vy[0:N_org])) ** 2))
            T_temp = sum(Ek_temp) / N_org
            CM_big[count] = np.mean(y_CM[ind_big])
            CM_small[count] = np.mean(y_CM[~ind_big])
            T_list[count] = T_temp
            Ek_mass_temp = 0.5 * m[0:N_org] * ((vx[0:N_org] - vx_mean_mass) ** 2 + (vy[0:N_org] - vy_mean_mass) ** 2)
            T_mass_list[count] = sum(Ek_mass_temp) / N
            Ek_list[count] = sum(Ek_temp)
            Ek_big[count] = sum(Ek_temp[ind_big])
            Ek_small[count] = sum(Ek_temp[~ind_big])
            T_big[count] = Ek_big[count] / Nbig
            T_small[count] = Ek_small[count] / Nsmall
            T_mass_big[count] = sum(Ek_mass_temp[ind_big]) / Nbig
            T_mass_small[count] = sum(Ek_mass_temp[~ind_big]) / Nsmall
            
            x_data[count] = x
            y_data[count] = y
            vx_data[count] = vx
            vy_data[count] = vy
            
            if (nt % 10000 == 0 and plotit):
                print(d_acc, T_temp)
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
    
    return nt, x, y, vx, vy, ax, ay, ax_old, ay_old, CM_big, CM_small, T_list, T_mass_list, T_big, T_mass_big, T_small, T_mass_small, Ek_list, Ek_big, Ek_small, x_data, y_data, vx_data, vy_data

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
                    Fx[nn] = Fx[nn] + F * dx - gv * m_red * v_dot_r * dx / dnm2
                    Fx[mm] = Fx[mm] - F * dx + gv * m_red * v_dot_r * dx / dnm2
                    Fy[nn] = Fy[nn] + F * dy - gv * m_red * v_dot_r * dy / dnm2
                    Fy[mm] = Fy[mm] - F * dy + gv * m_red * v_dot_r * dy / dnm2
    return Fx, Fy

# (G,phi,psi,Tamp,ReSet,en,HfillN,NondimStiff,sigma_erf,ic,seed)
# ic only has 0 (RD), 1 (ST), 3(SB) options
#main(2.0, 0.5, 1.0, 0.01, 2.5, 1, 24, 1000, 0.01, 0, 1) # uncomment to make it a function
if __name__ == "__main__": # uncomment this and the next line for cluster usage
    main()
    


# In[ ]:



