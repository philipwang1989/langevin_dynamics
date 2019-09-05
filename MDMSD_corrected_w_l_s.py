# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:44:51 2018

This code is to include diffusivity for large and small particles

@author: Philip
"""

import numpy as np
from numba import jit
import scipy.io as sio

def main(y, dy, Ly, layer_width, ind_big):
    N_dt = len(y) # number of time delays, also need info of the true time delay    
    
    output = []
    output_corrected = []
    output_l = []
    output_l_corrected = []
    output_s = []
    output_s_corrected = []    
    
    output.append([0])
    output_corrected.append([0])
    output_l.append([0])
    output_l_corrected.append([0])
    output_s.append([0])
    output_s_corrected.append([0])

    for i in range(0, N_dt):
        print(i)
        y_input = np.asarray(y[i])
        dy_input = np.asarray(dy[i])
        if (len(y_input) != 0):
            # print('pre_MSD initiated!')
            pre_MSD, pre_MSD_l, pre_MSD_s, pre_MSD_dy_mean, pre_MSD_l_dy_mean, pre_MSD_s_dy_mean = pre_MSD_y(y_input, dy_input, layer_width, ind_big)
            # print('pre_MSD done!')
            # print('post_MSD initiated!')
            # temp, temp_corrected = post_MSD_y(pre_MSD, pre_MSD_dy_mean, Ly)
            temp, temp_l, temp_s, temp_corrected, temp_l_corrected, temp_s_corrected = post_species_MSD_y(pre_MSD_l, pre_MSD_s, pre_MSD_l_dy_mean, pre_MSD_s_dy_mean, Ly)
            print('Post_all_MSD done!')
            output.append(temp)
            output_l.append(temp_l)
            output_s.append(temp_s)
            output_corrected.append(temp_corrected)
            output_l_corrected.append(temp_l_corrected)
            output_s_corrected.append(temp_s_corrected)
        else:
            output.append([])
            output_l.append([])
            output_s.append([])
            output_corrected.append([])
            output_l_corrected.append([])
            output_s_corrected.append([])
        # path = '/Users/philipwang/Desktop/output_test_' + str(i) + '.mat'
        # sio.savemat(path,{'pre_MSD_dy_mean': pre_MSD_dy_mean, 'pre_MSD_l_dy_mean': pre_MSD_l_dy_mean, 'pre_MSD_s_dy_mean': pre_MSD_s_dy_mean})
    del output[0]
    del output_corrected[0]
    del output_l[0]
    del output_s[0]
    del output_l_corrected[0]
    del output_s_corrected[0]
        
    print('Done appending!')
    return output, output_l, output_s, output_corrected, output_l_corrected, output_s_corrected

def pre_MSD_y(y, dy, layer_width, ind_big):
    
    # Step 1: Binning in time and space
    [T, N] = y.shape 
    N_layers = int(np.ceil(np.amax(np.amax(y)) / layer_width))
    # layer_width = 2
    dy = np.asarray(dy)
    # print(dy.shape)
    particle = []
    particle_layer_list = []
    particle_dy_mean = []
    
    for i in range (0, N):
        layer = []
        layer_list = []
        dy_mean = []
        count = 0 # number of timesteps for the particle to stay in the same layer
        count1 = 0 # number of fake particle to enter a new layer
        for j in range (0, T):
            ind_layer = np.ceil(y[j][i] / layer_width) # start from 1
            if (j == 0):
                ind_layer_old = ind_layer # 1st time steps
            if (j > 0):
                if (ind_layer_old != ind_layer):
                    count = 0
                    count1 = count1 + 1
                ind_layer_old = ind_layer
            # grow list here
            if (count1 >= len(layer)):
                layer.append([0])
                layer_list.append([0])                
                dy_mean.append([0])
            layer[count1].append(y[j][i])
            layer_list[count1].append(ind_layer)
            
            # print(len(dy[0][:]))
            # print(len(dy[:][0]))            
            dy_mean[count1].append(dy[j][int(ind_layer) - 1])
            count = count + 1            
        for ii in range (0, len(layer)):            
            del layer[:][ii][0]
            del layer_list[:][ii][0]
            del dy_mean[:][ii][0]
        if (i >= len(particle)):
            particle.append([0])
            particle_layer_list.append([0])
            particle_dy_mean.append([0])
        # each particle[i] saves the trajectory of this particle
        # each particle_layer_list[i] saves the time history of the layers the particle is locating
        particle[i].append(layer)
        particle_layer_list[i].append(layer_list)
        particle_dy_mean[i].append(dy_mean)
        del particle[i][0]
        del particle_layer_list[i][0]
        del particle_dy_mean[i][0]
    
    # del layer
    
    # Step 2: organize the data    
    layer = []
    layer_l = []
    layer_s = []
    layer_dy_mean = []
    layer_l_dy_mean = []
    layer_s_dy_mean = []
    
    for i in range (0, N_layers):
        layer.append([0])
        layer_l.append([0])
        layer_s.append([0])
        layer_dy_mean.append([0])
        layer_l_dy_mean.append([0])
        layer_s_dy_mean.append([0])
    
    for i in range(0, N):
        for j in range(0, len(particle_layer_list[i][0])):
            # if (len(particle_layer_list[i][0]) == 1):
            layer[int(particle_layer_list[i][0][j][0]) - 1].append(particle[i][0][j])
            layer_dy_mean[int(particle_layer_list[i][0][j][0]) - 1].append(particle_dy_mean[i][0][j])
            #else:
            #    layer[int(particle_layer_list[i][0][j][0]) - 1].append(particle[i][0][j])
            #    layer_dy_mean[int(particle_layer_list[i][0][j][0]) - 1].append(particle_dy_mean[i][0][j])
        if ind_big[i]:
            for j in range(0, len(particle_layer_list[i][0])):
                #if (len(particle_layer_list[i][0]) == 1):
                layer_l[int(particle_layer_list[i][0][j][0]) - 1].append(particle[i][0][j])
                layer_l_dy_mean[int(particle_layer_list[i][0][j][0]) - 1].append(particle_dy_mean[i][0][j])
        else:
            for j in range(0, len(particle_layer_list[i][0])):
                layer_s[int(particle_layer_list[i][0][j][0]) - 1].append(particle[i][0][j])
                layer_s_dy_mean[int(particle_layer_list[i][0][j][0]) - 1].append(particle_dy_mean[i][0][j])
        
    for i in range (0, N_layers):
        del layer[i][0]
        del layer_l[i][0]
        del layer_s[i][0]
        del layer_dy_mean[i][0]
        del layer_l_dy_mean[i][0]
        del layer_s_dy_mean[i][0]
    
    # sio.savemat('E:/Matlab_output/local/output_test.mat',{'particle': particle, 'particle_layer_list': particle_layer_list, 'layer': layer})
    
    return layer, layer_l, layer_s, layer_dy_mean, layer_l_dy_mean, layer_s_dy_mean

def post_species_MSD_y(preMSD_l, preMSD_s, preMSD_l_dy_mean, preMSD_s_dy_mean, Ly):
    output = []
    output_corrected = []
    output_l = []
    output_l_corrected = []
    output_s = []
    output_s_corrected = []
    
    if (len(preMSD_l) >= len(preMSD_s)):
        N_layers = len(preMSD_l)
    else:
        N_layers = len(preMSD_s)
    
    for j in range(0, N_layers): 
        # first loop through l
        MSD_all = []
        MSD_corrected_all = []
        MSD_all_all = []
        MSD_corrected_all_all = []
        # print(len(preMSD[j]))        
        for i in range(0, len(preMSD_l[j])):
            y = preMSD_l[j][i]
            dy = np.asarray(preMSD_l_dy_mean[j][i])
            # print(len(y))
            # MSD = interproc_MSD_r(y, Ly)
            # print('interproc_MSD_r done!')
            MSD, MSD_corrected = interproc_MSD_lm_corrected_r(y, dy, Ly)
            # print('interproc_MSD_r_more done!')
            if (len(MSD) > 1):
                MSD_all.append(MSD)
                MSD_corrected_all.append(MSD_corrected)
            # print(j,i)
        MSD_all_all = MSD_all
        MSD_corrected_all_all = MSD_corrected_all
        output_l.append(MSD_all)
        output_l_corrected.append(MSD_corrected_all)
        
        # then loop through s
        MSD_all = []
        MSD_corrected_all = []
        for i in range(0, len(preMSD_s[j])):
            y = preMSD_s[j][i]
            dy = np.asarray(preMSD_s_dy_mean[j][i])
            # print(len(y))
            # MSD = interproc_MSD_r(y, Ly)
            # print('interproc_MSD_r done!')
            MSD, MSD_corrected = interproc_MSD_lm_corrected_r(y, dy, Ly)
            # print('interproc_MSD_r_more done!')
            if (len(MSD) > 1):
                MSD_all.append(MSD)
                MSD_all_all.append(MSD)
                MSD_corrected_all.append(MSD_corrected)
                MSD_corrected_all_all.append(MSD_corrected)
                
        output_s.append(MSD_all)
        output_s_corrected.append(MSD_corrected_all)   
        output.append(MSD_all_all)
        output_corrected.append(MSD_corrected_all_all)
        
    return output, output_l, output_s, output_corrected, output_l_corrected, output_s_corrected

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


def post_MSD_y(preMSD, preMSD_dy_mean, Ly):
    N_layers = len(preMSD)  
    
    output = []
    output_corrected = []
    
    for j in range(0, N_layers):
        MSD_all = []
        MSD_corrected_all = []
        # print(len(preMSD[j]))
        for i in range(0, len(preMSD[j])):
            y = preMSD[j][i]
            dy = np.asarray(preMSD_dy_mean[j][i])
            # print(len(y))
            # MSD = interproc_MSD_r(y, Ly)
            # print('interproc_MSD_r done!')
            MSD, MSD_corrected = interproc_MSD_lm_corrected_r(y, dy, Ly)
            # print('interproc_MSD_r_more done!')
            if (len(MSD) > 1):
                MSD_all.append(MSD)
                MSD_corrected_all.append(MSD_corrected)
            # print(j,i)
        output.append(MSD_all)
        output_corrected.append(MSD_corrected_all)
        
    return output, output_corrected


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
