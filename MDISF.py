
# coding: utf-8

# In[ ]:

import sys
import numpy as np
from time import clock
from numba import jit
import scipy.io as sio

def ISF_main(y, Ly, D):
    N_dt = len(y)
    output = []
    
    for i in range(0, N_dt):   
        y_input = np.asarray(y[i])
        if (len(y_input) != 0):
            pre_ISF = pre_ISF_y(y_input)
            temp = post_ISF_y(pre_ISF, Ly, D)
            output.append(temp)
        else:
            output.append([])
    
    return output

def pre_ISF_y(y):
    
    # Step 1: Binning in time and space
    [T, N] = y.shape
    
    layer_width = 2
        
    particle = []
    particle_layer_list = []

    for i in range (0, N):
        layer = []
        layer_list = []
        count = 0
        count1 = 0
        for j in range (0, T):
            ind_layer = np.ceil(y[j][i] / layer_width)
            if (j == 0):
                ind_layer_old = ind_layer
            if (j > 0):
                if (ind_layer_old != ind_layer):
                    count = 0
                    count1 = count1 + 1
                ind_layer_old = ind_layer
            # grow list here
            if (count1 >= len(layer)):
                layer.append([0])
                layer_list.append([0])            
            layer[count1].append(y[j][i])
            layer_list[count1].append(ind_layer)
            count = count + 1        
        for ii in range (0, len(layer)):            
            del layer[:][ii][0]
            del layer_list[:][ii][0]        
        if (i >= len(particle)):
            particle.append([0])
            particle_layer_list.append([0])
        particle[i].append(layer)
        particle_layer_list[i].append(layer_list)
        del particle[i][0]
        del particle_layer_list[i][0]
    
    # del layer
    
    # Step 2: organize the data
    N_layers = int(np.ceil(np.amax(np.amax(y)) / layer_width))
    layer = []
    for i in range (0, N_layers):
        layer.append([0])
    
    for i in range(0, N):
        for j in range(0, len(particle_layer_list[i][0])):
            if (len(particle_layer_list[i][0]) == 1):
                layer[int(particle_layer_list[i][0][j][0]) - 1].append(particle[i][0][j])
            else:
                layer[int(particle_layer_list[i][0][j][0]) - 1].append(particle[i][0][j])
    
    for i in range (0, N_layers):
        del layer[i][0]
    
    # sio.savemat('E:/Matlab_output/local/output_test.mat',{'particle': particle, 'particle_layer_list': particle_layer_list, 'layer': layer})
    
    return layer

#@jit
def post_ISF_y(preISF, Ly, D):
    N_layers = len(preISF)
    
    ky = 2 * np.pi / D    
    
    output = []
    
    for j in range(0, N_layers):
        ISF_all = []
        for i in range(0, len(preISF[j])):
            y = preISF[j][i]
            ISF = interproc_ISF_r(y, Ly, ky)
            if (len(ISF) > 1):
                ISF_all.append(ISF)
        output.append(ISF_all)
    return output

@jit
def interproc_ISF_r(y, Ly, ky):
    counts = np.zeros(len(y))
    ISFy = np.zeros(len(y))
    for t1 in range(0, len(y) - 1):
        for t2 in range(t1 + 1, len(y)):
            y1 = y[t1]
            y2 = y[t2]
            dt_index = t2 - t1 - 1 # first index is 0
            dy = y2 - y1
            dy = dy - round(dy / Ly) * Ly
            ISFy[dt_index] = ISFy[dt_index] + np.cos(ky * dy)
            counts[dt_index] = counts[dt_index] + 1
    ind = counts > 0
    
    ISFy[ind] = ISFy[ind] / counts[ind]
    
    ISFy = ISFy[0:-1]
    
    return ISFy

#y_data = sio.loadmat('E:/Matlab_output/local/temp_0.1.mat')
#y_data = y_data['y_data_proc']

#pre_ISF = pre_ISF_y(y_input)

# wtime1 = clock()  
# output = ISF_main(y_data, 35, 2)
#output = post_ISF_y(pre_ISF, 35, 2)

# wtime2 = clock()
# print('')
# print('Elapsed time: %g.' % (wtime2 - wtime1))

# sio.savemat('E:/Matlab_output/local/output_test.mat',{'output': output})


# In[ ]:



