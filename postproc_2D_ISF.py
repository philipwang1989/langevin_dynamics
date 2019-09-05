# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:51:52 2018

@author: Philip
"""

import numpy as np

def preISF(x,y):
    
    T, N = x.shape
    
    layer_width = 2.0
    
    particle = []
    
    for i in range(0, N):
        layer = []
        count = 0
        count1 = 0        
        for j in range(0, T):            
            ind_layer = np.ceil( y[j,i] / layer_width )
            if (j == 0):
                ind_layer_old = ind_layer
            if (j > 0):
                if (ind_layer_old != ind_layer):
                    count = 0
                    count1 += 1
                ind_layer_old = ind_layer
            layer.append((x[j,i], y[j,i], int(ind_layer)))
            #layer[count1].xlist[count] = x[j,i]
            #layer[count1].ylist[count] = y[j,i]
            #layer[count1].layer = ind_layer
            count += 1
        particle.append(layer)
        
    N_layers = np.ceil(np.amax(np.amax(y)) / layer_width)    
    
    #for i in range(0, N):
        
    
        
    return particle