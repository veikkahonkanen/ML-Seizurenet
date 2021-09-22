# -*- coding: utf-8 -*-
"""
NOTE THAT THIS IS ORIGINALLY BASED ON https://github.com/IBM/seizure-type-classification-tuh
SeizureNet Preprocess Library extension
Group 5 - Health machinae pro
"Note that I might change file contents anyhow in the future" -Veikka

"""
import scipy
import numpy as np
from scipy.fftpack import ifft

#Gaussian kernel
GK = np.array([[1/16, 1/8, 1/16],
             [1/8, 1/4, 1/8],
             [1/16, 1/8, 1/16]])
#Local 3x3 averaging filter
af = np.array([[1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9]])

#---Saliency map S1---
class Substract_average_plus_P_2():
    """
    Job: Calculate FT-H*FT; substract average log amplitude from feature map FT
    """
    def get_name(self):
        return "substract-average-amplitude"

    def apply(self, data):
        #Data arrays dimensions and padding
        FT = data
        (dim_x, dim_y) = np.shape(FT)
        FT_padded = scipy.pad(array=FT, pad_width=[1, 1], mode='constant', constant_values=0)
        (dim_x_padded, dim_y_padded) = np.shape(FT_padded)

        #Ft-H*FT
        H_padded = np.zeros(FT_padded.shape)
        for i in range (1, dim_x_padded-1):
            for j in range(1, dim_y_padded-1):
                entry = FT_padded[i-1:i+2, j-1:j+2]
                #print(np.shape(entry))
                valor = entry*af
                #print(valor)
                H_padded[i-1:i+2, j-1:j+2] = valor
                #print(H_padded)
        H = np.zeros(FT.shape)
        for i in range (1, dim_x_padded-1):
            for j in range(1, dim_y_padded-1):
                H[i-1, j-1] = H_padded[i, j]
        #print(H)
        exp_FT_H_FT_P_2 = (np.exp((FT-(H*FT)+np.angle(FT))))**2
        #print(exp_FT_H_FT_P_2)
        return exp_FT_H_FT_P_2
    
class IFFT():
    """
    Job: Inverse Fourier transform F-1
    """
    def get_name(self):
        return "inverse-fourier"

    def apply(self, data):
        axis = data.ndim - 1
        return ifft(data, axis=axis)

class Smooth_Gaussian():
    """
    Job: Smooth the saliency feature map using a Gaussian kernel.
    """
    def get_name(self):
        return "smooth-gaussian-kernel"

    def apply(self, data):
        F1_padded = scipy.pad(array=data, pad_width=[1, 1], mode='constant', constant_values=0)
        (dim_x_padded, dim_y_padded) = np.shape(F1_padded)
        G_padded = np.zeros(F1_padded.shape)
        for i in range (1, dim_x_padded-1):
            for j in range(1, dim_y_padded-1):
                entry = F1_padded[i-1:i+2, j-1:j+2]
                valor = entry*af
                G_padded[i-1:i+2, j-1:j+2] = valor
                #print(G_padded)
        G = np.zeros(data.shape)
        for i in range (1, dim_x_padded-1):
            for j in range(1, dim_y_padded-1):
                G[i-1, j-1] = G_padded[i, j]
        #print(G)
        return G

#---Saliency map S2---
class Center_surround_diff():
    """
    Job: Capture saliency of each data point in the Fourier Transform
    feature map FT with respect to its surrounding data points by computing
    center-surround differences.
    """
    def get_name(self):
        return "compute-center-surround-differences"

    def apply(self, data):
        FT = data
        F2 = np.zeros(FT.shape)
        F2_padded = scipy.pad(array=F2, pad_width=[1, 1], mode='constant', constant_values=0)
        (dim_x_padded, dim_y_padded) = np.shape(F2_padded)
        #Note that here we suppose that radius ρ equals 1
        #Note that we're getting minimum value from each FTk,ρ that is padded, but the numbers are all negative so this doesn't matter
        for i in range (1, dim_x_padded-1):
            for j in range(1, dim_y_padded-1):
                entry = FT[i-1:i+2, j-1:j+2]
                flattened = np.matrix.flatten(entry)
                valor = min(flattened)
                F2[i-1, j-1] = valor
        #print(F2)

        S2 = np.zeros(FT.shape)
        (dim_x, dim_y) = np.shape(FT)
        for i in range (0, dim_x):
            for j in range(0, dim_y):
                min_val = np.subtract(FT[i, j],F2[i, j])
                #print(min_val)
                S2[i, j] = (min_val)
        #print(S2)
        return S2

#--- Saliency-encoded spectogram D
class Normalise():
    """
    Job: Normalise a saliency map
    """
    def get_name(self):
        return "saliency-map-normaliser"

    def apply(self, data):
        map_max, map_min = data.max(), data.min()
        map_norm = (data - map_min)/(map_max - map_min)
        return map_norm

class RGB_0_255():
    """
    Job: Normalise map to 0-255 range (RGB colors)
    """
    def get_name(self):
        return "RGB color converter"

    def apply(self, data):
        data = (data*255).astype(int)
        return data

class Concatenation():
    """
    Job: Concatenate normalised maps
    """
    def get_name(self):
        return "concatenate-maps"

    def apply(self, FT_data, S1_data, S2_data):
        D = (FT_data*S1_data*S2_data)
        return D