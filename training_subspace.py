# -*- coding: utf-8 -*-
"""
NOTE THAT THIS IS ORIGINALLY BASED ON https://github.com/IBM/seizure-type-classification-tuh
SeizureNet Training subspace creator replica
Group 5 - Health machinae pro
"Note that I might change file contents anyhow in the future" -Veikka

"""
import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse

import random
import dill as pickle
import numpy as np

#M-dimensional training dataset D={(Di,yi)|0≤i≤Nd}
def create_subspace(w, step_size, f, o, file_path):
    data = pickle.load(open(file_path, 'rb'))
    D = data.D
    
    subspace_part = []
        
    #This process is repeated Ne = 3 times
    for i in range (1,3):
        #The position at highest can be = amount_of_samples_in_data - (3 * window length + window step size)
        position = random.randint(0,(len(D)-(3*(w + step_size))))
        for j in range(position, position + w):
            subspace_part.append(D[j, f]);
            #
            position = position + (np.floor(o*step_size));
            
    return subspace_part

#These are the same as in researchers' code to simplify usage
def main():
    parser = argparse.ArgumentParser(description='Create training subspace from spectogram')

    if platform.system() == 'Linux':
        parser.add_argument('-l','--save_data_dir', default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output subspace')
        parser.add_argument('-b','--base_save_data_dir', default='/fast1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output subspace')
    elif platform.system() == 'Darwin':
        parser.add_argument('-l','--save_data_dir', default='/Users/jbtang/datasets/TUH/eeg_seizure/',
                            help='path to output subspace')
        parser.add_argument('-b','--saliency_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/',
                            help='path to output subspace')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.4.0',
                        help='path to output subspace')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    save_data_dir = os.path.join(args.save_data_dir,tuh_eeg_szr_ver,'subspace')
    saliency_data_dir = os.path.join(args.saliency_data_dir,tuh_eeg_szr_ver,'fft')

    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)

    fpaths = [os.path.join(save_data_dir,f) for f in fnames]
    #--- End of researchers' code ---
    
    #Frequency channels 24Hz, 48Hz, 64Hz, 96Hz
    freqs = [24,48,64,96]
    f = random.choice(freqs)
    #Window length in seconds
    window_lengths = [1, 2, 4, 8, 16]
    w = random.choice(window_lengths)
    #Overlap percentage parameter
    overlap_percentages = [0,10,20,50]
    o = random.choice(overlap_percentages)
    #Example window step size of 10
    step_size = 10
    #Example subspace size of 2000
    space_size = 2000
    
    subspace = []
    
    for i in range (0,space_size):
        #Randomise file
        r = random.randint(0,(len(fpaths)))
        file_path = fpaths[r]
        #I'd rather not save each part into a separate variable to save memory...
        subspace.append(create_subspace(w, step_size, f, o, file_path))

    #Referenced from researchers' code, however, now only 1 file will be created
    save_data_dir = os.path.join(save_data_dir, 'subspace')
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    else:
        exit('A subspace already exists!')
    
    #Write the subspace matrix into a subspace.pkl file
    pickle.dump(subspace, open(os.path.join(saliency_data_dir,'subspace'), 'wb'))         
                    
if __name__ == '__main__':
    main()