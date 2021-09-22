# -*- coding: utf-8 -*-
"""
NOTE THAT THIS IS MADE BASED ON generate_fft_images.py at https://github.com/IBM/seizure-type-classification-tuh
SeizureNet saliency map S1 creator replica
Group 5 - Health machinae pro
"Note that I might change file contents anyhow in the future" -Veikka
"""

import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse

import dill as pickle
import collections
from lib_extension import Substract_average_plus_P_2, IFFT, Smooth_Gaussian, Center_surround_diff, Normalise, RGB_0_255, Concatenation
from utils.pipeline import Pipeline
import numpy as np
from joblib import Parallel, delayed
import warnings

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

#Saliency map S1
def create_s1(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):

    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    pipeline = Pipeline([Substract_average_plus_P_2(), IFFT(), Smooth_Gaussian()])
    time_series_data = type_data.data
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    s1_data = []

    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:, start:stop]
        window = pipeline.apply(signal_window)
        s1_data.append(window)
        start, stop = start + step, stop + step

    s1_data = np.array(s1_data)
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=type_data.data, s1=s1_data)

    return named_data,os.path.basename(file_path)

#Saliency map S2
def create_s2(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):

    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    pipeline = Pipeline([Center_surround_diff()])
    time_series_data = type_data.data
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    s2_data = []

    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:, start:stop]
        window = pipeline.apply(signal_window)
        s2_data.append(window)
        start, stop = start + step, stop + step

    s2_data = np.array(s2_data)
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=type_data.data, s1=type_data.s1, s2=s2_data)

    return named_data,os.path.basename(file_path)

#RGB-encoded spectogram D
def create_d(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):

    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    #Three of these pipelines are needed, as concatenation takes a different kind of parameter (three maps)
    pipeline1 = Pipeline([Normalise()])
    pipeline2 = Pipeline([Concatenation()])
    pipeline3 = Pipeline([RGB_0_255()])
    
    #The three feature maps
    data_ft = type_data.data
    data_s1 = type_data.s1
    data_s2 = type_data.s2
    
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    d_data = []

    while stop < data_ft.shape[1]:
        #Window definitions, the maps are of same size & shape so 1 looper can be used for all
        window_ft = data_ft[:, start:stop]
        window_s1 = data_s1[:, start:stop]
        window_s2 = data_s2[:, start:stop]
        #Normalise each window value
        window_ft_norm = pipeline1.apply(window_ft)
        window_s1_norm = pipeline1.apply(window_s1)
        window_s2_norm = pipeline1.apply(window_s2)
        #Concatenate normalised values
        d_norm = pipeline2.apply(window_ft_norm, window_s1_norm, window_s2_norm)
        #RGB 0-255 conversion
        d_rgb = pipeline3.apply(d_norm)
        
        d_data.append(d_rgb)
        start, stop = start + step, stop + step

    d_data = np.array(d_data)
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=d_data)

    return named_data,os.path.basename(file_path)

#These here down below are the same as in researchers' code to simplify usage
def main():
    parser = argparse.ArgumentParser(description='Generate Saliency maps and spectogram from preprocessed data')

    if platform.system() == 'Linux':
        parser.add_argument('-l','--save_data_dir', default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output updated data')
        parser.add_argument('-b','--base_save_data_dir', default='/fast1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output updated data')
    elif platform.system() == 'Darwin':
        parser.add_argument('-l','--save_data_dir', default='/Users/jbtang/datasets/TUH/eeg_seizure/',
                            help='path to output updated data')
        parser.add_argument('-b','--preprocess_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/',
                            help='path to output updated data')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.4.0',
                        help='path to output updated data')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    save_data_dir = os.path.join(args.save_data_dir,tuh_eeg_szr_ver,'spectogram')
    preprocess_data_dir = os.path.join(args.preprocess_data_dir,tuh_eeg_szr_ver,'fft')

    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)

    fpaths = [os.path.join(save_data_dir,f) for f in fnames]

    sampling_frequency = 250  # Hz
    fft_min_freq = 1  # Hz

    window_lengths = [1, 2, 4, 8, 16]#[0.25, 0.5, 1]#[1, 2, 4, 8, 16]
    fft_max_freqs = [12, 24, 48, 64, 96]#[12, 24]

    for window_length in window_lengths:
        window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))
        #window_steps = list(np.arange(window_length / 8, window_length / 2 + window_length / 8, window_length / 8))
        print(window_steps)
        for window_step in window_steps:
            for fft_max_freq_actual in fft_max_freqs:
                fft_max_freq = fft_max_freq_actual * window_length
                fft_max_freq = int(np.floor(fft_max_freq))
                print('window length: ', window_length, 'window step: ', window_step, 'fft_max_freq', fft_max_freq)
                save_data_dir = os.path.join(preprocess_data_dir, 'fft_seizures_' + 'wl' + str(window_length) + '_ws_' + str(window_step) \
                                + '_sf_' + str(sampling_frequency) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
                                str(fft_max_freq_actual))
                if not os.path.exists(save_data_dir):
                    os.makedirs(save_data_dir)
                else:
                    exit('Saliency map S1 data already exists!')

                '''
                converted_data = Parallel(n_jobs=15)(delayed(convert_to_fft)(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path=item) for item in fpaths)
                count = 0
                for item in converted_data:
                    if item.data.ndim == 3:
                        pickle.dump(item, open( os.path.join(save_data_dir, file_name_base), 'wb'))
                        count += 1
                '''
                
                #Create each map in order, then create spectogram D
                for file_path in sorted(fpaths):
                    converted_data,file_name_base = create_s1(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency,file_path)
                    if converted_data.data.ndim == 3:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir, file_name_base), 'wb'))
                    converted_data,file_name_base = create_s2(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency,file_path)
                    if converted_data.data.ndim == 3:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir, file_name_base), 'wb'))
                    converted_data,file_name_base = create_d(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency,file_path)
                    if converted_data.data.ndim == 3:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir, file_name_base), 'wb'))
                    
if __name__ == '__main__':
    main()