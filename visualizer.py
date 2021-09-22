from tensorflow_utils import plot_to_image
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from cnn_params import *
import pickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt



data_dir = "/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
cross_val_file = "../seizure-type-classification-tuh/data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"

def generate_eeg_visualizations_for_fold(data_dir, fold_data, dataType, labelEncoder):
    classesUsed = { i : [] for i in SZR_CLASSES}

    data = fold_data.get(dataType)
    X = np.empty((len(data), EEG_WINDOWS, EEG_COLUMNS), dtype=np.float64)
    y = list()
    for i, fname in enumerate(data):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        label = seizure.seizure_type
        y.append(label)
        Xtemp = seizure.data.copy()
        length = len(seizure.data)
        if(EEG_WINDOWS > length):
          X[i] = np.pad(seizure.data, ((0, EEG_WINDOWS -length), (0,0)))
        else:
          X[i] = np.resize(seizure.data, (EEG_WINDOWS, len(seizure.data[0])))

        if(len(classesUsed[label]) < 10):
            figure = plot_eeg_data(label, Xtemp, X[i])
            eeg_img = plot_to_image(figure, False)
            classesUsed[label].append(eeg_img)
        
    
    file_writer_cm = tf.summary.create_file_writer("logs/checkpoints/train_data") # Writer for the confusion matrix
    with file_writer_cm.as_default():
        for (label, imgs) in classesUsed.items():
            tf.summary.image(f"Training Data - {label}", imgs, max_outputs=len(imgs), step=1)

    

def plot_eeg_data(label, eeg_data, eeg_data_processed):
  figure, (ax1, ax2) = plt.subplots(2,1)
  figure.suptitle(label)
  ax1.pcolormesh(eeg_data.transpose(), cmap="Greens")
  ax1.set_title("Unprocessed")
  ax2.pcolormesh(eeg_data_processed.transpose(), cmap="Greens")
  ax2.set_title("Processed")
  figure.tight_layout()
  return figure


def visualize_examples_of_data(eeg_data, labels):
  _, indices = np.unique(labels, return_index = True)
  for currPloti, i in enumerate(indices):
    ax = plt.subplot(4,2, currPloti + 1)
    ax.pcolormesh(eeg_data[i].transpose(), cmap="Greens")
    ax.set_title(labels[i])
  plt.tight_layout()
  
  plt.show()

if __name__ =="__main__":
  le = LabelBinarizer()
  le.fit(SZR_CLASSES)
  seizure_folds = pickle.load(open(cross_val_file, "rb"))
  seizure_folds = list(seizure_folds.values())[0]
  generate_eeg_visualizations_for_fold(data_dir, seizure_folds, "train", le)


