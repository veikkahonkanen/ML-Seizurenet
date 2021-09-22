import pickle
import os 
import numpy as np
import math
import tensorflow as tf

class MovingAverage():
    def __init__(self):
        self.__count = 0
        self.__mean = 0

    def update(self, newVal):
        self.__count += 1
        diferential = (newVal - self.__mean) / self.__count
        self.__mean = diferential + self.__mean
    
    def get_mean(self):
        return self.__mean



class DatasetCreator():
    def __init__(self, batch_size, rows, dataset):
        self.batch_size = batch_size
        self.rows = rows
        self.dataset = dataset

    def _load_eeg(self, path):
        pass

"""
Method 0 consists in adding up all the windows, thus returning a 2D array
Method 1 padds according to the maximum row size (adds 0 to all the others).
It's bad because for instance the maxRow can be very big, making the final array too big to fit into memory:
    maxRow = 4894
    cols = 660
    instances = 2031
    4894*660 * 2031 * 32 / 8 / 1024 / 1024 = 25025 GB
Method 2 resizes --> cuts off the number of rows to match to the minimum. The minimum row can be as little as one, which makes 
this method completely unusable
Method 3 --> does cummulative moving average of size
"""
def get_fold_data(data_dir, fold_data, dataType, labelEncoder, method = 0):
    X = list()#np.empty(len(fold_data))
    y = list()#np.empty(len(fold_data))
    maxRow = -1
    minRow = math.inf
    average = MovingAverage()
    for i, fname in enumerate(fold_data.get(dataType)):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        
        if(method == 0):
            #Sum rows method
            X.append(np.sum(seizure.data, axis=0))
        elif(method == 1):
            # Pad with rows of zeros method
            if(seizure.data.shape[0] > maxRow):
                maxRow = seizure.data.shape[0]
            X.append(seizure.data)
        elif(method == 2):
            if(seizure.data.shape[0] < minRow):
                minRow = seizure.data.shape[0]
            X.append(seizure.data)
        elif(method == 3):
            average.update(seizure.data.shape[0])
            X.append(seizure.data)
        
        y.append(seizure.seizure_type)
    if(method == 1):
        for i in range(len(X)):
            X[i] = np.pad(X[i], ((0, maxRow -len(X[i])), (0,0)))
    elif(method == 2):
        minRow=32
        for i in range(len(X)):
            X[i] = np.resize(X[i], (minRow, len(X[i][0])))
    elif(method == 3):
        
        avg = int(average.get_mean())
        print("Avg", avg)
        for i in range(len(X)):
            if(avg > len(X[i])):
                X[i] = np.pad(X[i], ((0, avg -len(X[i])), (0,0)))
            else:
                X[i] = np.resize(X[i], (avg, len(X[i][0])))
            
    y = labelEncoder.transform(y)
    return X, y


def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory