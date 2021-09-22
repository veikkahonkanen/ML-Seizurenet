import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn import metrics,utils
import itertools
import seaborn as sns
import math
from tensorflow_utils import *
from tensorboard.plugins.hparams import api as hp
from cnn_params import *
import pandas as pd
#tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()
#tf.debugging.set_log_device_placement(False)

# Cross validation inpired in https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
else:
        raise SystemError("NO GPUS")


#config.set_visible_devices([], 'GPU')
print("GPUS {}", gpus)


#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


METRICS = [
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tfa.metrics.F1Score(name='f1_score_macro', average='macro', num_classes=7),
      tfa.metrics.F1Score(name='f1_score_weighted', average='weighted', num_classes=7)
]




def get_fold_data(data_dir, fold_data, dataType, labelEncoder):
    data = fold_data.get(dataType)
    X = np.empty((len(data), EEG_WINDOWS, EEG_COLUMNS), dtype=np.float64)
    y = list()
    for i, fname in enumerate(data):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        y.append(seizure.seizure_type)
        length = len(seizure.data)
        if(EEG_WINDOWS > length):
          X[i] = np.pad(seizure.data, ((0, EEG_WINDOWS -length), (0,0)))
        else:
          X[i] = np.resize(seizure.data, (EEG_WINDOWS, len(seizure.data[0])))
    
        
    if labelEncoder != None:
      y = labelEncoder.transform(y)

    return X, y


@tf.function
def resize_eeg(data, label):
  return tf.stack([data,data,data], axis=-1), label#tf.py_function(lambda: tf.convert_to_tensor(np.stack([data.numpy()]*3, axis=-1), dtype=data.dtype), label, Tout=[tf.Tensor, type(label)] )

def get_dataset(X, y):
  return tf.data.Dataset.from_tensor_slices((X, y)).map(resize_eeg, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_fold_datasets(data_dir, fold_data, le, class_probs=None, oversample=False, undersample=False):
  X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
  X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)

  print("DIMENSIONS: ",len(X_train), ",", len(X_train[0]))
  train_dataset = get_dataset(X_train, y_train)
  class_target_probs = None
  if class_probs:
    class_target_probs = {label_name: 0.5 for label_name in class_probs}
    class_target_probs = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(tf.constant(list(class_target_probs.keys())), tf.constant(list(class_target_probs.values()), dtype=tf.dtypes.float64)),
      default_value=0
    )
    class_probs = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(tf.constant(list(class_probs.keys())), tf.constant(list(class_probs.values()), dtype=tf.dtypes.float64)),
      default_value=0
    )

  if oversample:
    #train_dataset = train_dataset.map(, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(oversample_classes(y,class_probs, class_target_probs)))
  if undersample:
    #train_dataset = train_dataset.map()
    train_dataset =train_dataset.filter(lambda x,y: undersampling_filter(x,y, class_probs, class_target_probs))
  
  if oversample or undersample:
    train_dataset = train_dataset.shuffle(300).repeat()

  train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(PREFETCH)
  val_dataset = get_dataset(X_val, y_val).batch(BATCH_SIZE).prefetch(PREFETCH)

  return train_dataset,  val_dataset

def get_test_dataset(data_dir, fold_data, le):
  X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
  X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)

  train_dataset = get_dataset(X_train, y_train)
  val_dataset = get_dataset(X_val, y_val)
  return train_dataset.concatenate(val_dataset).batch(BATCH_SIZE).prefetch(PREFETCH)


data_dir ="/media/david/Extreme SSD1/Machine Learning/raw_data/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_24" #"/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
cross_val_file = "../seizure-type-classification-tuh/data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"


def calculate_weights_and_probs(le):
  sz = pickle.load(open(cross_val_file, "rb"))

  y_labels = list()
  for data in sz.values():
    _, y_train = get_fold_data(data_dir, data, "train", None)
    _, y_val = get_fold_data(data_dir, data, "val", None)
    y_labels += y_val + y_train
  
  unique, counts = np.unique(y_labels, return_counts=True)
  

  total = sum(counts)
  probs = dict(zip(np.argmax(le.transform(unique), axis=1), counts))
  probs = {key : probs[key]/total for key in probs}
  class_weights = utils.class_weight.compute_class_weight('balanced',
                                                 classes=le.classes_,
                                                 y=y_labels)

  return dict(enumerate(class_weights)), probs
  

def plot_dataset_dist(dataset, probs, le):
  data = {i : 0 for i in le.classes_}
  for _, y in dataset.unbatch().take(5000).as_numpy_iterator():
    y = np.argmax(y, axis=-1)
    data[le.classes_[y]] += 1

  total = sum(data.values())

  aug_probs = {key : data[key]/total for key in data}

  probs_dict = {
    "augmented_prob": aug_probs,
    "real_prob": {le.classes_[key]: probs[key] for key in probs}
  }
  print(probs_dict["real_prob"])
  df = pd.DataFrame(probs_dict).plot(kind="bar")
 
  plt.show()
 
def calculate_dataset_min_max():
  def calcMax(currMax, data):
    valMax = np.amax(data)
    if(valMax >  currMax):
      return valMax
    return currMax
  def calcMin(currMin, data):
    valMin = np.amin(data)
    if(valMin <  currMin):
      return valMin
    return currMin

  sz = pickle.load(open(cross_val_file, "rb"))
  max = -math.inf
  min = math.inf
  for data in sz.values():
    X_train, y_train = get_fold_data(data_dir, data, "train", None)
    X_val, y_val = get_fold_data(data_dir, data, "val", None)
      
    max = calcMax(max, X_train)
    max = calcMax(max, X_val)
    min = calcMin(min, X_train)
    min = calcMin(min, X_val)
  
  return min, max


def preprocess_input(x, min=DATASET_MIN,max=DATASET_MAX):
  """
  Preprocesses a tensor encoding a batch of eeg data between -1 and 1
  Based on https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/applications/imagenet_utils.py#L103-L119
  """
  #min, max = calculate_dataset_min_max()
  x += DATASET_MIN
  x /= DATASET_MAX/2.
  x -= 1.

  return x

from tensorflow.keras.mixed_precision import experimental as mixed_precision
def create_model(metrics=METRICS, output_bias=None):
  
  #TODO: look into mixed precision to increase time performance https://www.tensorflow.org/g[]uide/mixed_precision

  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)

  data_augmentation = tf.keras.Sequential([
    
  ])



  # Create the base model from the pre-trained model MobileNet V2
  base_model = tf.keras.applications.ResNet50V2(input_shape=EEG_SHAPE + (3,),
                                                include_top=False,
                                                weights='imagenet')


  base_model.trainable = False


  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


  prediction_layer = tf.keras.layers.Dense(len(SZR_CLASSES), activation="softmax", bias_initializer=output_bias)


  inputs = tf.keras.Input(shape=EEG_SHAPE + (3,))
  #x = data_augmentation(inputs)
  #x = preprocess_input(x) 
  #x = base_model(x, training=False)
  x =base_model(inputs, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)
  model.summary()

  base_learning_rate = 0.1#0.0001
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=metrics)
  return model

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, 
              annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title("Confusion matrix")
    
    return figure






def train_model(hparams, logs_dir, le, weights, class_probs = None):
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs_dir,histogram_freq = 1,profile_batch = '490,510')  

  train_dataset, val_dataset = get_fold_datasets(data_dir, fold_data, le, class_probs, True, True)

  #plot_dataset_dist(train_dataset, class_probs, le)
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Create the model
  model = create_model()

  # Train the model
  history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=np.ceil(2200/BATCH_SIZE), validation_data=val_dataset, class_weight=weights, callbacks=[tboard_callback])
  
  # Allow GC to collect the datasets. If not, they will be available in the next iteration fo the for loop
  # and won't end up fitting in the RAM
  #train_dataset = None
  #val_dataset = None
  return model
  
def save_confusion_matrix(test_labels, test_pred, classes, logs_dir):
  file_writer_cm = tf.summary.create_file_writer(logs_dir + '/cm') # Writer for the confusion matrix
  cm = tf.math.confusion_matrix(test_labels, test_pred)
  figure = plot_confusion_matrix(cm, class_names=classes)
  cn_image =plot_to_image(figure)
  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cn_image, step=1)
  


if __name__ == "__main__":
  # Create a TensorBoard callback
  early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)

  
  le = LabelBinarizer()
  le.fit(SZR_CLASSES)
  weights, probs = calculate_weights_and_probs(le)
  print("Classes:", le.classes_)
  print("Class weights: ", weights)
  print("Probs: ", probs)


  seizure_folds = pickle.load(open(cross_val_file, "rb"))
  
  seizure_folds = list(seizure_folds.values())

  k_validation_folds = seizure_folds[:-1] # TODO: Choose which dataset will be the test randomly
  test_fold = seizure_folds[-1]

  test_dataset = get_test_dataset(data_dir, test_fold, le)    
  test_labels = np.concatenate([y for x, y in test_dataset], axis=0).argmax(axis=1)
  currentTime = datetime.now().strftime("%Y%m%d-%H%M%S")
  # Define per-fold score containers
  acc_per_fold = []
  loss_per_fold = []
  for fold_no, fold_data in enumerate(k_validation_folds):
    logs_dir = f"logs/fit/{currentTime}/fold_{fold_no}"  
  

    model = train_model(None, logs_dir, le, weights, probs)
    # evaluate the model
    print("[INFO] evaluating network...")
    scores = model.evaluate(test_dataset, verbose=0)


    with tf.summary.create_file_writer(logs_dir + '/run').as_default():
      tf.summary.scalar(model.metrics_names[5], scores[5], step=1)
      tf.summary.scalar(model.metrics_names[6], scores[6], step=1)

      
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    
    print("[INFO] Predicting network and creating confusion matrix...")

    test_pred = np.argmax(model.predict(test_dataset), axis=1)
    save_confusion_matrix(test_labels, test_pred, le.classes_, logs_dir)
    
    
    
    
  # == Provide average scores ==
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')
