import io
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
def plot_to_image(figure, add_dim = True):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    if(add_dim):
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
    
    return image

# https://github.com/tensorflow/tensorflow/issues/14451
# if equal to 0 then oversample_classes() always returns 1
def oversample_classes(label, label_probs, label_target_probs, oversampling_coef = 0.2):
    """
    Returns the number of copies of given example
    """
    print(label)
    label = tf.math.argmax(label, axis= -1, output_type=tf.dtypes.int32)
    label_prob = label_probs.lookup(label)
    label_target_prob = label_target_probs.lookup(label)
   
    prob_ratio = tf.cast(label_target_prob/label_prob, dtype=tf.float32)
    # soften ratio is oversampling_coef==0 we recover original distribution
    prob_ratio = prob_ratio ** oversampling_coef 
    # for classes with probability higher than class_target_prob we
    # want to return 1
    prob_ratio = tf.math.maximum(prob_ratio, 1) 
    # for low probability classes this number will be very large
    repeat_count = tf.math.floor(prob_ratio)
    # prob_ratio can be e.g 1.9 which means that there is still 90%
    # of change that we should return 2 instead of 1
    repeat_residual = prob_ratio - repeat_count # a number between 0-1
    residual_acceptance = tf.math.less_equal(
                        tf.random.uniform([], dtype=tf.float32), repeat_residual
    )

    residual_acceptance = tf.cast(residual_acceptance, tf.int64)
    repeat_count = tf.cast(repeat_count, dtype=tf.int64)
    
    return tf.reshape(repeat_count + residual_acceptance, []) #convert tensor to scalar tensor


# undersampling coef if equal to 0 then oversampling_filter() always returns True
def undersampling_filter(value, label, label_probs, label_target_probs,undersampling_coef = 0.8):
    """
    Computes if given example is rejected or not.
    """
    print(label)

    newlabel = tf.math.argmax(label, axis= -1, output_type=tf.dtypes.int32)
    label_prob = label_probs.lookup(newlabel)
    label_target_prob = label_target_probs.lookup(newlabel)
    prob_ratio = tf.cast(label_target_prob/label_prob, dtype=tf.float32)
    prob_ratio = prob_ratio ** undersampling_coef
    prob_ratio = tf.math.minimum(prob_ratio, 1.0)
    
    #return tf.cond(tf.random.uniform([], dtype=tf.float32) <= prob_ratio, lambda: tf.data.Dataset.from_tensors((value, label)), lambda: tf.data.Dataset())
    return tf.reshape(tf.math.less_equal(tf.random.uniform([], dtype=tf.float32), prob_ratio), [])

    

