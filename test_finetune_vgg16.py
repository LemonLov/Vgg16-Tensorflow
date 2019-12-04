import numpy as np
import tensorflow as tf

import vgg16_trainable as vgg16
from datagenerator import ImageDataGenerator
from datetime import datetime
#from tensorflow.data import Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
test_file = './datatxt/test.txt'

# Learning params
num_epochs = 10
batch_size = 128

# Network params
num_classes = 2
train_layers = ['fc8']

# Path for tf.summary.FileWriter and to store model checkpoints
checkpoint_path = "./finetune_vgg16/"

"""
Main Part of the finetuning Script.
"""
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(test_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(test_data.data.output_types,
                                               test_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the test iterators
test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
train_mode = tf.placeholder(tf.bool)

vgg = vgg16.Vgg16('./vgg16.npy', num_classes, train_layers)
vgg.build(x, train_mode)
# print number of variables used: 138357544 variables, i.e. ideal size = 527MB
print("The weight's numbers: ",  vgg.get_var_count())

score = vgg.fc8

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

# 让tensorflow按需索取显存
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
# Start Tensorflow session
with tf.Session(config = gpu_config) as sess:

    # To continue training from one of your checkpoints
    saver.restore(sess, checkpoint_path + "model_epoch" + str(num_epochs) + ".ckpt")

    # Validate the model on the entire validation set
    print("{} Start test...".format(datetime.now()))
    sess.run(test_init_op)
    test_acc = 0.
    test_count = 0
    for _ in range(test_batches_per_epoch+1):
        img_batch, label_batch = sess.run(next_batch)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                            y: label_batch,
                                            train_mode: False})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print("{} Test Accuracy = {:.4f}".format(datetime.now(), test_acc))
