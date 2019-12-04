"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np

import vgg16_trainable as vgg16
from datagenerator import ImageDataGenerator
from datetime import datetime

import os

# Path to the textfiles for the trainings and validation set
train_file = './datatxt/train.txt'
val_file = './datatxt/val.txt'

# Learning params
learning_rate = 0.002
num_epochs = 10
batch_size = 128

# Network params
num_classes = 2
train_layers = ['fc8']

# Path for tf.summary.FileWriter and to store model checkpoints
checkpoint_path = "./finetune_vgg16/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                               tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
train_mode = tf.placeholder(tf.bool)

vgg = vgg16.Vgg16('./vgg16.npy', num_classes, train_layers)
vgg.build(x, train_mode)
# print number of variables used: 138357544 variables, i.e. ideal size = 527MB
print("The weight's numbers: ",  vgg.get_var_count())

score = vgg.fc8
# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

# Train op
with tf.name_scope("train"):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# 让tensorflow按需索取显存
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config = gpu_config) as sess:

    sess.run(tf.global_variables_initializer())

    print("{} Start training...".format(datetime.now()))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          train_mode: True})

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        val_acc = 0.
        val_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                train_mode: False})
            val_acc += acc
            val_count += 1
        val_acc /= val_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), val_acc))

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
