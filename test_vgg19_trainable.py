"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

# 让tensorflow按需索取显存
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config = gpu_config) as sess:
    
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print("The weight's numbers: ",  vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # simple 1-step training
    #loss = tf.reduce_sum((vgg.prob - true_out) ** 2)
    score = vgg.fc8
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = true_out))
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    sess.run(train_op, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './trainable_vgg19.npy')
