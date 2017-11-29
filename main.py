import numpy as np
import tensorflow as tf
from model import denoisenet
import utils
import os

batch_size = 64
image_width = 128
image_height = 128
channels = 3
learning_rate = 1e-4


# load data and crop
gt = utils.crop_images('data/VOCdevkit/VOC2012/JPEGImages')
noise = utils.add_poisson_noise_to_images(gt)

# scale to [-0.5, 0.5]
gt_scaled = gt / 255.0 - 0.5
noise_scaled = noise / 255.0 - 0.5

# setup loss function and learning algo
x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, channels])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, None, None, channels])

y = denoisenet(x)

loss = tf.nn.l2_loss(y - y_true)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#
n_epochs = 10
n_samples = len(gt_scaled)

saver = tf.train.Saver()

losses = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        for start_index in range(0, n_samples - batch_size + 1, batch_size):
            batch_x = noise_scaled[start_index: start_index + batch_size]
            batch_y = gt_scaled[start_index: start_index + batch_size]

            _, cur_loss = sess.run([train_step, loss], feed_dict={
                                   x: batch_x, y_true: batch_y})
            losses.append(cur_loss)


saver.save(sess, 'model/denoisenet.ckpt')
