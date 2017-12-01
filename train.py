import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils
from model import denoisenet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--image_height', '-ih', type=int, default=128)
parser.add_argument('--image_width', '-iw', type=int, default=128)
parser.add_argument('--channels', '-ch', type=int, default=3)
parser.add_argument('--image_dir', '-dir', type=str, default='images')
parser.add_argument('--n_layers', '-nl', type=int, default=6)
parser.add_argument('--n_epochs', '-ne', type=int, default=100)
args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
image_height = args.image_height
image_width = args.image_width
channels = args.channels
image_dir = args.image_dir
n_layers = args.n_layers
n_epochs = args.n_epochs

# load data and crop
gt = utils.crop_images(image_dir)
noise = utils.add_poisson_noise_to_images(gt)

# scale to [-0.5, 0.5]
gt_scaled = gt / 255.0 - 0.5
noise_scaled = noise / 255.0 - 0.5

# setup loss function and learning algo
x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, channels])
y_true = tf.placeholder(dtype=tf.float32, shape=[None, None, None, channels])

y_pred = denoisenet(x, n_layers)

loss = tf.nn.l2_loss(y_pred - y_true)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

n_samples = len(gt_scaled)

saver = tf.train.Saver()

losses = []
avg_psnr_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        for start_index in range(0, n_samples - batch_size + 1, batch_size):
            batch_x = noise_scaled[start_index: start_index + batch_size]
            batch_y = gt_scaled[start_index: start_index + batch_size]

            _, cur_loss = sess.run([train_step, loss], feed_dict={
                                   x: batch_x, y_true: batch_y})
            losses.append(cur_loss)

        batch_pred = sess.run(y_pred, feed_dict={x: batch_x})
        batch_pred = np.clip(batch_pred + 0.5, 0.0, 1.0)
        batch_y = batch_y + 0.5

        avg_psnr = utils.avg_psnr(batch_y, batch_pred)
        avg_psnr_list.append(avg_psnr)

        print('Current loss: {}, average psnr: {}'.format(cur_loss, avg_psnr))

    saver.save(sess, 'model/denoisenet.ckpt')


fig, ax = plt.subplots(ncols=2)
ax[0].plot(losses)
ax[1].plot(avg_psnr_list)
plt.show()
