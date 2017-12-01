import tensorflow as tf
import numpy as np
import utils

def denoisenet(x, n_layers=20, filters=64):
    channels = int(x.get_shape()[-1])

    out = x
    with tf.variable_scope('DenoiseNet', reuse=tf.AUTO_REUSE):        
        for i in range(n_layers):

            res_x = tf.layers.conv2d(x,
                                     filters=channels,
                                     kernel_size=[3, 3],
                                     padding='same',
                                     name='res_conv_{}'.format(i))

            out += res_x

            x = tf.layers.conv2d(x,
                                 filters=filters - channels,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 name='conv_{}'.format(i))
            
            if i < n_layers - 2:
                x = tf.nn.relu(x)

    return out


def train(gt_scaled,
          noise_scaled,
          batch_size,
          learning_rate,
          layers,
          epochs,
          filters,
          save_path,
          batch_norm=False):

    shape = [None, None, None, gt_scaled.shape[-1]]

    # setup loss function and learning algo
    x = tf.placeholder(tf.float32, shape)
    y_true = tf.placeholder(tf.float32, shape)

    y_pred = denoisenet(x, layers, filters)
    loss = tf.nn.l2_loss(y_pred - y_true)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    losses = []
    avg_psnr_list = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for start_index in range(0, len(gt_scaled) - batch_size + 1, batch_size):
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

            print('#{} epoch, current loss: {}, average psnr: {}'.format(
                epoch, cur_loss, avg_psnr))

        if save_path:
            tf.train.Saver().save(sess, save_path)

    return losses, avg_psnr_list
