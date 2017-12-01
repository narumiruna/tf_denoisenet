import tensorflow as tf


def denoisenet(x, n_layers=20, conv_filters=63, residual_conv_filters=3):

    out = x

    with tf.variable_scope('DenoiseNet'):
        for i in range(n_layers):

            res_x = tf.layers.conv2d(x,
                                     filters=residual_conv_filters,
                                     kernel_size=[3, 3],
                                     padding='same',
                                     name='{}_res_conv'.format(i),
                                     reuse=tf.AUTO_REUSE)

            out += res_x

            if i < n_layers - 2:
                activation = tf.nn.relu
            else:
                activation = None

            x = tf.layers.conv2d(x,
                                 filters=conv_filters,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 name='{}_conv'.format(i),
                                 reuse=tf.AUTO_REUSE, activation=activation)

    return out
