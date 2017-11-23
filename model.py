import tensorflow as tf 


def denoisenet(x, n_layers=20):

    out = x

    with tf.variable_scope('DenoiseNet'):
        for i in range(n_layers):

            res_x = tf.layers.conv2d(x,
                                     filters=1,
                                     kernel_size=[3, 3],
                                     padding='same',
                                     name='{}_res_conv'.format(i),
                                     reuse=tf.AUTO_REUSE)
            out += res_x

            x = tf.layers.conv2d(x,
                                 filters=63, 
                                 kernel_size=[3 ,3],
                                 padding='same',
                                 name='{}_conv'.format(i),
                                 reuse=tf.AUTO_REUSE)
            if i < n_layers - 2:
                x = tf.nn.relu(x)
    
    return out        

def main():
    w = 128
    h = 128
    gt_img = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 1], name='input')
    noise_img = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 1], name='input')
    out_img = denoisenet(noise_img)

if __name__ == '__main__':
    main()
