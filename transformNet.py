import tensorflow as tf

def ImageNet(input_img):
    conv1 = conv_layer(input_img, 32, 9, 1)
    conv2 = conv_layer(conv1, 64, 3, 2)
    conv3 = conv_layer(conv2, 128, 3, 2)
    resid1 = residual_block(conv3, 3)
    resid2 = residual_block(resid1, 3)
    resid3 = residual_block(resid2, 3)
    resid4 = residual_block(resid3, 3)
    resid5 = residual_block(resid4, 3)
    conv_t1 = conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = conv_layer(conv_t2, 3, 9, 1, relu=False)
    out_img = tf.nn.tanh(conv_t3,name='out_img') * 150 + 255./2
    return out_img


def conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = conv_init_vars(net, num_filters, filter_size)  # weight init
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = instance_norm(net)  # batch_norm
    if relu:
        net = tf.nn.relu(net)

    return net


def conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, _ = [i.value for i in net.get_shape()]
    # get new rows and cols
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)  # new_shape to tf.stack
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = instance_norm(net)
    return tf.nn.relu(net)


def residual_block(net, filter_size=3):
    tmp = conv_layer(net, 128, filter_size, 1)
    return net + conv_layer(tmp, 128, filter_size, 1, relu=False)


def instance_norm(net, train=True):
    _, _, _, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift


def conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, _, _, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32)
    return weights_init