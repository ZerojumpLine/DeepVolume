import tensorflow as tf
import BasicConvLSTMCell as cl

batch_size = 1

def conv_layer(inputs, kernel_size, stride, num_features, is_training, idx, linear=False):
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
        weights = tf.get_variable('weights', [kernel_size, kernel_size, inputs.get_shape()[-1], num_features],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [num_features], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        bn = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='bn', decay=0.9,
                                          zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        rn = tf.nn.relu(bn)
        if linear:
            return conv
        return rn


def transpose_conv_layer(inputs, kernel_size, stride, num_features, is_training, idx, linear=False):
    with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
        weights = tf.get_variable('weights', [kernel_size, kernel_size, num_features, inputs.get_shape()[-1]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        shape = inputs.get_shape().as_list()
        output_shape = [shape[0], shape[1] * stride, shape[2] * stride, num_features]
        deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn', decay=0.9,
                                          zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r = tf.nn.relu(bn)
        if linear:
            return deconv
        return r


def SpatialConnectionAwareNetwork(inputs, hidden, lstm=True):
    conv1 = conv_layer(inputs, 5, 1, 32, 1, 'encode_1')
    conv2 = conv_layer(conv1, 3, 2, 64, 1, 'encode_2')
    conv3 = conv_layer(conv2, 3, 1, 64, 1, 'encode_3')
    conv4 = conv_layer(conv3, 3, 2, 128, 1, 'encode_4')
    y_0 = conv4

    if lstm:
        # conv lstm cell
        with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
            cell = cl.BasicConvLSTMCell([90, 108], [3, 3], 128)
            if hidden is None:
                hidden = cell.zero_state(batch_size, tf.float32)
            y_1, hidden = cell(y_0, hidden)
    else:
        y_1 = conv_layer(y_0, 3, 1, 128, 1, 'encode_5')

    conv6 = conv_layer(y_1, 3, 1, 128, 1, 'decode_6')
    conv7 = transpose_conv_layer(conv6, 4, 2, 64, 1, 'decode_7') + conv3
    conv8 = conv_layer(conv7, 3, 1, 64, 1, 'decode_8')
    conv9 = transpose_conv_layer(conv8, 4, 2, 32, 1, 'decode_9') + conv1
    conv10 = conv_layer(conv9, 3, 1, 64, 1, 'decode_10')
    # x_1
    conv11 = conv_layer(conv10, 5, 1, 1, 1, 'decode_11', True) + inputs[:, :, :, 0:1]  # set activation to linear

    x_1 = conv11

    return x_1, hidden