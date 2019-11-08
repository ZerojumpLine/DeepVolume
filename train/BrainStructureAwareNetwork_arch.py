import tensorflow as tf

nclass = 3

def conv3d(input_, output_dim, f_size, is_training, scope='conv3d'):
    with tf.variable_scope(scope) as scope:
        # VGG network uses two 3*3 conv layers to effectively increase receptive field
        w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
        b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.bias_add(conv1, b1)
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1', decay=0.9,
                                           zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r1 = tf.nn.relu(bn1)

        w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME')
        b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(conv2, b2)
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2', decay=0.9,
                                           zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r2 = tf.nn.relu(bn2)
        return r2


def deconv3d(input_, output_shape, f_size, is_training, scope='deconv3d'):
    with tf.variable_scope(scope) as scope:
        output_dim = output_shape[-1]
        w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn', decay=0.9,
                                          zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r = tf.nn.relu(bn)
        return r


def crop_and_concat(x1, x2):
    offsets = [0, (tf.shape(x1)[1] - tf.shape(x2)[1]) // 2, (tf.shape(x1)[2] - tf.shape(x2)[2]) // 2, (tf.shape(x1)[3] - tf.shape(x2)[3]) // 2, 0]
    size = [-1, tf.shape(x2)[1], tf.shape(x2)[2], tf.shape(x2)[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    x1_crop = tf.reshape(x1_crop, [tf.shape(x1_crop)[0], tf.shape(x1_crop)[1], tf.shape(x1_crop)[2], tf.shape(x1_crop)[3], x1.get_shape()[4]])
    return tf.concat([x1_crop, x2], 4)


def BrainStructureAwareNetwork(LR, keep_prob):
    conv_size = 3
    dropout = 0.5
    deconv_size = 2
    pool_stride_size = 2
    pool_kernel_size = 3  # Use a larger kernel
    layers = 3
    features_root = 32
    loss_type = 'cross_entropy'

    # Encoding path
    connection_outputs = []
    for layer in range(layers):
        features = 2 ** layer * features_root
        if layer == 0:
            prev = LR
        else:
            prev = pool

        conv = conv3d(prev, features, conv_size, is_training=1, scope='encoding' + str(layer))
        connection_outputs.append(conv)
        pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                padding='SAME')

    bottom = conv3d(pool, 2 ** layers * features_root, conv_size, is_training=True, scope='bottom')
    bottom = tf.nn.dropout(bottom, keep_prob)

    # Decoding path
    for layer in range(layers):
        conterpart_layer = layers - 1 - layer
        features = 2 ** conterpart_layer * features_root
        if layer == 0:
            prev = bottom
        else:
            prev = conv_decoding

        deconv_output_shape = [tf.shape(prev)[0],  tf.shape(prev)[1] * deconv_size,  tf.shape(prev)[2] * deconv_size, tf.shape(prev)[3] * deconv_size, features]
        deconv = deconv3d(prev, deconv_output_shape, deconv_size, is_training=1, scope='decoding' + str(conterpart_layer))
        cc = crop_and_concat(connection_outputs[conterpart_layer], deconv)
        conv_decoding = conv3d(cc, features, conv_size, is_training=True, scope='decoding' + str(conterpart_layer))

    with tf.variable_scope('probs') as scope:
        w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
        b = tf.get_variable('b', 1, initializer=tf.constant_initializer(0.0))
        probs = tf.nn.bias_add(logits, b)

    with tf.variable_scope('logits') as scope:
        w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], nclass], initializer=tf.truncated_normal_initializer(stddev=0.1))
        logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
        b = tf.get_variable('b', nclass, initializer=tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(logits, b)

    return probs, logits