import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


def generator_api(inputs, is_train=True, reuse=False):
    image_size = FLAGS.output_size
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), \
                      int(image_size / 8), int(image_size / 16)

    gf_dim = 64  # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim  # n_color 3
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim * 8 * 8, W_init=w_init,
                            act=tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, 8, 8, gf_dim], name='g/h0/reshape')

        net_h1 = Conv2d(net_h0, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h1/conv2d_a')
        net_h1 = Conv2d(net_h1, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h1/conv2d_b')

        if image_size == 128:
            net_h2 = UpSampling2dLayer(net_h1, (s8, s8), is_scale=False, method=1)
            net_h2 = Conv2d(net_h2, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                            padding='SAME', W_init=w_init, name='g/h2/conv2d_a')
            net_h2 = Conv2d(net_h2, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                            padding='SAME', W_init=w_init, name='g/h2/conv2d_b')
        else:
            net_h2 = net_h1

        net_h3 = UpSampling2dLayer(net_h2, (s4, s4), is_scale=False, method=1)
        net_h3 = Conv2d(net_h3, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h3/conv2d_a')
        net_h3 = Conv2d(net_h3, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h3/conv2d_b')

        net_h4 = UpSampling2dLayer(net_h3, (s2, s2), is_scale=False, method=1)
        net_h4 = Conv2d(net_h4, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h4/conv2d_a')
        net_h4 = Conv2d(net_h4, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h4/conv2d_b')

        net_h5 = UpSampling2dLayer(net_h4, (image_size, image_size), is_scale=False, method=1)
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h5/conv2d_a')
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='g/h5/conv2d_b')

        net_h6 = Conv2d(net_h5, c_dim, (3, 3), (1, 1), act=tf.identity,
                        padding='SAME', W_init=w_init, name='g/h6/conv2d')

    return net_h6


def encoder_api(inputs, is_train=True, reuse=False):
    image_size = FLAGS.output_size
    df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
    z_dim = FLAGS.z_dim
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='enc/in')

        net_h0 = Conv2d(net_in, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h0/conv2d_a')
        net_h0 = Conv2d(net_h0, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h0/conv2d_b')
        net_h0 = Conv2d(net_h0, 2 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h0/conv2d_c')

        net_h1 = PoolLayer(net_h0)
        net_h1 = Conv2d(net_h1, 2 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h1/conv2d_a')
        net_h1 = Conv2d(net_h1, 3 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h1/conv2d_b')

        net_h2 = PoolLayer(net_h1)
        net_h2 = Conv2d(net_h2, 3 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h2/conv2d_a')
        net_h2 = Conv2d(net_h2, 4 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h2/conv2d_b')

        net_h3 = PoolLayer(net_h2)
        net_h3 = Conv2d(net_h3, 4 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h3/conv2d_a')
        net_h3 = Conv2d(net_h3, 5 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='enc/h3/conv2d_b')

        if image_size == 128:
            net_h4 = PoolLayer(net_h3)
            net_h4 = Conv2d(net_h4, 5 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                            padding='SAME', W_init=w_init, name='enc/h4/conv2d_a')
            net_h4 = Conv2d(net_h4, 5 * df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                            padding='SAME', W_init=w_init, name='enc/h4/conv2d_b')
        else:
            net_h4 = net_h3

        net_h5 = FlattenLayer(net_h4, name='enc/h5/flatten')
        net_h5 = DenseLayer(net_h5, n_units=z_dim, act=tf.identity,
                            W_init=w_init, name='enc/h5/lin')
    return net_h5


def decoder_api(inputs, is_train=True, reuse=False):
    image_size = FLAGS.output_size
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), \
                      int(image_size / 8), int(image_size / 16)
    df_dim = 64  # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim  # n_color 3
    batch_size = FLAGS.batch_size  # 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='dec/in')
        net_h0 = DenseLayer(net_in, n_units=df_dim * 8 * 8, W_init=w_init,
                            act=tf.identity, name='dec/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, 8, 8, df_dim], name='dec/h0/reshape')

        net_h1 = Conv2d(net_h0, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h1/conv2d_a')
        net_h1 = Conv2d(net_h1, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h1/conv2d_b')

        if image_size == 128:
            net_h2 = UpSampling2dLayer(net_h1, (s8, s8), is_scale=False, method=1)
            net_h2 = Conv2d(net_h2, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                            padding='SAME', W_init=w_init, name='dec/h2/conv2d_a')
            net_h2 = Conv2d(net_h2, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                            padding='SAME', W_init=w_init, name='dec/h2/conv2d_b')
        else:
            net_h2 = net_h1

        net_h3 = UpSampling2dLayer(net_h2, (s4, s4), is_scale=False, method=1)
        net_h3 = Conv2d(net_h3, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h3/conv2d_a')
        net_h3 = Conv2d(net_h3, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h3/conv2d_b')

        net_h4 = UpSampling2dLayer(net_h3, (s2, s2), is_scale=False, method=1)
        net_h4 = Conv2d(net_h4, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h4/conv2d_a')
        net_h4 = Conv2d(net_h4, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h4/conv2d_b')

        net_h5 = UpSampling2dLayer(net_h4, (image_size, image_size), is_scale=False, method=1)
        net_h5 = Conv2d(net_h5, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h5/conv2d_a')
        net_h5 = Conv2d(net_h5, df_dim, (3, 3), (1, 1), act=tf.nn.elu,
                        padding='SAME', W_init=w_init, name='dec/h5/conv2d_b')

        net_h6 = Conv2d(net_h5, c_dim, (3, 3), (1, 1), act=tf.identity,
                        padding='SAME', W_init=w_init, name='dec/h6/conv2d')

    return net_h6
