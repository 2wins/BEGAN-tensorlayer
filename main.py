import os
import pprint
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from random import shuffle
from model import *
from utils import *

pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of BEGAN to generate face image.

Usage : see README.md
"""
flags = tf.app.flags
flags.DEFINE_integer("gpu_num", 0, "GPU number to use [0]")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_string("point", None, "The starting point (x, y) of cropped region [None]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be cropped) [128]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 25, "The number of sample images [25]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 64, "Dimension of seed vector [64]")
flags.DEFINE_float("kt", 0.0, "Parameter kt [0.0]")
flags.DEFINE_float("gamma", 0.5, "Parameter gamma [0.5]")
flags.DEFINE_float("lamda", 0.001, "Parameter lambda [0.001]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saving checkpoints. [500]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("data_dir", "data", "Directory name to contain the dataset [data]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    kt_np = np.float32(FLAGS.kt)
    lr_np = np.float32(FLAGS.learning_rate)
    if FLAGS.point is None:
        point = None
    else:
        point = [int(x) for x in FLAGS.point.split()]
        assert len(point) == 2, "invalid starting point \"{}\" for cropping.".format(FLAGS.point)

    with tf.device("/gpu:{}".format(FLAGS.gpu_num)):
        ##========================= DEFINE MODEL ===========================##
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='z_noise')

        # for generating sample images
        z_s = tf.placeholder(tf.float32, [FLAGS.sample_size, FLAGS.z_dim], name='z_sample')
        
        real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')
        kt = tf.placeholder(tf.float32, name='kt')
        lr = tf.placeholder(tf.float32, name='lr')

        # z --> generator for training
        net_g = generator_api(z, is_train=FLAGS.is_train, reuse=False)
        net_gs = generator_api(z_s, is_train=False, reuse=True)

        # generated fake images --> discriminator
        net_enc = encoder_api(net_g.outputs, is_train=FLAGS.is_train, reuse=False)
        net_d = decoder_api(net_enc.outputs, is_train=FLAGS.is_train, reuse=False)

        # real images --> discriminator
        net_enc2 = encoder_api(real_images, is_train=FLAGS.is_train, reuse=True)
        net_d2 = decoder_api(net_enc2.outputs, is_train=FLAGS.is_train, reuse=True)

        # z --> decoder (reconstruction)
        net_d3 = decoder_api(z, is_train=FLAGS.is_train, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        # cost for updating discriminator and generator
        d_loss_real = tl.cost.absolute_difference_error(net_d2.outputs, real_images, is_mean=True)
        d_loss_fake = tl.cost.absolute_difference_error(net_d.outputs, net_g.outputs, is_mean=True)
        d_loss = d_loss_real - kt * d_loss_fake

        g_loss = d_loss_fake
        m_global = d_loss_real + tf.abs(FLAGS.gamma * d_loss_real - d_loss_fake)

        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

        net_g.print_params(False)
        print("---------------")
        net_d.print_params(False)

        # optimizers for updating discriminator and generator
        d_optim = tf.train.AdamOptimizer(lr, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(lr, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    model_dir = "%s_%s" % (FLAGS.dataset, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    tl.files.exists_or_mkdir(save_dir)

    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    # load the list of image files
    data_files = glob(os.path.join(FLAGS.data_dir, FLAGS.dataset, "*.jpg"))

    sample_seed = np.random.uniform(-1, 1, size=(FLAGS.sample_size, FLAGS.z_dim)).astype(np.float32)

    ##========================= TRAIN MODELS ================================##
    iter_counter = 0
    for epoch in range(FLAGS.epoch):
        # shuffle data
        shuffle(data_files)

        # update sample files based on shuffled data
        sample_files = data_files[0:FLAGS.batch_size]
        sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        print("[*] Sample images updated!")

        # load image data
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

        for idx in range(0, batch_idxs):
            batch_files = data_files[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size]
            # get real images
            # more image augmentation functions in http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html
            batch = [get_image(batch_file, FLAGS.image_size, point=point, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z = np.random.uniform(-1, 1, size=(FLAGS.batch_size, FLAGS.z_dim)).astype(np.float32)
            start_time = time.time()

            # update the networks
            feed_dict = {z: batch_z, real_images: batch_images, kt: kt_np, lr: lr_np}
            errG, _ = sess.run([g_loss, g_optim], feed_dict=feed_dict)
            errD, _ = sess.run([d_loss, d_optim], feed_dict=feed_dict)
            errM, dlr, dlf = sess.run([m_global, d_loss_real, d_loss_fake], feed_dict=feed_dict)

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, m_global: %.8f"
                  % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errD, errG, errM))

            # update the parameter kt
            kt_np = np.maximum(np.minimum(1., kt_np + FLAGS.lamda * (FLAGS.gamma * dlr - dlf)), 0.)

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # generate and visualize generated images
                img = sess.run(net_gs.outputs, feed_dict={z_s: sample_seed})
                img = np.round((img + 1) * 127.5).astype(np.uint8)
                tl.visualize.save_images(img, [5, 5], './{}/train_{:06d}.png'.format(FLAGS.sample_dir, iter_counter))

            if np.mod(iter_counter, FLAGS.save_step) == 0:
                # save current network parameters
                print("[*] Saving checkpoints...")
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")

        # update learning rate
        lr_np *= 0.95


if __name__ == '__main__':
    tf.app.run()
