import os
import pprint
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("gpu_num", 0, "GPU number to use [0]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("num_imgs", 100, "The number of generated images. [100]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 64, "Dimension of seed vector [64]")
flags.DEFINE_string("load_dir", "checkpoint/celebA", "Directory name to load the checkpoints [checkpoint/celebA]")
flags.DEFINE_string("save_dir", "result", "Directory name to save the images [result]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if tl.files.folder_exists(FLAGS.checkpoint_dir) is False:
        raise ValueError("checkpoint_dir {} does not exist.".format(FLAGS.checkpoint_dir))
        
    with tf.device("/gpu:{}".format(FLAGS.gpu_num)):
        z = tf.placeholder(tf.float32, [1, FLAGS.z_dim], name='z_noise')

        net_g = generator_api(z, is_train=False)
        net_g.print_params(False)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    load_dir = FLAGS.load_dir
    tl.files.exists_or_mkdir(FLAGS.save_dir)

    # load the latest checkpoint
    net_g_name = os.path.join(load_dir, 'net_g.npz')
    print("[*] Loading checkpoints...")
    if tl.files.load_and_assign_npz(sess=sess, name=net_g_name, network=net_g) is False:
        print("[*] Loading checkpoints FAILURE!")
    else:
        print("[*] Loading checkpoints SUCCESS!")
        for i in range(FLAGS.num_imgs):
            sample_seed = np.random.uniform(-1, 1, size=(1, FLAGS.z_dim)).astype(np.float32)
            img = sess.run(net_g.outputs, feed_dict={z: sample_seed})
            img = np.round((img + 1) * 127.5).astype(np.uint8)
            tl.visualize.save_image(img[0, :], os.path.join(FLAGS.save_dir, '{:06d}.png'.format(i)))

if __name__ == '__main__':
    tf.app.run()
