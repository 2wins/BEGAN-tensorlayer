# BEGAN in TensorFlow / TensorLayer

TensorFlow / TensorLayer implementation of [BEGAN: Boundary Equilibrium Generative Adversarial Networks](http://arxiv.org/abs/1703.10717)


## Prerequisites
- Python 2.7 or Python 3.3+
- [TensorFlow==1.0+](https://www.tensorflow.org/)
- [TensorLayer==1.4+](https://github.com/tensorlayer/tensorlayer)


## Usage

First, download images to `data/celebA`:

    $ python download.py celebA		[202599 face images]

Second, train the GAN:

    $ python main.py --point "25 58"

Third, generate faces with the trained generator:

    $ python generate.py --num_imgs 1000


## Result on CelebA
From scratch to 60k (frames captured every 500 iter.)

`gamma=0.5`
<p>
<img src="img/training.gif"/>
</p>