"""
Code for the BISCUIT model, 2016, Prabhakaran et al 2016
http://proceedings.mlr.press/v48/prabhakaran16.html
"""

# Author: Pedro Ferreira

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

import numpy as np

from models.networks import DefaultMLP
from models.utils import to_gpu, var_to_numpy, numpy_to_var
from models.losses import d_loss, g_loss
from data.utils import get_batch


class Biscuit(object):
    """
    This class allows for the training of Generative Adversarial Networks in their most basic form. It is designed to be
    as versatile as possible. Although the class can be instantiated with default parameters, it is fully customizable:
    you can use any generator, discriminator, optimizers and latent space. Currently, there are four loss functions
    available, corresponding to different "flavour" keyword arguments: 'x_entropy', 'least_squares', 'wasserstein' and
    'wasserstein-gp'. You can also make your GAN a Conditional GAN by setting the keyword argument "c_dim" to a positive
    integer, corresponding to the number of labels of your data.

    References: https://arxiv.org/abs/1406.2661 - GAN
                https://arxiv.org/abs/1611.04076 - LSGAN
                https://arxiv.org/abs/1701.07875 - WGAN
                https://arxiv.org/abs/1704.00028 - WGAN-GP

    :param x_dim: dimensionality of the data (not needed if both generator and discriminator are passed)
    :param generator: PyTorch module containing a neural network to act as the GAN's generator
    :param discriminator: PyTorch module containing a neural network to act as the GAN's discriminator
    :param flavour: loss function to use. May be 'x_entropy', 'least_squares', 'wasserstein', 'wasserstein-gp'
    :param c_dim: number of labels associated with the data
    :param z_dim: latent space dimensionality
    :param h_dim: number of hidden units in the networks. Used only if generator and discriminator are not passed
    :param use_cuda: whether to run everything on GPU or not
    """
    def __init__(self, x_dim=None, generator=None, discriminator=None, flavour='x_entropy', c_dim=0, z_dim=16, h_dim=100, use_cuda=True):
        if generator is None:
            assert x_dim is not None
            generator = DefaultMLP(z_dim + c_dim, h_dim, x_dim, out_activation=nn.Sigmoid())
        if discriminator is None:
            assert x_dim is not None
            if flavour == 'wasserstein':
                discriminator = DefaultMLP(x_dim + c_dim, h_dim, z_dim)
            else:
                discriminator = DefaultMLP(x_dim + c_dim, h_dim, z_dim, out_activation=nn.Sigmoid())
        self.generator = to_gpu(use_cuda, generator)
        self.discriminator = to_gpu(use_cuda, discriminator)

        self.flavour = flavour

        self.z_dim = z_dim
        self.c_dim = c_dim  # for conditional data generation
        self.use_cuda = use_cuda

        if flavour == 'wasserstein':
            # WGAN uses RMS prop optimization with low learning rates
            self.generator_optimizer = optim.RMSprop(self.generator.parameters(), lr=5e-4)
            self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=5e-4)
        else:
            self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-3)

        # Sampler from the latent space probability distribution
        self.z_sampler = lambda n: to_gpu(self.use_cuda, Variable(torch.randn(n, self.z_dim)))

    def generate(self, num_samples=None, discrete_index=None, z_feed=None, to_numpy=True):
        """
        Sample the generator network. If no num_samples is passed, z_feed must be, and vice-versa. If the GAN is a CGAN,
        discrete_index must be passed.

        :param num_samples: number of samples to generate
        :param discrete_index: the index of the label of the pattern to generate, in the case of a CGAN
        :param z_feed: latent space sample
        :param to_numpy: whether to return the data as a numpy array
        :return: (num_samples x x_dim)-shaped numpy array or torch Variable, depending on "to_numpy" argument
        """

        if z_feed is None:
            assert num_samples is not None
            z_feed = self.sample_z(num_samples)
        else:
            z_feed = to_gpu(self.use_cuda, numpy_to_var(z_feed))

        if self.c_dim != 0:  # check if is CGAN
            assert discrete_index is not None
            c = np.zeros([num_samples, self.c_dim])
            c[range(num_samples), discrete_index] = 1
            c = to_gpu(True, numpy_to_var(c))
            generated = self.generator(z_feed, c)
        else:
            generated = self.generator(z_feed)

        if to_numpy:
            return var_to_numpy(generated)
        return generated

    def fit(self, patterns, labels=None, num_epochs=100, batch_size=50, d_rounds=1, clip_thres=0.01, lam=10, verbose=True):
        """
        Train the GAN.

        :param patterns: the training data
        :param labels: one-hot encoded labels array
        :param num_epochs: number of epochs to run
        :param batch_size: number of samples per batch
        :param d_rounds: number of discriminator optimization steps per generator steps
        :param clip_thres: weight-clipping threshold for flavour='wasserstein'
        :param lam: weight of gradient penalization for flavour='wasserstein-gp'
        :param verbose: whether to print out the losses at the end of each epoch
        :return: numpy arrays with discriminator and generator losses over time
        """

        discriminator_loss_hist = np.zeros((num_epochs, 1))
        generator_loss_hist = np.zeros((num_epochs, 1))

        for epoch in range(num_epochs):
            discriminator_loss, generator_loss = self.train_epoch(patterns, batch_size, labels=labels,
                                                                  d_rounds=d_rounds, clip_thres=clip_thres, lam=lam)

            discriminator_loss_hist[epoch] = var_to_numpy(discriminator_loss)
            generator_loss_hist[epoch] = var_to_numpy(generator_loss)

            if verbose:
                print("Epoch %d, Discriminator loss %f, Generator loss %f" % (epoch, discriminator_loss_hist[epoch],
                                                                              generator_loss_hist[epoch]))

        return discriminator_loss_hist, generator_loss_hist

    def generate_conditioned(self, num_samples, discrete_index, z_feed=None, to_numpy=True):
        """
        Generate data conditioned on the label given by discrete_index.

        :param num_samples: number of conditioned samples to generate
        :param discrete_index: index of the desired condition. must be < self.c_dim
        :param z_feed: latent feed variable
        :param to_numpy: whether to return the samples as a numpy array
        :return: the generated conditioned samples
        """

        assert self.c_dim != 0

        if z_feed is None:
            z_feed = self.sample_z(num_samples)

        c = np.zeros([num_samples, self.c_dim])
        c[range(num_samples), discrete_index] = 1
        c = to_gpu(True, numpy_to_var(c))

        generated = self.generator(z_feed, c)
        if to_numpy:
            return var_to_numpy(generated)
        return generated

    def set_optimizers(self, generator_optimizer, discriminator_optimizer):
        """
        :param generator_optimizer: optimizer for G
        :param discriminator_optimizer: optimizer for D
        """

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def get_optimizers(self):
        return self.generator_optimizer, self.discriminator_optimizer

    def reset_gradients(self):
        """
        Clean the gradients of both the generator and discriminator

        :return: None
        """

        self.generator.zero_grad()
        self.discriminator.zero_grad()

    def set_z_sampler(self, func):
        """
        Set the sampler from the latent space probability distribution

        :param func: a function with an argument defining the number of samples to return
        """

        self.z_sampler = func

    def sample_z(self, num_samples, to_numpy=False):
        """
        Returns samples from the latent space distribution.

        :param num_samples: number of samples to return
        :param to_numpy: whether to return the samples as a numpy array
        :return: (num_samples x self.z_dim)-shaped numpy array or torch Variable, depending on "to_numpy" argument
        """

        z = to_gpu(self.use_cuda, numpy_to_var(self.z_sampler(num_samples)))
        if to_numpy:
            return var_to_numpy(z)
        return z

    def train_epoch(self, samples, batch_size, labels=None, d_rounds=1, clip_thres=0.01, lam=10):
        """
        Train the models for one epoch (one full pass of the training set, updating the model's parameters batch_size
        times)

        :param samples: the real data. Can be either a torch Tensor or a numpy array of shape
                        (num_samples x num_features)
        :param batch_size: number of samples per batch
        :param labels: labels, for if the GAN is a CGAN.
        :param d_rounds: number of discriminator optimization steps per generator steps
        :param clip_thres: weight-clipping threshold for flavour='wasserstein'
        :param lam: weight of gradient penalization for flavour='wasserstein-gp'
        :return: discriminator and generator losses as PyTorch Variables
        :raises: ValueError if number of samples == batch_size
        """

        num_samples = samples.shape[0]
        if num_samples == batch_size:
            print("Error: samples.shape[0] can't be equal to batch_size.")
            raise ValueError

        samples = to_gpu(self.use_cuda, numpy_to_var(samples))
        if labels is not None:
            labels = to_gpu(self.use_cuda, numpy_to_var(labels))

        discriminator_loss = 0
        generator_loss = 0
        labels_batch = None

        for batch_idx in range(0, int(num_samples / batch_size) - d_rounds, d_rounds):
            for d_iter in range(d_rounds):
                patterns_batch = get_batch(samples, batch_size, batch_idx + d_iter, labels)
                labels_batch = None
                if labels is not None:
                    labels_batch = patterns_batch[1]
                patterns_batch = patterns_batch[0]
                discriminator_loss = self.train_d(patterns_batch, batch_size, labels_batch=labels_batch, clip_thres=clip_thres, lam=lam)

            generator_loss = self.train_g(batch_size, labels_batch=labels_batch)

        return discriminator_loss, generator_loss

    def train_d(self, patterns_batch, batch_size, labels_batch=None, clip_thres=0.01, lam=10):
        """
        Update the discriminator's parameters.

        :param patterns_batch: a batch of patterns
        :param batch_size: number of samples per batch
        :param labels_batch: a batch of labels, for if the GAN is a CGAN.
        :param clip_thres: weight-clipping threshold for flavour='wasserstein'
        :param lam: weight of gradient penalization for flavour='wasserstein-gp'
        :return: discriminator loss as a PyTorch Variable
        """

        self.reset_gradients()
        self.generator.eval()

        if self.flavour == 'wasserstein-gp':
            return self.train_d_gp(patterns_batch, batch_size, labels_batch, lam)

        if labels_batch is not None:
            discriminator_loss = self.train_d_conditioned(patterns_batch, labels_batch, batch_size, clip_thres)
        else:
            fake_sample = self.generator(self.sample_z(batch_size))

            discriminator_real = self.discriminator(patterns_batch)
            discriminator_fake = self.discriminator(fake_sample)

            discriminator_loss = d_loss(self.flavour, discriminator_real, discriminator_fake)

            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if self.flavour == 'wasserstein':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_thres, clip_thres)

        return discriminator_loss

    def train_d_conditioned(self, patterns_batch, labels_batch, batch_size, clip_thres=0.01):
        """
        Updating scheme for the discriminator's parameters in the case of a CGAN.

        :param patterns_batch: a batch of patterns
        :param batch_size: number of samples per batch
        :param labels_batch: a batch of labels, for if the GAN is a CGAN.
        :param clip_thres: weight-clipping threshold for flavour='wasserstein'
        :return: discriminator loss as a PyTorch Variable
        """

        assert labels_batch.size(1) == self.c_dim

        fake_sample = self.generator(self.sample_z(batch_size), labels_batch)

        discriminator_real = self.discriminator(patterns_batch, labels_batch)
        discriminator_fake = self.discriminator(fake_sample, labels_batch)

        discriminator_loss = d_loss(self.flavour, discriminator_real, discriminator_fake)

        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        if self.flavour == 'wasserstein':
            for p in self.discriminator.parameters():
                p.data.clamp_(-clip_thres, clip_thres)

        return discriminator_loss

    def train_d_gp(self, patterns_batch, batch_size, labels_batch=None, lam=10):
        """
        Updating scheme for the discriminator's parameters in the case of self.flavour='wasserstein-gp'.

        :param patterns_batch: a batch of patterns
        :param batch_size: number of samples per batch
        :param labels_batch: a batch of labels, for if the GAN is a CGAN.
        :param lam: weight of gradient penalization
        :return: discriminator loss as a PyTorch Variable
        """

        self.reset_gradients()
        self.generator.eval()

        if labels_batch is not None:
            discriminator_loss = self.train_d_gp_conditioned(patterns_batch, labels_batch, batch_size, lam)
        else:
            eps_batch = to_gpu(self.use_cuda, Variable(torch.rand(batch_size, 1)))
            eps_batch = eps_batch.expand(patterns_batch.size())

            fake_batch = self.generator(self.sample_z(batch_size))
            x_hat = eps_batch * patterns_batch + (1 - eps_batch) * fake_batch

            discriminator_hat = self.discriminator(x_hat)

            gradients = autograd.grad(outputs=discriminator_hat, inputs=x_hat,
                                      grad_outputs=to_gpu(self.use_cuda, torch.ones(discriminator_hat.size())),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2)) * lam

            discriminator_fake = self.discriminator(fake_batch)
            discriminator_real = self.discriminator(patterns_batch)

            discriminator_loss = torch.mean(discriminator_fake) - torch.mean(discriminator_real) + gradient_penalty
            # discriminator_loss = d_loss(self.flavour, discriminator_real, discriminator_fake) + gradient_penalty

            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        return discriminator_loss

    def train_d_gp_conditioned(self, patterns_batch, labels_batch, batch_size, lam=10):
        """
        Updating scheme for the discriminator's parameters in the case of self.flavour='wasserstein-gp' and CGAN

        :param patterns_batch: a batch of patterns
        :param batch_size: number of samples per batch
        :param labels_batch: a batch of labels, for if the GAN is a CGAN.
        :param lam: weight of gradient penalization
        :return: discriminator loss as a PyTorch Variable
        """

        assert labels_batch.size(1) == self.c_dim

        eps_batch = to_gpu(self.use_cuda, Variable(torch.rand(batch_size, 1)))
        eps_batch = eps_batch.expand(patterns_batch.size())

        fake_batch = self.generator(self.sample_z(batch_size), labels_batch)
        x_hat = eps_batch * patterns_batch + (1 - eps_batch) * fake_batch

        discriminator_hat = self.discriminator(x_hat, labels_batch)

        gradients = autograd.grad(outputs=discriminator_hat, inputs=x_hat,
                                  grad_outputs=to_gpu(self.use_cuda, torch.ones(discriminator_hat.size())),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2)) * lam

        discriminator_fake = self.discriminator(fake_batch, labels_batch)
        discriminator_real = self.discriminator(patterns_batch, labels_batch)

        discriminator_loss = d_loss(self.flavour, discriminator_real, discriminator_fake) + gradient_penalty

        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss

    def train_g(self, batch_size, labels_batch=None):
        """
        Update the generator's parameters.

        :param batch_size: number of samples per batch
        :param labels_batch: a batch of labels, for if the GAN is a CGAN.
        :return: generator loss as a PyTorch Variable
        """

        self.reset_gradients()
        self.generator.eval()

        if labels_batch is not None:
            generator_loss = self.train_g_conditioned(batch_size, labels_batch)
        else:
            fake_sample = self.generator(self.sample_z(batch_size))

            discriminator_fake = self.discriminator(fake_sample)

            generator_loss = g_loss(self.flavour, discriminator_fake)

            generator_loss.backward()
            self.generator_optimizer.step()

        return generator_loss

    def train_g_conditioned(self, batch_size, labels_batch):
        """
        Update the generator's parameters for when the GAN is a CGAN

        :param batch_size: number of samples per batch
        :param labels_batch: a batch of labels
        :return: generator loss as a PyTorch Variable
        """

        assert labels_batch.size(1) == self.c_dim

        fake_sample = self.generator(self.sample_z(batch_size), labels_batch)

        discriminator_fake = self.discriminator(fake_sample, labels_batch)

        generator_loss = g_loss(self.flavour, discriminator_fake)

        generator_loss.backward()
        self.generator_optimizer.step()

        return generator_loss
