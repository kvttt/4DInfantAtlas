import torch
import torch.nn as nn


# import tensorflow as tf


# def _film_reshape(gamma, beta, x):
#     """
#     Reshape gamma and beta for FiLM.
#     """
#
#     gamma = tf.tile(
#         tf.reshape(gamma, (tf.shape(gamma)[0], 1, 1, 1, tf.shape(gamma)[-1])),
#         (1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], 1))
#     beta = tf.tile(
#         tf.reshape(beta, (tf.shape(beta)[0], 1, 1, 1, tf.shape(beta)[-1])),
#         (1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], 1))
#
#     return gamma, beta


def _film_reshape_torch(gamma, beta, x):
    """
    Reshape gamma and beta for FiLM.
    :param gamma: scale parameter learned from z.
    :param beta: shift parameter learned from z.
    :param x: input tensor.
    :return: reshaped gamma and beta.
    """

    gamma = torch.tile(
        torch.reshape(
            gamma,
            (gamma.shape[0], gamma.shape[1], 1, 1, 1)
        ),
        (1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
    )
    beta = torch.tile(
        torch.reshape(
            beta,
            (beta.shape[0], beta.shape[1], 1, 1, 1)
        ),
        (1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
    )

    return gamma, beta


class FiLM(nn.Module):
    """
    FiLM layer by (Perez et al., 2017).
    """

    def __init__(self, z_dim=64, x_channel=32, init='default', wt_decay=None):
        """
        :param z_dim: dimension of the conditional embedding vector z.
        :param x_channel: number of channels of the input tensor x.
        :param init: How to initialize dense layers.
        :param wt_decay: L2 penalty on FiLM projection.
        """
        super(FiLM, self).__init__()

        self.z_dim = z_dim
        self.x_channel = x_channel
        self.wt_decay = wt_decay  # not used
        self.fc = nn.Linear(self.z_dim, 2 * self.x_channel)
        if init == 'orthogonal':
            nn.init.orthogonal_(self.fc.weight)
        elif init == 'default' or init is None:
            pass
        else:
            raise ValueError('Unknown init: {}'.format(init))

    def hypernetwork(self, z):
        """
        :param z: latent tensor.
        """
        x = self.fc(z)
        return x[:, :self.x_channel, ...], x[:, self.x_channel:, ...]

    def forward(self, x, z):
        """
        :param x: input tensor.
        :param z: latent tensor.
        :return: FiLMed tensor.
        """
        gamma, beta = self.hypernetwork(z)
        gamma, beta = _film_reshape_torch(gamma, beta, x)

        return (1. + gamma) * x + beta
