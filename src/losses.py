import math

import torch
import torch.nn.functional as F

import numpy as np

from src.layers import ResizeTransform, SpatialTransformer


torch.cuda.set_device(0)


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    (https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py)
    """

    def __init__(self, win=None, n_channel=1):
        self.win = win
        self.n_channel = n_channel

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, self.n_channel, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class Grad:
    """
    N-D gradient loss.
    (https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py)
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1 - dice


loss_object = torch.nn.MSELoss()
loss_object_NCC = NCC()
loss_object_Dice = Dice()

# Load layers
rescale = ResizeTransform(0.5, 3).cuda()
stn = SpatialTransformer(size=(160, 192, 160)).cuda()


def generator_loss(
    disc_opinion_fake_local,
    disp_ms,
    disp,
    moved_atlases,
    fixed_images,
    loss_wts,
    reg_loss_type='NCC',
):
    """
    Generator loss function.
    :param disc_opinion_fake_local: Local feedback from discriminator.
    :param disp_ms: Moving average of displacement fields.
    :param disp: Displacement fields.
    :param moved_atlases: Moved atlases.
    :param fixed_images: Target images.
    :param loss_wts: List of regularization weights for gan loss, deformation, and TV (not used).
    :param reg_loss_type: Registration loss type.
    :return: All generator losses.
    """

    lambda_gan, lambda_reg, _ = loss_wts

    # GAN loss
    gan_loss = loss_object(
        torch.ones_like(disc_opinion_fake_local).cuda(),
        disc_opinion_fake_local
    )

    # Similarity terms
    moved_t1 = moved_atlases[:, 0, :, :, :].reshape(-1, 1, 160, 192, 160)  # extract only T1WI
    fixed_t1 = fixed_images[:, 0, :, :, :].reshape(-1, 1, 160, 192, 160)  # extract only T1WI
    if reg_loss_type == 'NCC':
        similarity_loss = torch.mean(
            loss_object_NCC.loss(
                moved_t1,
                fixed_t1
            )
        )
    else:
        raise ValueError('Unknown Reg Loss Type: {}'.format(reg_loss_type))

    # Dice term
    moved_tissue = moved_atlases[:, 1:, :, :, :].reshape(1, 3, 160, 192, 160)  # extract only tissue probability maps
    fixed_tissue = fixed_images[:, 1:, :, :, :].reshape(1, 3, 160, 192, 160)  # extract only tissue probability maps
    dice_loss = loss_object_Dice.loss(
        moved_tissue,
        fixed_tissue
    )

    # Smoothness term
    smoothness_loss = torch.mean(
        Grad('l2').loss(
            torch.zeros_like(disp).cuda(),
            disp
        )
    )

    # Magnitude terms
    magnitude_loss = loss_object(
        torch.zeros_like(disp).cuda(),
        disp
    )
    moving_magnitude_loss = loss_object(
        torch.zeros_like(disp_ms).cuda(),
        disp_ms
    )

    # Total generator loss
    total_gen_loss = (
        lambda_gan * gan_loss +
        (lambda_reg * smoothness_loss) +
        (0.01 * lambda_reg * magnitude_loss) +
        (lambda_reg * moving_magnitude_loss) +
        1 * similarity_loss +
        0.1 * dice_loss
    )

    return (total_gen_loss, gan_loss, smoothness_loss, magnitude_loss,
            similarity_loss, moving_magnitude_loss, dice_loss)


def discriminator_loss(
    disc_opinion_real_local,
    disc_opinion_fake_local,
):
    """
    Discriminator loss function.

    :param disc_opinion_real_local: Local feedback from discriminator on moved templates.
    :param disc_opinion_fake_local: Local feedback from discriminator on real fixed images.
    :return: All discriminator losses.
    """

    gan_fake_loss = loss_object(
        torch.zeros_like(disc_opinion_fake_local),
        disc_opinion_fake_local
    )
    gan_real_loss = loss_object(
        torch.ones_like(disc_opinion_real_local),
        disc_opinion_real_local
    )

    # Total discriminator loss
    total_loss = 0.5 * (gan_fake_loss + gan_real_loss)

    return total_loss
