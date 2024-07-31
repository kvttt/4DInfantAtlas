import torch
import torch.nn as nn
import torch.nn.functional as F


class TrilinearResizeLayer(nn.Module):
    """
    Resize 3D volumes (without trainable parameters)
    """

    def __init__(self, size_3d):
        """
        :param size_3d: 3-element integers set the output 3D spatial shape e.g. (D, H, W)
        """
        super(TrilinearResizeLayer, self).__init__()
        self.size = size_3d

    def forward(self, x):
        """
        :param x: input tensor (3D volume) of shape B x C x D x H x W
        :return: interpolated volume of shape B x C x size_3d[0] x size_3d[1] x size_3d[2].
        """
        return F.interpolate(x, size=self.size, mode='trilinear')  # align_corners=False to match TF2 implementation
