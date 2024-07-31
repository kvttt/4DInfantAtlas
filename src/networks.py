import torch
import torch.nn as nn

from src.film import FiLM
from src.layers import SpatialTransformer, VecInt, ResizeTransform
from src.mean_stream import MeanStream
from src.trilinear import TrilinearResizeLayer


class ConvBlockFiLM(nn.Module):
    """
    FiLMed convolutional block. input -> Conv -> SN -> FiLM -> ReLU -> output.
    """

    def __init__(
        self,
        z_dim=64,
        in_channel=32,
        out_channel=32,
        activation=True
    ):
        """
        :param z_dim: dimension of the conditional embedding vector z.
        :param in_channel: number of channels of the input tensor x.
        :param out_channel: number of channels of the output tensor x.
        :param activation: whether to apply LeakyReLU activation.
        """

        super(ConvBlockFiLM, self).__init__()

        self.activation = activation

        # Convolutional layer (bias to False because of FiLM)
        conv = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding='same',
            bias=False
        )

        # Spectral normalization
        self.conv = nn.utils.spectral_norm(conv)

        # FiLM layer
        self.film = FiLM(z_dim=z_dim, x_channel=out_channel)

        # ReLU activation
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        x = self.conv(x)
        x = self.film(x, z)
        if self.activation:
            x = self.relu(x)
        else:
            pass
        return x


class AtlasGen(nn.Module):
    """
    Atlas generation network
    Learns the residual between the rough template and the sharp atlas from age.
    """

    def __init__(
        self,
        num_res_blocks=5,
        clip_bkgnd=1e-1,
    ):
        """
        Sorry but most are hard-coded.

        :param num_res_blocks: number of ConvBlockFiLM residual blocks.
        :param clip_bkgnd: whether to clip the generated atlas.
        """
        super(AtlasGen, self).__init__()

        self.num_res_blocks = num_res_blocks
        self.clip_bkgnd = clip_bkgnd

        # MLP layer: maps age to conditional embedding z. (B, 1) -> (B, 64)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(0.2),
        )

        # "Learned parameters": maps age to a volume. (B, 1) -> (B, 8, 80, 96, 80)
        self.learned_params = nn.Linear(in_features=1, out_features=8 * 80 * 96 * 80)

        # First FiLM block before ConvBlockFiLM layers
        self.film0 = FiLM(z_dim=64, x_channel=8)

        # First ConvBlockFiLM layer
        self.conv_block1 = ConvBlockFiLM(z_dim=64, in_channel=8, out_channel=32)

        # ConvBlockFiLM layers
        self.conv_block = ConvBlockFiLM(z_dim=64, in_channel=32, out_channel=32)

        # Tri-linear resize layer
        self.trilinear = TrilinearResizeLayer(size_3d=(160, 192, 160))

        # Final ConvBlockFilM layers
        self.conv_block12 = ConvBlockFiLM(z_dim=64, in_channel=32, out_channel=8)
        self.conv_block13 = ConvBlockFiLM(z_dim=64, in_channel=8, out_channel=8)
        self.conv_block14 = ConvBlockFiLM(z_dim=64, in_channel=8, out_channel=8, activation=False)

        # Final Convolutional layer
        self.last_conv = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding='same')
        self.tanh = nn.Tanh()

    def forward(self, age, rough_template):
        """
        :param age: age tensor of shape (B, 1).
        :param rough_template: rough template tensor of shape (B, 1, 160, 192, 160).
        :return: atlas tensor of shape (B, 1, 160, 192, 160).
        """

        # MLP layer
        z = self.mlp(age)

        # "Learned parameters" layer
        learned_params = self.learned_params(age)
        learned_params = learned_params.view(-1, 8, 80, 96, 80)

        # First FiLM block
        x = self.film0(learned_params, z)

        # First ConvBlockFiLM layer
        x = self.conv_block1(x, z)

        # ConvBlockFiLM layers
        for _ in range(self.num_res_blocks):
            sip = x  # we use the same naming convention as in the paper and sip stands for skip input
            x = self.conv_block(x, z)
            x = self.conv_block(x, z)
            x = x + sip

        if self.use_SA:
            x = self.conv_block_sa(x)

        # Tri-linear resize layer
        x = self.trilinear(x)

        # Final ConvBlockFilM layers
        x = self.conv_block12(x, z)
        x = self.conv_block13(x, z)
        x = self.conv_block14(x, z)

        # Final Convolutional layer
        x = self.last_conv(x)
        x = self.tanh(x)

        # Add the rough template
        assert x.shape == rough_template.shape, 'shape of x and rough_template not match'
        x = x + rough_template

        t1 = rough_template[:, 0, :, :, :].unsqueeze(1)

        # Clip the atlas
        if self.clip_bkgnd is not None:
            x = x * (t1 > self.clip_bkgnd).float()

        return x


class UNet(nn.Module):
    """
    UNet for generating deformation field to warp the atlas to the target image.
    """

    def __init__(self, in_channel=8):
        """
        Sorry but most are hard-coded.

        :param in_channel: number of input channels.
        """
        super(UNet, self).__init__()

        # UNet down sample arm
        self.down0 = Down(in_channel=in_channel, out_channel=32)
        self.down1 = Down(in_channel=32, out_channel=32)
        self.down2 = Down(in_channel=32, out_channel=32)
        self.down3 = Down(in_channel=32, out_channel=32)

        # UNet bottleneck
        self.bottleneck = Up(in_channel=32, out_channel=32, is_bottleneck=True)

        # UNet up sample arm
        self.up0 = Up(in_channel=32, out_channel=32)
        self.up1 = Up(in_channel=64, out_channel=32)
        self.up2 = Up(in_channel=64, out_channel=32)
        self.up3 = Up(in_channel=64, out_channel=32, is_final=True)

        # Final convolutional layers
        self.conv0 = Up(in_channel=32, out_channel=32, is_final=True)

        self.conv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding='same')

        # VecInt: Integrate the velocity field.
        self.vecint = VecInt(inshape=(80, 96, 80), nsteps=5)

        # Rescale: Rescale the deformation field to (B, 3, 160, 192, 160).
        self.rescale = ResizeTransform(0.5, 3)

        # STN: Spatial transformer network.
        self.stn = SpatialTransformer(size=(160, 192, 160))

        # Mean Stream: Collect running average of the half-size deformation field.
        self.mean_stream = MeanStream()

    def forward(self, atlas, target):
        """
        :param atlas: atlas tensor of shape (B, 1, 160, 192, 160).
        :param target: target tensor of shape (B, 1, 160, 192, 160).
        :return: a total of 4 tensors:
            - x: moved atlas tensor of shape (B, 1, 160, 192, 160).
            - disp_field_ms: moving average of deformation field of shape (B, 3, 80, 96, 80).
            - atlas: generated atlas (unmoved) of shape (B, 1, 160, 192, 160).
            - disp_field_half: half size deformation field of shape (B, 3, 80, 96, 80)
        """

        # Concatenate the atlas and the target
        x = torch.cat((atlas, target), dim=1).float()

        # UNet down sample arm
        x = self.down0(x)  # x.shape = (B, 32, 80, 96, 80)
        x1 = x  # x1.shape = (B, 32, 80, 96, 80)
        x = self.down1(x)  # x.shape = (B, 32, 40, 48, 40)
        x2 = x  # x2.shape = (B, 32, 40, 48, 40)
        x = self.down2(x)  # x.shape = (B, 32, 20, 24, 20)
        x3 = x  # x3.shape = (B, 32, 20, 24, 20)
        x = self.down3(x)  # x.shape = (B, 32, 10, 12, 10)

        # UNet bottleneck
        x = self.bottleneck(x)  # x.shape = (B, 32, 10, 12, 10)

        # UNet up sample arm
        x = self.up0(x)  # x.shape = (B, 32, 20, 24, 20)
        x = torch.cat((x, x3), dim=1)  # x.shape = (B, 64, 20, 24, 20)
        x = self.up1(x)  # x.shape = (B, 32, 40, 48, 40)
        x = torch.cat((x, x2), dim=1)  # x.shape = (B, 64, 40, 48, 40)
        x = self.up2(x)  # x.shape = (B, 32, 80, 96, 80)
        x = torch.cat((x, x1), dim=1)  # x.shape = (B, 64, 80, 96, 80)
        x = self.up3(x)  # x.shape = (B, 32, 80, 96, 80)

        # Final convolutional layers
        x = self.conv0(x)  # x.shape = (B, 32, 80, 96, 80)
        x = self.conv1(x)  # x.shape = (B, 16, 80, 96, 80)
        x = self.conv2(x)  # x.shape = (B, 3, 80, 96, 80)

        # VecInt: Integrate the velocity field.
        x = self.vecint(x)  # x.shape = (B, 3, 80, 96, 80)
        disp_field_half = x
        disp_field_ms = self.mean_stream(x)

        # Rescale: Rescale the deformation field to (B, 3, 160, 192, 160).
        x = self.rescale(x)  # x.shape = (B, 3, 160, 192, 160)

        # STN: Spatial transformer network.
        x = self.stn(atlas, x)  # x.shape = (B, 1, 160, 192, 160)

        # return x, disp_field_ms, atlas, disp_field_half
        return x, disp_field_ms, atlas, disp_field_half


class Down(nn.Module):
    """
    Down sample layer for UNet.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        instance_norm=False
    ):
        """
        :param in_channel: number of input channels.
        :param out_channel: number of output channels.
        :param instance_norm: whether to use instance normalization.
        """
        super(Down, self).__init__()

        self.instance_norm = instance_norm

        # Convolution layer
        self.conv = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Instance normalization layer
        self.norm = nn.InstanceNorm3d(num_features=out_channel)

        # ReLU activation
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class Up(nn.Module):
    """
    Up sample layer for UNet.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        instance_norm=False,
        is_bottleneck=False,
        is_final=False
    ):
        """
        :param in_channel: number of input channels.
        :param out_channel: number of output channels.
        :param instance_norm: whether to use instance normalization.
        :param is_bottleneck: whether this is the bottleneck layer.
        :param is_final: whether this is one of the final convolution layers.
        """
        super(Up, self).__init__()

        self.instance_norm = instance_norm
        self.is_bottleneck = is_bottleneck
        self.is_final = is_final

        # Convolution layer
        self.conv = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        # Instance normalization layer
        self.norm = nn.InstanceNorm3d(num_features=out_channel)

        # ReLU activation
        self.relu = nn.LeakyReLU(0.2)

        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.norm(x)
        x = self.relu(x)
        if not self.is_bottleneck and not self.is_final:
            x = self.upsample(x)
        return x


class Generator(nn.Module):
    """
    Generator network.

    Consists of a FiLMed atlas generation network, a UNet, and an STN that use the output of the UNet to warp the atlas.
    """

    def __init__(
        self,
        clip_bkgnd=1e-1,
        pre_trained=False,
    ):
        """
        :param clip_bkgnd: the threshold used for clipping the background.
        :param pre_trained: whether to load the pre-trained model.
        """
        super(Generator, self).__init__()

        self.atlas_gen = AtlasGen(clip_bkgnd=clip_bkgnd)
        self.unet = UNet()

        if pre_trained:
            self.unet.load_state_dict(
                torch.load(
                    './model_195.pt',
                    map_location=torch.device('cuda:0')
                )
            )

        # Rescale: Rescale the deformation field to (B, 3, 160, 192, 160).
        self.rescale = ResizeTransform(0.5, 3)

        # STN: Spatial transformer network.
        self.stn = SpatialTransformer(size=(160, 192, 160))

    def forward(self, age, rough_template, target):
        """
        :param age: age tensor of shape (B, 1).
        :param rough_template: rough template tensor of shape (B, 1, 160, 192, 160).
        :param target: target tensor of shape (B, 1, 160, 192, 160).
        """

        # FiLMed atlas generation network
        atlas = self.atlas_gen(age, rough_template)

        return self.unet(atlas, target)  # moved_atlas, disp_field_ms, atlas, disp_field_half


class ConvBlock(nn.Module):
    """
    Convolutional encoder block used in the discriminator.
    """

    def __init__(
        self,
        in_channel,
        out_channel
    ):
        """
        :param in_channel: number of channels of the input tensor x.
        :param out_channel: number of channels of the output tensor x.
        """
        super(ConvBlock, self).__init__()

        # Convolutional layer
        conv = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        )

        # Spectral normalization
        self.conv = nn.utils.spectral_norm(conv)

        # ReLU activation
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network that assesses both the realism of the generated atlas and its specificity.
    """

    def __init__(self, in_channels=1):
        """
        :param in_channels: number of input channels.
        """
        super(Discriminator, self).__init__()

        # Convolutional encoder blocks (w/ spectral normalization)
        self.conv0 = ConvBlock(in_channel=in_channels, out_channel=64)
        self.conv1 = ConvBlock(in_channel=64, out_channel=128)
        self.conv2 = ConvBlock(in_channel=128, out_channel=256)
        self.conv3 = ConvBlock(in_channel=256, out_channel=512)

        # Convolutional encoder blocks (w/o spectral normalization)
        self.conv4 = nn.Conv3d(
            in_channels=512,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same',
            bias=True
        )
        self.conv5 = nn.Conv3d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding='same',
            bias=True
        )

        # Maps the age to a 1D vector. (B, 1) -> (B, 64)
        self.fc = nn.Linear(in_features=1, out_features=64)

    def forward(self, x, age):
        """
        :param x: warped atlas to be evaluated of shape (B, 1, 160, 192, 160).
        :param age: age of the subject of shape (B, 1).
        :return: output of the discriminator of shape (B, 1, 10, 12, 10).
        """

        # Convolutional encoder blocks
        x = self.conv0(x)  # x.shape = (B, 64, 80, 96, 80)
        x = self.conv1(x)  # x.shape = (B, 128, 40, 48, 40)
        x = self.conv2(x)  # x.shape = (B, 256, 20, 24, 20)
        x = self.conv3(x)  # x.shape = (B, 512, 10, 12, 10)
        x = self.conv4(x)  # x.shape = (B, 64, 10, 12, 10)
        x1 = x  # x1.shape = (B, 64, 10, 12, 10)
        x = self.conv5(x)  # x.shape = (B, 1, 10, 12, 10)

        # Maps the age to a 1D vector
        age = self.fc(age)  # age.shape = (B, 64)

        # Reshapes the age vector to a 5D vector
        age = age.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # age.shape = (B, 64, 1, 1, 1)

        # Broadcast the age vector to the same shape as x1
        age = torch.tile(age, (1, 1, 10, 12, 10))  # age.shape = (B, 64, 10, 12, 10)

        # Element-wise multiplication of age and x1
        x1 = x1 * age  # x1.shape = (B, 64, 10, 12, 10)

        # Summing along the channel dimension to reduce the number of channels from 64 to 1
        x1 = torch.sum(x1, dim=1, keepdim=False)  # x1.shape = (B, 1, 10, 12, 10)

        # Element-wise addition of x and x1
        x = x + x1  # x.shape = (B, 1, 10, 12, 10)

        return x
