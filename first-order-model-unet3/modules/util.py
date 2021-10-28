from torch import nn

import torch.nn.functional as F
import torch
from .init_weights import init_weights

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class Decoder(nn.Module):
    """
    Hourglass Decoder
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

class DownBlock2d_Unet_3(nn.Module):
    """
    Downsampling block for use in encoder unit 3.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d_Unet_3, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        #self.pool= nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out     

class UpBlock2d_Unet_3(nn.Module):
    """
    Upsampling block for use in decoder Unet 3.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d_Unet_3, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out      

class Encoder_Unet_3(nn.Module):
    """
    Unet_3 Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder_Unet_3, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d_Unet_3(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class Decoder_Unet_3(nn.Module):
    """
    Unet_3 Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder_Unet_3, self).__init__()        

        in_filters=[]

        for i in range(num_blocks):
            in_filters.append(min(max_features,block_expansion * (2 ** (i + 1))))
        
        self.CatChannels = in_filters[0] #dimensión de los canales
        self.CatBlocks = 5 #número de bloques
        self.UpChannels = self.CatChannels * self.CatBlocks #número de canales de subida (320, ver el paper)
        self.n_classes= 35#1
        self.num_blocks = num_blocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        #self.h1_PT_hd4_conv = nn.Conv2d(in_filters[0], self.CatChannels, 3, padding=1)
        #self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd4_conv,self.h1_PT_hd4_bn=mid.conv,mid.norm
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        #self.h2_PT_hd4_conv = nn.Conv2d(in_filters[1], self.CatChannels, 3, padding=1)
        #self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_PT_hd4_conv,self.h2_PT_hd4_bn=mid.conv,mid.norm
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        #self.h3_PT_hd4_conv = nn.Conv2d(in_filters[2], self.CatChannels, 3, padding=1)
        #self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[2], self.CatChannels, kernel_size=3, padding=1)
        self.h3_PT_hd4_conv,self.h3_PT_hd4_bn=mid.conv,mid.norm
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        #self.h4_Cat_hd4_conv = nn.Conv2d(in_filters[3], self.CatChannels, 3, padding=1)
        #self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[3], self.CatChannels, kernel_size=3, padding=1)
        self.h4_Cat_hd4_conv,self.h4_Cat_hd4_bn=mid.conv,mid.norm
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        #self.hd5_UT_hd4_conv = nn.Conv2d(in_filters[4], self.CatChannels, 3, padding=1)
        #self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd4_conv,self.hd5_UT_hd4_bn=mid.conv,mid.norm
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)    

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        #self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        #self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv4d_1,self.bn4d_1=mid.conv,mid.norm
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        #self.h1_PT_hd3_conv = nn.Conv2d(in_filters[0], self.CatChannels, 3, padding=1)
        #self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd3_conv,self.h1_PT_hd3_bn=mid.conv,mid.norm
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        #self.h2_PT_hd3_conv = nn.Conv2d(in_filters[1], self.CatChannels, 3, padding=1)
        #self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_PT_hd3_conv,self.h2_PT_hd3_bn=mid.conv,mid.norm
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        #self.h3_Cat_hd3_conv = nn.Conv2d(in_filters[2], self.CatChannels, 3, padding=1)
        #self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[2], self.CatChannels, kernel_size=3, padding=1)
        self.h3_Cat_hd3_conv,self.h3_Cat_hd3_bn=mid.conv,mid.norm
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        #self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        #self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd3_conv,self.hd4_UT_hd3_bn=mid.conv,mid.norm
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        #self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        #self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd3_conv,self.hd5_UT_hd3_bn=mid.conv,mid.norm
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        #self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        #self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv3d_1,self.bn3d_1=mid.conv,mid.norm
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        #self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        #self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd2_conv,self.h1_PT_hd2_bn=mid.conv,mid.norm
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        #self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        #self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_Cat_hd2_conv,self.h2_Cat_hd2_bn=mid.conv,mid.norm
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        #self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        #self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd3_UT_hd2_conv,self.hd3_UT_hd2_bn=mid.conv,mid.norm
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        #self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        #self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd2_conv,self.hd4_UT_hd2_bn=mid.conv,mid.norm
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        #self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        #self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd2_conv,self.hd5_UT_hd2_bn=mid.conv,mid.norm
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        #self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        #self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv2d_1,self.bn2d_1=mid.conv,mid.norm
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        #self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        #self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_Cat_hd1_conv,self.h1_Cat_hd1_bn=mid.conv,mid.norm
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        #self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        #self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd2_UT_hd1_conv,self.hd2_UT_hd1_bn=mid.conv,mid.norm
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        #self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        #self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd3_UT_hd1_conv,self.hd3_UT_hd1_bn=mid.conv,mid.norm
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        #self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        #self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd1_conv,self.hd4_UT_hd1_bn=mid.conv,mid.norm
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        #self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        #self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd1_conv,self.hd5_UT_hd1_bn=mid.conv,mid.norm
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        #self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        #self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv1d_1,self.bn1d_1=mid.conv,mid.norm
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, self.n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
   
    
        #self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features
    """
    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out
    """

    def forward(self, inputs):

        #print(len(inputs))
        inputs.pop(0)
        #print(hs_len)
        hs=[]

        for x in range (self.num_blocks):
            hs.append(inputs[x])

        """
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024
        """      

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(hs[0]))))#h1
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(hs[1]))))#h2
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(hs[2]))))#h3
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(hs[3])))#h4
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hs[4]))))#hd5
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(hs[0]))))#h1
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(hs[1]))))#h2
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(hs[2])))#h3
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hs[4]))))#hd5
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(hs[0]))))#h1
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(hs[1])))#h2
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hs[4]))))#hd5
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(hs[0])))#h1
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hs[4]))))#hd5
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        out = F.sigmoid(d1)
        return out

class Unet_3_Plus(nn.Module):
    """
    UNET 3+ architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Unet_3_Plus, self).__init__()
        self.encoder = Encoder_Unet_3(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder_Unet_3(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class DownBlock2d_Unet_3_new_hope(nn.Module):
    """
    Downsampling block for use in encoder unit 3.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d_Unet_3_new_hope, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        #self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        #self.pool= nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        #out = self.pool(out)
        return out  

class Unet_3_Plus_new_hope(nn.Module):
    """
    UNET 3+ architecture.
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Unet_3_Plus_new_hope, self).__init__()
        #self.encoder = Encoder_Unet_3(block_expansion, in_features, num_blocks, max_features)
        #self.decoder = Decoder_Unet_3(block_expansion, in_features, num_blocks, max_features)
        #self.out_filters = self.decoder.out_filters
        in_filters=[]

        for i in range(num_blocks):
            in_filters.append(min(max_features,block_expansion * (2 ** (i + 1))))

        """
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d_Unet_3(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)
        """        

        self.conv1 = DownBlock2d_Unet_3_new_hope(self.in_channels, in_filters[0], kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = DownBlock2d_Unet_3_new_hope(in_filters[0], in_filters[1], kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = DownBlock2d_Unet_3_new_hope(in_filters[1], in_filters[2], kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = DownBlock2d_Unet_3_new_hope(in_filters[2], in_filters[3], kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = DownBlock2d_Unet_3_new_hope(in_filters[3], in_filters[4], kernel_size=3, padding=1)  
        
        self.CatChannels = in_filters[0] #dimensión de los canales
        self.CatBlocks = 5 #número de bloques
        self.UpChannels = self.CatChannels * self.CatBlocks #número de canales de subida (320, ver el paper)
        self.n_classes= 1
        self.num_blocks = num_blocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd4_conv,self.h1_PT_hd4_bn=mid.conv,mid.norm
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        mid=UpBlock2d_Unet_3(in_filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_PT_hd4_conv,self.h2_PT_hd4_bn=mid.conv,mid.norm
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        mid=UpBlock2d_Unet_3(in_filters[2], self.CatChannels, kernel_size=3, padding=1)
        self.h3_PT_hd4_conv,self.h3_PT_hd4_bn=mid.conv,mid.norm
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        mid=UpBlock2d_Unet_3(in_filters[3], self.CatChannels, kernel_size=3, padding=1)
        self.h4_Cat_hd4_conv,self.h4_Cat_hd4_bn=mid.conv,mid.norm
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd4_conv,self.hd5_UT_hd4_bn=mid.conv,mid.norm
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)    

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv4d_1,self.bn4d_1=mid.conv,mid.norm
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd3_conv,self.h1_PT_hd3_bn=mid.conv,mid.norm
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        mid=UpBlock2d_Unet_3(in_filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_PT_hd3_conv,self.h2_PT_hd3_bn=mid.conv,mid.norm
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        mid=UpBlock2d_Unet_3(in_filters[2], self.CatChannels, kernel_size=3, padding=1)
        self.h3_Cat_hd3_conv,self.h3_Cat_hd3_bn=mid.conv,mid.norm
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd3_conv,self.hd4_UT_hd3_bn=mid.conv,mid.norm
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd3_conv,self.hd5_UT_hd3_bn=mid.conv,mid.norm
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv3d_1,self.bn3d_1=mid.conv,mid.norm
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_PT_hd2_conv,self.h1_PT_hd2_bn=mid.conv,mid.norm
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        mid=UpBlock2d_Unet_3(in_filters[1], self.CatChannels, kernel_size=3, padding=1)
        self.h2_Cat_hd2_conv,self.h2_Cat_hd2_bn=mid.conv,mid.norm
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd3_UT_hd2_conv,self.hd3_UT_hd2_bn=mid.conv,mid.norm
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd2_conv,self.hd4_UT_hd2_bn=mid.conv,mid.norm
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd2_conv,self.hd5_UT_hd2_bn=mid.conv,mid.norm
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv2d_1,self.bn2d_1=mid.conv,mid.norm
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        mid=UpBlock2d_Unet_3(in_filters[0], self.CatChannels, kernel_size=3, padding=1)
        self.h1_Cat_hd1_conv,self.h1_Cat_hd1_bn=mid.conv,mid.norm
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd2_UT_hd1_conv,self.hd2_UT_hd1_bn=mid.conv,mid.norm
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd3_UT_hd1_conv,self.hd3_UT_hd1_bn=mid.conv,mid.norm
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(self.UpChannels, self.CatChannels, kernel_size=3, padding=1)
        self.hd4_UT_hd1_conv,self.hd4_UT_hd1_bn=mid.conv,mid.norm
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        mid=UpBlock2d_Unet_3(in_filters[4], self.CatChannels, kernel_size=3, padding=1)
        self.hd5_UT_hd1_conv,self.hd5_UT_hd1_bn=mid.conv,mid.norm
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        mid=UpBlock2d_Unet_3(self.UpChannels, self.UpChannels, kernel_size=3, padding=1)
        self.conv1d_1,self.bn1d_1=mid.conv,mid.norm
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, self.n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
   
    
        #self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):

        ## -------------Encoder-------------
        h1 = self.conv1(x)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))#h1
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))#h2
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))#h3
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))#h4
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))#hd5
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))#h1
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))#h2
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))#h3
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))#hd5
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))#h1
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))#h2
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))#hd5
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))#h1
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))#hd5
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        out = F.sigmoid(d1)
        return out


