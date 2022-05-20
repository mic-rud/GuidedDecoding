import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2,3]) # Replacement of avgPool for large kernels for trt
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)



class Guided_Upsampling_Block(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(Guided_Upsampling_Block, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(self.guide_features, expand_features,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type =='raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)


    def forward(self, guide, depth):
        x = self.feature_conv(depth)

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            xy = torch.cat([x, y], dim=1)
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + depth)
