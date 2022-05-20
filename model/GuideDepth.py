import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DDRNet_23_slim import DualResNet_Backbone
from model.modules import Guided_Upsampling_Block, SELayer


class GuideDepth(nn.Module):
    def __init__(self, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(GuideDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")


    def forward(self, x):
        y = self.feature_extractor(x)

        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_3(x, y)
        return y
