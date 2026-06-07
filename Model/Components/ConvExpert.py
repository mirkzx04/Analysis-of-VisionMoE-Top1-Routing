import torch
import torch.functional as F

from torch import nn as  nn
# import timmù
class ConvExpert(nn.Module):
    def __init__(self, in_channel, out_channel, use_residual=True, kernel_size = 3):
        super().__init__()
        """
        Constructor of one convolution expert
        It is a ResNet-like block with two convolution layers, batch normalization, ReLU activation

        Args :
            kernel_size -> kernel size of convolution
            in_channel -> input channel of convolution
            out_channel -> output channel of convolution
        """

        self.use_residual = use_residual
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_size = out_channel * 1
        self.kernel_size = kernel_size
        self.final_act = nn.SiLU(inplace=False)

        padding = kernel_size // 2
        stride = 1

        self.last_conv = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=out_channel,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=False
        )

        # Define experts operation
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=padding,
                stride= stride,
                bias=False
            ),
            nn.GroupNorm(num_groups=min(8, self.hidden_size), num_channels=self.hidden_size),
            nn.SiLU(inplace=False),
            self.last_conv,
            nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel),
        )

        if self.use_residual:
            if stride != 1 or in_channel != out_channel:
                self.skip = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride= stride,
                        bias=False
                    ),
                    nn.GroupNorm(num_groups=min(8, out_channel), num_channels=out_channel),
                ) 
            else : 
                self.skip = nn.Identity()

        last_conv = self.conv_block[3]
        nn.init.normal_(last_conv.weight, mean=0.0, std=5e-3)

    def forward(self, X):
        out = self.conv_block(X)
        if self.use_residual:
            out = out + self.skip(X)
        return out
