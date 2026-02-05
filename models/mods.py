import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import math

class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer



class LogGaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, theta, f0, sigma):
        super(LogGaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.log_gabor_filter = self.create_log_gabor_filter(kernel_size, theta, f0, sigma)
        self.weight = nn.Parameter(self.log_gabor_filter.repeat(in_channels, 1, 1, 1), requires_grad=True)
    
    def create_log_gabor_filter(self, kernel_size, theta, f0, sigma):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        # Calculate the radius in the frequency domain
        radius = np.sqrt(x_theta**2 + y_theta**2)
        radius[radius == 0] = 1  # Avoid division by zero

        # Create the log Gabor filter
        gb = np.exp(- (np.log(radius / f0) ** 2) / (2 * np.log(sigma / f0) ** 2))
        gb = gb * np.cos(2 * np.pi * radius)
        gb = gb[np.newaxis, np.newaxis, :, :]  # Add batch and channel dimensions
        return torch.from_numpy(gb).float()

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2, groups=self.in_channels)

class LogGaborModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LogGaborModule, self).__init__()
        self.log_gabor1 = LogGaborConv2d(in_channels, in_channels, kernel_size=7, theta=0, f0=0.1, sigma=0.55)
        self.log_gabor2 = LogGaborConv2d(in_channels, in_channels, kernel_size=7, theta=np.pi / 4, f0=0.1, sigma=0.55)
        self.log_gabor3 = LogGaborConv2d(in_channels, in_channels, kernel_size=7, theta=np.pi / 2, f0=0.1, sigma=0.55)
        self.log_gabor4 = LogGaborConv2d(in_channels, in_channels, kernel_size=7, theta=3 * np.pi / 4, f0=0.1, sigma=0.55)
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.cross_attention0 = cross_attention(out_channels, num_heads=8)
        self.cross_attention1 = cross_attention(out_channels, num_heads=8)

    def forward(self, x):
        log_gabor1 = self.log_gabor1(x)
        log_gabor2 = self.log_gabor2(x)
        log_gabor3 = self.log_gabor3(x)
        log_gabor4 = self.log_gabor4(x)
        
        map1 = self.cross_attention0(log_gabor1, log_gabor2)
        map2 = self.cross_attention1(log_gabor3, log_gabor4) 
        LogGabor_map = torch.cat([map1, map2], dim=1)
        LogGabor_map = self.conv(LogGabor_map)
        return LogGabor_map


class HFB(nn.Module):
    expansion = 1
    def __init__(self, in_channels=3, dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        self.conv_head = Depth_conv(in_channels, dim)
        self.LogGabor = LogGaborModule(dim, dim)
        self.conv_tail = Depth_conv(dim, in_channels)
        self.norm = norm_layer(in_channels)

    def forward(self, x):
        
        residual = x

        x = self.conv_head(x)
        x = self.LogGabor(x)
        x = self.norm(self.conv_tail(x))
        # x = F.relu(x)

        return x + residual

# if __name__ == "__main__":
#     # Generate a random input tensor with values in the range [0, 255]
#     batch_size = 6
#     in_channels = 3
#     out_channels = 64
#     height = 128
#     width = 128
#     input_tensor = torch.randint(0, 256, (batch_size, in_channels, height, width), dtype=torch.float32)
    
#     # Instantiate the HFRM module
#     hfrm_module = HFB()
    
#     # Forward pass through the HFRM module
#     output_tensor = hfrm_module(input_tensor)
    
#     # Print input and output statistics
#     def print_stats(tensor, name):
#         print(f"{name} stats:")
#         print(f"  Min: {tensor.min().item()}")
#         print(f"  Max: {tensor.max().item()}")
#         print(f"  Mean: {tensor.mean().item()}")
#         print(f"  Std: {tensor.std().item()}")
#         print(f"  Shape: {tensor.shape}")
#         print()
    
#     print_stats(input_tensor, "Input Tensor")
#     print_stats(output_tensor, "Output Tensor")