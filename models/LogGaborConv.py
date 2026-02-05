import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        log_gabor1 = self.log_gabor1(x)
        log_gabor2 = self.log_gabor2(x)
        log_gabor3 = self.log_gabor3(x)
        log_gabor4 = self.log_gabor4(x)
        
        LogGabor_map = torch.cat([log_gabor1, log_gabor2, log_gabor3, log_gabor4], dim=1)
        LogGabor_map = self.conv(LogGabor_map)
        return LogGabor_map
