"""
SimpleCNN architecture for transfer experiments.

A non-residual CNN to test if Infusion perturbations transfer across architectures.
Different from TinyResNet to test generalization of influence-based poisoning.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Non-residual CNN for transfer experiments.

    Architecture:
        Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool
        Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool
        Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool
        AdaptiveAvgPool → FC(128 → 10)

    Key differences from TinyResNet:
    - No residual connections
    - MaxPool instead of strided convolutions for downsampling
    - Same channel widths as ResNet (32→64→128)
    - Uses AdaptiveAvgPool like ResNet
    """

    def __init__(self, input_channels=3, num_classes=10, dropout_rate=0.0):
        super().__init__()

        # Block 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global pooling + classifier (like ResNet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def make_simple_cnn(input_channels=3, num_classes=10, device='cuda'):
    """Factory function to create SimpleCNN model."""
    return SimpleCNN(input_channels=input_channels, num_classes=num_classes).to(device)


if __name__ == "__main__":
    # Test the model
    model = SimpleCNN()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
