import torch


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ResNet9(torch.nn.Module):
    """
    ResNet-9 model for CIFAR-10/100.

    Input shape: (batch_size, 3, 32, 32)
    - batch_size: Number of images in the batch
    - 3: RGB channels
    - 32: Height in pixels
    - 32: Width in pixels

    This model is designed for CIFAR datasets which have 32x32 RGB images.
    """

    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )
        self.res1 = Residual(torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        ))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.MaxPool2d(2)
        self.res2 = Residual(torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        ))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                256,
                128,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )
        self.adapool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = torch.nn.Linear(128, num_classes, bias=False)
        self.mul = Mul(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.adapool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.mul(x)
        return x
