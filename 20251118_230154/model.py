import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=200):
        super(CNN, self).__init__()
        class SqueezeExcitation(nn.Module):
            def __init__(self, channels, reduction=8):
                super().__init__()
                self.avg = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, max(1, channels // reduction)),
                    nn.ReLU(inplace=True),
                    nn.Linear(max(1, channels // reduction), channels),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                w = self.avg(x).view(b, c)
                w = self.fc(w).view(b, c, 1, 1)
                return x * w

        class DepthwiseSeparableBlock(nn.Module):
            def __init__(self, in_channels, out_channels, use_se=True, residual=True):
                super().__init__()
                self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
                self.dw_bn = nn.BatchNorm2d(in_channels)
                self.se = SqueezeExcitation(in_channels) if use_se else nn.Identity()
                self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                self.pw_bn = nn.BatchNorm2d(out_channels)
                self.act = nn.ReLU(inplace=True)
                self.residual = residual and (in_channels == out_channels)

            def forward(self, x):
                y = self.act(self.dw_bn(self.dw(x)))
                y = self.se(y)
                y = self.act(self.pw_bn(self.pw(y)))
                if self.residual:
                    y = y + x
                return y

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DepthwiseSeparableBlock(32, 64, use_se=True, residual=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DepthwiseSeparableBlock(64, 128, use_se=True, residual=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DepthwiseSeparableBlock(128, 128, use_se=True, residual=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        # 简要参数统计（不在前向中使用，仅方便调试）
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"Model params: {total_params/1e6:.2f}M")

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x