import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=200):
        super(CNN, self).__init__()
        # 轻量深度可分离卷积网络：在不大幅增加参数和显存的前提下提升表达能力
        self.features = nn.Sequential(
            # Block 1: 常规卷积
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Depthwise + Pointwise
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: Depthwise + Pointwise
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: Depthwise + Pointwise（不再下采样）
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
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