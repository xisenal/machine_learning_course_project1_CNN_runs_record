import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=200):
        super(CNN, self).__init__()
        # 轻量级的三层卷积网络，控制通道数量以降低显存占用
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 使用自适应平均池化避免全连接层输入维度受输入图像大小影响
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
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