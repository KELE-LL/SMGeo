import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorFreeHead(nn.Module):
    """
    @class AnchorFreeHead
    @desc 适配Swin-MoE主干输出的Anchor-Free检测头，输出中心点热力图和bbox参数
    @param in_channels: 输入特征通道数
    @param feat_channels: 中间特征通道数
    """
    def __init__(self, in_channels, feat_channels=256, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        # 输出：1通道中心点热力图 + 4通道bbox参数（中心点偏移、宽高）
        self.head_heatmap = nn.Conv2d(feat_channels, num_classes, 1)
        self.head_bbox = nn.Conv2d(feat_channels, 4, 1)
        # 新增：初始化heatmap分支bias为-2.0
        nn.init.constant_(self.head_heatmap.bias, -2.0)

    def forward(self, x):
        """
        @param x: [B, C, H, W]
        @return: heatmap [B, 1, H, W], bbox [B, 4, H, W]
        """
        feat = self.conv(x)
        heatmap = self.head_heatmap(feat)
        bbox = self.head_bbox(feat)
        return heatmap, bbox 