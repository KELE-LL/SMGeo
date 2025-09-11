# -*- coding:utf8 -*-
"""
@file detgeo_swinmoe.py
@desc DetGeo-main集成Swin-MoE多输入主干，支持配置文件集中管理参数。
"""
from model.swin_moe_geo import SwinTransformer_MoE_MultiInput
from model.swin_moe_geo_config import swin_moe_geo_cfg

# 可选：定义一个工厂函数，便于主程序调用

def build_swinmoe_backbone():
    """
    @function build_swinmoe_backbone
    @desc 根据配置文件构建Swin-MoE多输入主干
    @return {nn.Module} SwinTransformer_MoE_MultiInput实例
    """
    cfg = swin_moe_geo_cfg
    return SwinTransformer_MoE_MultiInput(
        in_channels=cfg.get('in_channels', 3),
        embed_dim=cfg.get('embed_dim', 96),
        patch_size=cfg.get('patch_size', 4),
        window_size=cfg.get('window_size', 8),
        depths=cfg.get('depths', (2,2,6,2)),
        num_heads=cfg.get('num_heads', (3,6,12,24)),
        ffn_ratio=cfg.get('ffn_ratio', 4),
        num_experts=cfg.get('num_experts', 4),
        top_k=cfg.get('top_k', 2),
        moe_block_indices=cfg.get('moe_block_indices', None),
        datasets=cfg.get('datasets', ('query','sat'))
    ) 