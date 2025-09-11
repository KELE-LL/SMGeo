# -*- coding:utf8 -*-
"""
@file swin_moe_geo_config.py
@desc Swin-MoE多输入主干及DetGeo-main项目的统一配置文件。
      便于结构、参数、权重、实验等集中管理和修改。
      
      重要说明：
      - moe_block_indices: 用于灵活指定每个Stage哪些Block用MoE-FFN替代普通FFN。
        例如：
        moe_block_indices = {
            0: [],           # 第1个stage全部用普通FFN
            1: [0,2],       # 第2个stage第1/3个Block用MoE
            2: [0,2,4,6,8], # 第3个stage第1/3/5/7/9个Block用MoE
            3: [0,2]        # 第4个stage第1/3个Block用MoE
        }
      - 下标均从0开始，未列出的stage默认全部Block用普通FFN。
"""
import os
swin_moe_geo_cfg = {
    # -------- Swin Transformer结构参数 --------
    'in_channels': 4,
    'embed_dim': 96,
    'patch_size': 4,
    'window_size': 8,
    'depths': (2, 2, 6, 2),
    'num_heads': (3, 6, 12, 24),
    'ffn_ratio': 4,
    'datasets': ('query', 'sat'),  # 支持的模态名

    # -------- MoE相关参数 --------
    'num_experts': 6,   # 适中的专家数量，平衡表达能力和计算效率
    'top_k': 2,         # 适中的top-k，避免过度激活
    # 'use_moe_stages': (2,4,),  # 已废弃，无需再用
    'moe_block_indices': {
        0: [],               # 第1个stage：全部用普通FFN ✅ 早期特征学习用普通FFN更稳定
        1: [0,1],              # 第2个stage：第1个Block用MoE ✅ 开始引入MoE，但保守使用
        2: [0, 2, 4],       # 第3个stage：第1/3/5个Block用MoE ✅ 核心层适度使用MoE
        3: [0, 1]           # 第4个stage：2个Block全部用MoE ✅ 后期层大量使用MoE，发挥专家优势
    },

    # -------- 预训练权重 --------
    'pretrained': './data/pretrained/swin_tiny_patch4_window7_224.pth',  # Swin预训练权重路径

    # -------- 训练/数据相关参数 --------
    'batch_size': 8,
    'img_size': 1024,
    'max_epoch': 25,
    'data_root': './data',
    'data_name': 'CVOGL_DroneAerial',
    'num_workers': 4,
    'lr': 3e-4,  # 初始学习率，已与训练脚本同步
    'beta': 1.0,
    'emb_size': 512,
    'gpu': '0',

    # -------- 其他可选参数 --------
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'use_abs_pos_embed': False,
    'act_cfg': {'type': 'GELU'},
    'norm_cfg': {'type': 'LN'},
    'with_cp': False,
    'convert_weights': False,
    'frozen_stages': -1,
    'init_cfg': None,
    # ...可继续扩展
}
