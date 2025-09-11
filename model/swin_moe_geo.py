# -*- coding:utf8 -*-
"""
@file swin_moe_multiinput.py
@desc 独立实现的SwinTransformer_MoE_MultiInput主干，支持多模态输入，参考SM3Det实现。
      用于DetGeo-main项目的多模态特征提取。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import math

# print("MultiheadAttention from:", torch.nn.MultiheadAttention)  # 注释掉重复输出

# ========== Patch Embedding ==========
class PatchEmbed(nn.Module):
    """
    @class PatchEmbed
    @desc 将输入图片分割为patch并投影为token
    """
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.norm(x)
        return x, (H, W)

# ========== 全局专家池 ==========
class GlobalExpertPool(nn.Module):
    """
    @class GlobalExpertPool
    @desc 全局专家池，所有MoE块共享同一组专家
    """
    def __init__(self, embed_dim, ffn_dim, num_experts=6):
        super().__init__()
        self.num_experts = num_experts
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        # 专家网络初始化
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

# ========== MoE FFN ==========
class MoEFFN(nn.Module):
    """
    @class MoEFFN
    @desc 简化版MoE FFN，使用全局专家池，top-k门控
    """
    def __init__(self, embed_dim, ffn_dim, expert_pool, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.expert_pool = expert_pool  # 使用外部专家池
        self.num_experts = expert_pool.num_experts
        self.top_k = top_k
        
        # 门控网络
        self.gate = nn.Linear(embed_dim, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
        # 优化门控网络初始化，确保专家均匀分布
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, return_gate=False, noise_std=0.1, return_entropy=True):
        """
        @function forward
        @desc 前向传播，门控选择top-k专家进行加权融合，支持返回门控分布和熵
        @param {Tensor} x - 输入特征 [N, C]
        @param {bool} return_gate - 是否返回门控分布
        @param {float} noise_std - 门控高斯噪声标准差
        @param {bool} return_entropy - 是否返回门控分布的熵
        @return {Tensor|tuple} 输出特征 [N, C] 或 (输出特征, topk_idx) 或 (输出特征, topk_idx, gate_entropy)
        """
        gate_logits = self.gate(x)  # [N, num_experts]
        
        # ====== 加入高斯噪声 ======
        if self.training and noise_std > 0:
            noise = torch.randn_like(gate_logits) * noise_std
            gate_logits = gate_logits + noise
            
        # ====== 计算门控分布熵 ======
        gate_probs = self.softmax(gate_logits)  # [N, num_experts]
        gate_entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()  # 标量
        
        # ====== top-k门控 ======
        topk_val, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # [N, top_k]
        gate_weights = self.softmax(topk_val)  # [N, top_k]
        
        # ====== 计算专家输出 ======
        expert_outputs = torch.stack([expert(x) for expert in self.expert_pool.experts], dim=0)  # [num_experts, N, C]
        
        # ====== 加权组合 ======
        out = 0
        N = x.size(0)
        device = x.device
        arange_indices = torch.arange(N, device=device)
        
        for k in range(self.top_k):
            idx = topk_idx[:, k]  # [N]
            selected = expert_outputs[idx, arange_indices]  # [N, C]
            out = out + gate_weights[:, k:k+1] * selected
            
        if return_gate and return_entropy:
            return out, topk_idx, gate_entropy
        elif return_gate:
            return out, topk_idx
        elif return_entropy:
            return out, gate_entropy
        else:
            return out

def window_partition(x, window_size):
    # x: [B, H, W, C]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows  # [num_windows*B, window_size*window_size, C]

def window_reverse(windows, window_size, H, W):
    # windows: [num_windows*B, window_size*window_size, C]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# ========== Swin Transformer Block ==========
class SwinBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, ffn_dim=384, use_moe=False, expert_pool=None, top_k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_moe = use_moe
        if use_moe:
            if expert_pool is None:
                raise ValueError("MoE block requires expert_pool to be provided")
            self.ffn = MoEFFN(embed_dim, ffn_dim, expert_pool, top_k=top_k)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, embed_dim)
            )

    def forward(self, x, hw_shape, return_gate=False):
        B, HW, C = x.shape
        H, W = hw_shape
        assert H * W == HW, f"[SwinBlock] H*W={H*W}不等于HW={HW}"
        x = x.view(B, H, W, C)
        identity = x
        x = self.norm1(x)
        windows = window_partition(x, self.window_size)  # [num_windows*B, ws*ws, C]
        attn_windows, _ = self.attn(windows, windows, windows)
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]
        x = identity + x
        identity2 = x
        x = self.norm2(x)
        x = x.view(B, H*W, C)
        assert x.shape[-1] == self.norm2.normalized_shape[0], f"[SwinBlock] x最后一维{ x.shape[-1] }不等于embed_dim={ self.norm2.normalized_shape[0] }"
        x = x.view(-1, C)  # 保证MoEFFN输入为二维[N, C]
        if self.use_moe:
            out = self.ffn(x, return_gate=return_gate, return_entropy=True)
            if return_gate:
                if isinstance(out, tuple) and len(out) >= 3:
                    x, topk_idx, entropy = out[0], out[1], out[2]
                    # 存储熵值到backbone
                    if hasattr(self, '_store_entropy'):
                        self._store_entropy(entropy)
                    x = x.view(B, H*W, C)
                    x = identity2.view(B, H*W, C) + x
                    x = x.view(B, H, W, C)
                    return x.view(B, H*W, C), (H, W), topk_idx
                else:
                    # fallback
                    x = out[0] if isinstance(out, tuple) else out
                    entropy = out[2] if isinstance(out, tuple) and len(out) >= 3 else torch.tensor(0.0, device=x.device)
                    if hasattr(self, '_store_entropy'):
                        self._store_entropy(entropy)
                    x = x.view(B, H*W, C)
                    x = identity2.view(B, H*W, C) + x
                    x = x.view(B, H, W, C)
                    return x.view(B, H*W, C), (H, W), None
            else:
                # 训练时：return_gate=False, return_entropy=True
                # 返回值是 (out, gate_entropy) - 2个值
                if isinstance(out, tuple) and len(out) == 2:
                    x, entropy = out[0], out[1]  # out[0]是输出，out[1]是熵值
                    # 存储熵值到backbone
                    if hasattr(self, '_store_entropy'):
                        self._store_entropy(entropy)
                elif isinstance(out, tuple) and len(out) >= 3:
                    # 兼容性处理：如果返回3个值
                    x, entropy = out[0], out[2]
                    if hasattr(self, '_store_entropy'):
                        self._store_entropy(entropy)
                else:
                    # 兜底处理
                    x = out
        else:
            x = self.ffn(x)
        x = x.view(B, H*W, C)
        x = identity2.view(B, H*W, C) + x
        x = x.view(B, H, W, C)
        return x.view(B, H*W, C), (H, W)

# ========== Swin Stage ==========
class SwinStage(nn.Module):
    def __init__(self, depth, embed_dim, num_heads, window_size, ffn_dim, num_experts=4, top_k=2, moe_block_indices=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # 如果这个stage有MoE块，创建专家池
        self.expert_pool = None
        self.moe_block_indices = moe_block_indices  # 保存配置
        if moe_block_indices is not None and len(moe_block_indices) > 0:
            self.expert_pool = GlobalExpertPool(embed_dim, ffn_dim, num_experts)
            print(f"[DEBUG] SwinStage: 创建专家池, embed_dim={embed_dim}, ffn_dim={ffn_dim}, num_experts={num_experts}")
            print(f"[DEBUG] SwinStage: moe_block_indices={moe_block_indices}")
        else:
            print(f"[DEBUG] SwinStage: 无MoE块, moe_block_indices={moe_block_indices}")
        
        for i in range(depth):
            use_moe = (moe_block_indices is not None and i in moe_block_indices)
            block = SwinBlock(embed_dim, num_heads, window_size, ffn_dim, use_moe=use_moe, expert_pool=self.expert_pool, top_k=top_k)
            self.blocks.append(block)
            if use_moe:
                print(f"[DEBUG] Block {i}: MoE=True, expert_pool={block.ffn.expert_pool is not None if hasattr(block, 'ffn') else 'N/A'}")
            else:
                print(f"[DEBUG] Block {i}: MoE=False")

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            x, hw_shape = blk(x, hw_shape)
        return x, hw_shape

# ========== Patch Merging ==========
class PatchMerging(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)
    def forward(self, x, hw):
        # x: [B, HW, C], hw: (H, W)
        B, HW, C = x.shape
        H, W = hw
        x = x.view(B, H, W, C)
        # 2x2邻域合并
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H//2, W//2, 4C]
        x = x.view(B, -1, 4 * C)  # [B, H//2*W//2, 4C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, H//2*W//2, 2C]
        return x, (H // 2, W // 2)

# ========== 主干 ==========
class SwinTransformer_MoE_MultiInput(nn.Module):
    """
    @class SwinTransformer_MoE_MultiInput
    @desc 多模态输入Swin+MoE主干，支持query/sat等多模态，输出融合特征
    """
    def __init__(self,
                 in_channels=4,  # 仅query分支用
                 embed_dim=96,
                 patch_size=4,
                 window_size=8,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 ffn_ratio=4,
                 num_experts=6,  # 固定为6个专家
                 top_k=2,        # 固定使用2个专家
                 moe_block_indices=None,
                 datasets=('query', 'sat')):
        super().__init__()
        self.datasets = datasets
        self.num_experts = num_experts
        self.top_k = top_k
        
        # --- query分支4通道，sat分支3通道 ---
        self.patch_embeds = nn.ModuleDict({
            'query': PatchEmbed(4, embed_dim, patch_size),
            'sat': PatchEmbed(3, embed_dim, patch_size)
        })
        
        # 不再创建全局专家池，让每个stage创建自己的专家池
        print(f"[DEBUG] MoE配置: num_experts={num_experts}, top_k={top_k}")
        print(f"[DEBUG] MoE块分布: {moe_block_indices}")
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur_dim = embed_dim
        # 初始化熵值存储
        self._last_entropy = None
        self._entropy_list = []
        
        for i, depth in enumerate(depths):
            # 读取每个stage的moe_block_indices
            stage_moe_blocks = None
            if moe_block_indices is not None and i in moe_block_indices:
                stage_moe_blocks = moe_block_indices[i]
                print(f"[DEBUG] Stage {i}: 找到MoE配置: {stage_moe_blocks}")
            else:
                print(f"[DEBUG] Stage {i}: 无MoE配置, moe_block_indices={moe_block_indices}, i={i}")
            
            # 每个stage创建自己的专家池，但共享专家数量
            stage = SwinStage(
                depth=depth,
                embed_dim=cur_dim,
                num_heads=num_heads[i],
                window_size=window_size,
                ffn_dim=cur_dim*ffn_ratio,
                num_experts=num_experts,
                top_k=top_k,
                moe_block_indices=stage_moe_blocks
            )
            print(f"[DEBUG] Stage {i}: depth={depth}, embed_dim={cur_dim}, moe_blocks={stage_moe_blocks}, expert_pool={stage.expert_pool is not None}")
            self.stages.append(stage)
            
            # 为每个MoE块设置熵值存储回调
            if stage_moe_blocks:
                for block_idx in stage_moe_blocks:
                    if block_idx < len(stage.blocks):
                        stage.blocks[block_idx]._store_entropy = self._store_entropy
            
            if i < len(depths) - 1:
                self.downsamples.append(PatchMerging(cur_dim))
                cur_dim = cur_dim * 2
            else:
                self.downsamples.append(None)
        self.out_dim = cur_dim

    def forward(self, x_list, datasets=None):
        """
        @function forward
        @desc 多模态输入，返回query全局向量、sat空间特征和MoE熵值
        @param {list[Tensor]} x_list - 各模态图片 [B, 3, H, W]
        @param {list[str]} datasets - 对应模态名
        @return {tuple} (query_vec, sat_feat, avg_entropy)
        """
        if datasets is None:
            datasets = self.datasets
        # 假设datasets顺序 ['query', 'sat']
        query_x, query_hw = self.patch_embeds['query'](x_list[0])  # [B, HW1, C]
        sat_x, sat_hw = self.patch_embeds['sat'](x_list[1])      # [B, HW2, C]
        
        # 收集所有MoE块的熵值
        all_entropies = []
        
        for i, stage in enumerate(self.stages):
            query_x, query_hw = stage(query_x, query_hw)
            sat_x, sat_hw = stage(sat_x, sat_hw)
            if i < len(self.stages) - 1:
                query_x, query_hw = self.downsamples[i](query_x, query_hw)
                sat_x, sat_hw = self.downsamples[i](sat_x, sat_hw)
        
        # query分支全局池化
        B, HW1, C = query_x.shape
        query_vec = query_x.mean(dim=1)  # [B, C]
        # sat分支reshape为[B, C, H, W]
        B, HW2, C = sat_x.shape
        H2, W2 = sat_hw
        sat_feat = sat_x.transpose(1,2).reshape(B, C, H2, W2)
        
        # 计算平均MoE熵值（如果没有MoE块，返回0）
        avg_entropy = torch.tensor(0.0, device=query_vec.device)
        if hasattr(self, '_last_entropy') and self._last_entropy is not None:
            avg_entropy = self._last_entropy
        
        return query_vec, sat_feat, avg_entropy
    
    def _store_entropy(self, entropy):
        """
        @function _store_entropy
        @desc 存储MoE块的熵值
        @param {Tensor} entropy - 熵值
        """
        if entropy is not None:
            # 多卡环境下，确保熵值在主设备上
            if hasattr(self, '_main_device'):
                entropy = entropy.to(self._main_device)
            else:
                # 如果没有设置主设备，使用第一个参数所在的设备
                self._main_device = next(self.parameters()).device
                entropy = entropy.to(self._main_device)
            
            self._entropy_list.append(entropy.detach())
            # 只保留最近的几个熵值
            if len(self._entropy_list) > 10:
                self._entropy_list.pop(0)
            # 计算平均熵值
            if self._entropy_list:
                self._last_entropy = torch.stack(self._entropy_list).mean()
    
    def get_moe_entropy(self):
        """
        @function get_moe_entropy
        @desc 获取MoE平均熵值
        @return {Tensor} 平均熵值
        """
        if self._last_entropy is not None:
            return self._last_entropy
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward_with_gate(self, x_list, datasets=None):
        """
        @function forward_with_gate
        @desc 多模态输入，返回特征和所有MoE门控分布，便于可视化专家激活
        @param {list[Tensor]} x_list - 各模态图片 [B, 3, H, W]
        @param {list[str]} datasets - 对应模态名
        @return {dict} { 'query_vec':..., 'sat_feat':..., 'moe_gates': {stage_idx: {'query':[topk_idx...], 'sat':[topk_idx...]}} }
        """
        if datasets is None:
            datasets = self.datasets
        query_x, query_hw = self.patch_embeds['query'](x_list[0])
        sat_x, sat_hw = self.patch_embeds['sat'](x_list[1])
        moe_gates = {}
        for i, stage in enumerate(self.stages):
            stage_query_gates = []
            stage_sat_gates = []
            for block in stage.blocks:
                if hasattr(block, 'use_moe') and block.use_moe:
                    query_x, query_hw, query_gate = block(query_x, query_hw, return_gate=True)
                    sat_x, sat_hw, sat_gate = block(sat_x, sat_hw, return_gate=True)
                    stage_query_gates.append(query_gate.detach().cpu())
                    stage_sat_gates.append(sat_gate.detach().cpu())
                else:
                    query_x, query_hw = block(query_x, query_hw)
                    sat_x, sat_hw = block(sat_x, sat_hw)
            if stage_query_gates or stage_sat_gates:
                moe_gates[i] = {'query': stage_query_gates, 'sat': stage_sat_gates}
            if i < len(self.stages) - 1:
                query_x, query_hw = self.downsamples[i](query_x, query_hw)
                sat_x, sat_hw = self.downsamples[i](sat_x, sat_hw)
        B, HW1, C = query_x.shape
        query_vec = query_x.mean(dim=1)
        B, HW2, C = sat_x.shape
        H2, W2 = sat_hw
        sat_feat = sat_x.transpose(1,2).reshape(B, C, H2, W2)
        return {'query_vec': query_vec, 'sat_feat': sat_feat, 'moe_gates': moe_gates}

def initialize_moe_experts_from_ffn(moe_ffn, ffn):
    """
    @function initialize_moe_experts_from_ffn
    @desc 用主干FFN权重初始化MoE所有专家权重
    @param moe_ffn: MoEFFN实例
    @param ffn: nn.Sequential类型的普通FFN（Linear-GELU-Linear）
    """
    for expert in moe_ffn.experts:
        for e_layer, f_layer in zip(expert, ffn):
            if isinstance(e_layer, nn.Linear) and isinstance(f_layer, nn.Linear):
                e_layer.weight.data.copy_(f_layer.weight.data)
                if e_layer.bias is not None and f_layer.bias is not None:
                    e_layer.bias.data.copy_(f_layer.bias.data)
class CrossViewFusionModule(nn.Module):
    """
    @class CrossViewFusionModule
    @desc 查询全局特征引导卫星空间特征的点积注意力融合
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_sat = nn.LayerNorm(embed_dim)
    def forward(self, query_vec, sat_feat):
        # query_vec: [B, C], sat_feat: [B, C, H, W]
        B, C, H, W = sat_feat.shape
        # L2归一化
        q = F.normalize(self.norm_query(query_vec), dim=-1)  # [B, C]
        s = F.normalize(self.norm_sat(sat_feat.permute(0,2,3,1)), dim=-1)  # [B, H, W, C]
        # 点积注意力
        attn = (s * q.unsqueeze(1).unsqueeze(1)).sum(-1)  # [B, H, W]
        attn = attn.view(B, -1)
        attn = F.softmax(attn, dim=-1).view(B, H, W)  # [B, H, W]
        # 加权融合
        fused = sat_feat * attn.unsqueeze(1)  # [B, C, H, W]
        return fused, attn

