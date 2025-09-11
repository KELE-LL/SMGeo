#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具（只支持增强版训练脚本保存的权重！）
====================================================
【重要说明】
本脚本只允许加载由 enhanced_training.py 训练并保存的权重文件。
所有模型结构、参数、数据流必须与训练时完全一致，
禁止在本脚本中自定义或更改模型结构、参数或数据流。
如检测到权重文件不合规或参数不一致，将直接报错并提示用户重新训练！
====================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import ast

from dataset.data_loader import RSDataset
from model.swin_moe_geo import SwinTransformer_MoE_MultiInput
from model.anchorfree_head import AnchorFreeHead
from model.swin_moe_geo_config import swin_moe_geo_cfg
from model.loss import build_target_anchorfree, anchorfree_loss
from visualization_core import draw_visualization, to_pixel_box, to_pixel_point

# 全局定义custom_collate_fn函数，避免多进程pickle问题
def custom_collate_fn(batch):
    """自定义数据批处理函数"""
    queryimg_4ch, rsimg, bbox, idx, click_xy, ori_hw = zip(*batch)
    queryimg_4ch = torch.stack(queryimg_4ch)
    rsimg = torch.stack(rsimg)
    # 关键修正：保证每个bbox都是1维4元素tensor
    bbox = [b.view(-1) if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float32).view(-1) for b in bbox]
    bbox = torch.stack(bbox)
    idx = torch.tensor(idx)
    click_xy = torch.stack(click_xy)
    ori_hw = torch.stack(ori_hw)
    return queryimg_4ch, rsimg, bbox, idx, click_xy, ori_hw

class ModelVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = f'./visualization_outputs/{args.savename}'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self, checkpoint_path):
        """
        加载模型（只支持增强版训练脚本保存的权重！）
        """
        from model.swin_moe_geo import SwinTransformer_MoE_MultiInput
        from model.anchorfree_head import AnchorFreeHead
        from model.swin_moe_geo_config import swin_moe_geo_cfg
        swin_cfg = swin_moe_geo_cfg
        # 只保留SwinTransformer_MoE_MultiInput支持的参数
        backbone_kwargs = dict(
            in_channels=swin_cfg.get('in_channels', 4),
            embed_dim=swin_cfg.get('embed_dim', 96),
            patch_size=swin_cfg.get('patch_size', 4),
            window_size=swin_cfg.get('window_size', 8),
            depths=swin_cfg.get('depths', (2,2,6,2)),
            num_heads=swin_cfg.get('num_heads', (3,6,12,24)),
            ffn_ratio=swin_cfg.get('ffn_ratio', 4),
            num_experts=swin_cfg.get('num_experts', 6),
            top_k=swin_cfg.get('top_k', 2),
            moe_block_indices=swin_cfg.get('moe_block_indices', None),
            datasets=swin_cfg.get('datasets', ('query','sat'))
        )
        swin_backbone = SwinTransformer_MoE_MultiInput(**backbone_kwargs)
        out_dim = swin_backbone.out_dim
        anchorfree_head = AnchorFreeHead(in_channels=out_dim, feat_channels=256, num_classes=1)
        class DetGeoSwinMoE_AF(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
            def forward(self, query_img, sat_img, click_map=None):
                x_list = [query_img, sat_img]
                # SwinTransformer_MoE_MultiInput返回3个值：(query_vec, sat_feat, avg_entropy)
                query_vec, sat_feat, avg_entropy = self.backbone(x_list)
                heatmap, bbox = self.head(sat_feat)
                return heatmap, bbox
        model = DetGeoSwinMoE_AF(swin_backbone, anchorfree_head)
        model = torch.nn.DataParallel(model).to(self.device)
        
        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # 自动对比config
            if 'swin_cfg' in checkpoint:
                mismatch = []
                for k in swin_moe_geo_cfg:
                    if k in checkpoint['swin_cfg']:
                        if swin_moe_geo_cfg[k] != checkpoint['swin_cfg'][k]:
                            mismatch.append(f"{k}: 当前={swin_moe_geo_cfg[k]}，权重={checkpoint['swin_cfg'][k]}")
                if mismatch:
                    print("❌ 配置文件与权重文件参数不一致，以下参数不同：")
                    for line in mismatch:
                        print("   ", line)
                    print("====================================================")
                    print("本可视化脚本只支持增强版训练脚本保存的权重！\n请确保训练和可视化用的配置文件完全一致，否则请重新训练！")
                    print("====================================================")
                    raise ValueError("配置文件与权重文件参数不一致，请确保完全一致后再可视化！")
                else:
                    print("✓ 当前配置与权重文件参数完全一致。\n【可视化脚本已严格依托增强版训练权重，禁止另起炉灶！】")
            else:
                print("====================================================")
                print("⚠ 权重文件未保存swin_cfg，无法自动对比参数。\n本脚本只支持增强版训练脚本保存的权重！\n请用enhanced_training.py重新训练并保存权重！")
                print("====================================================")
                raise ValueError("权重文件不合规，请用增强版训练脚本重新训练！")
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✓ 成功加载模型权重: {checkpoint_path}")
            print(f"  训练轮次: {checkpoint.get('epoch', 'Unknown')}")
            print(f"  最佳性能: {checkpoint.get('accu', 'Unknown'):.4f}")
        else:
            print("⚠ 未找到权重文件，使用随机初始化的模型\n【警告：此模式仅供调试，不能用于正式可视化！】")
        
        return model
    
    def load_data(self):
        """加载数据"""
        input_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = RSDataset(
            data_root=self.args.data_root, 
            data_name=self.args.data_name,
            split_name='val', 
            img_size=self.args.img_size, 
            transform=input_transform
        )
        
        # Windows系统下禁用多进程，避免pickle问题
        num_workers = 0 if os.name == 'nt' else self.args.num_workers
        
        data_loader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            pin_memory=True, 
            drop_last=False, 
            num_workers=num_workers,
            collate_fn=custom_collate_fn
        )
        return data_loader
    
    def visualize_model_outputs(self, model, data_loader, num_samples=5):
        """只针对swinmoe分支的可视化，所有点和框严格做映射，所有子图尺寸一致"""
        print(f"\n=== 开始可视化模型输出 (样本数: {num_samples}) ===")
        model.eval()
        sample_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) > 6:
                    batch = batch[:6]
                if len(batch) == 6:
                    query_imgs, rs_imgs, bbox, idx, click_xy, ori_img_shape = batch
                else:
                    raise ValueError(f"数据batch结构不支持，长度为{len(batch)}")
                query_imgs, rs_imgs = query_imgs.to(self.device), rs_imgs.to(self.device)
                bbox = bbox.to(self.device)
                B, _, H, W = query_imgs.shape
                for i in range(B):
                    if sample_count >= num_samples:
                        break
                    qimg = self.denormalize_image(query_imgs[i])
                    simg = self.denormalize_image(rs_imgs[i])
                    # 处理click_xy（多点兼容，像素坐标→归一化→可视化窗口）
                    raw_click_xy = click_xy[i].cpu().numpy() if hasattr(click_xy[i], 'cpu') else click_xy[i]
                    click_xy_arr = np.array(raw_click_xy)
                    if click_xy_arr.ndim == 2 and click_xy_arr.shape[0] > 0:
                        click_xy_main = click_xy_arr[0]
                        all_click_points = click_xy_arr
                    else:
                        click_xy_main = click_xy_arr
                        all_click_points = None
                    click_x, click_y = float(click_xy_main[0]), float(click_xy_main[1])
                    # 获取原始query图像尺寸
                    def to_scalar(x):
                        if hasattr(x, 'item'):
                            return int(x.item())
                        return int(x)
                    ori_H, ori_W = ori_img_shape[i] if isinstance(ori_img_shape, (list, tuple, np.ndarray, torch.Tensor)) else ori_img_shape
                    ori_H = to_scalar(ori_H)
                    ori_W = to_scalar(ori_W)
                    img_H, img_W = simg.shape[:2]

                    # 真实框做原图到当前图片的坐标映射
                    gt_box_pixel = bbox[i].cpu().numpy() if hasattr(bbox[i], 'cpu') else bbox[i]
                    gt_box_pixel = np.array(gt_box_pixel, dtype=np.float32)
                    # 做一次线性缩放（坐标映射）
                    scale_x = img_W / ori_W
                    scale_y = img_H / ori_H
                    x1 = gt_box_pixel[0] * scale_x
                    y1 = gt_box_pixel[1] * scale_y
                    x2 = gt_box_pixel[2] * scale_x
                    y2 = gt_box_pixel[3] * scale_y
                    # clip到图片范围
                    x1 = np.clip(x1, 0, img_W - 1)
                    x2 = np.clip(x2, 0, img_W - 1)
                    y1 = np.clip(y1, 0, img_H - 1)
                    y2 = np.clip(y2, 0, img_H - 1)
                    gt_box_pixel = [x1, y1, x2, y2]
                    gt_center_pixel = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # 获取特征图尺寸（用于anchor-free解码和过滤）
                    heatmap_pred, bbox_pred = model(query_imgs[i:i+1], rs_imgs[i:i+1])
                    pred_hm = heatmap_pred[0, 0].sigmoid().cpu().numpy()
                    _, _, hH, hW = heatmap_pred.shape
                    gt_heatmap, gt_bbox, mask = build_target_anchorfree(
                        bbox[i:i+1], hH, hW, self.args.img_size, self.args.img_size)
                    gt_hm = gt_heatmap[0, 0].cpu().numpy()
                    pred_center = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
                    gt_center = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
                    pred_box_params = bbox_pred[0, :, pred_center[0], pred_center[1]].cpu().numpy()
                    gt_box_params = gt_bbox[0, :, gt_center[0], gt_center[1]].cpu().numpy()
                    bbox_pred_values = bbox_pred[0].cpu().numpy().flatten()
                    feat_H, feat_W = pred_hm.shape

                    # 2. 预测框和中心点 anchor-free 解码
                    def box_params_to_pixel(box_params, center, img_W, img_H, feat_W, feat_H):
                        ny, nx = center  # (y, x)
                        cx_nx, cy_ny, w, h = box_params
                        w = np.clip(w, 1e-2, feat_W)
                        h = np.clip(h, 1e-2, feat_H)
                        cx = nx + cx_nx
                        cy = ny + cy_ny
                        scale_x = img_W / feat_W
                        scale_y = img_H / feat_H
                        cx_img = (cx + 0.5) * scale_x
                        cy_img = (cy + 0.5) * scale_y
                        w_img = w * scale_x
                        h_img = h * scale_y
                        x1 = cx_img - w_img / 2
                        y1 = cy_img - h_img / 2
                        x2 = cx_img + w_img / 2
                        y2 = cy_img + h_img / 2
                        x1 = np.clip(x1, 0, img_W)
                        y1 = np.clip(y1, 0, img_H)
                        x2 = np.clip(x2, 0, img_W)
                        y2 = np.clip(y2, 0, img_H)
                        return [x1, y1, x2, y2]
                    def center_feat2pixel(center, img_H, img_W, feat_H, feat_W):
                        y, x = center
                        x_pixel = (x + 0.5) * img_W / feat_W
                        y_pixel = (y + 0.5) * img_H / feat_H
                        return (x_pixel, y_pixel)
                    pred_center_pixel = center_feat2pixel(pred_center, ori_H, ori_W, feat_H, feat_W) if pred_center is not None else None
                    pred_box_pixel = box_params_to_pixel(pred_box_params, pred_center, ori_W, ori_H, feat_W, feat_H) if pred_box_params is not None else None
                    # 缩放到可视化窗口
                    pred_center_pixel_vis = (pred_center_pixel[0] * scale_x, pred_center_pixel[1] * scale_y) if pred_center_pixel is not None else None
                    pred_box_pixel_vis = [
                        pred_box_pixel[0] * scale_x,
                        pred_box_pixel[1] * scale_y,
                        pred_box_pixel[2] * scale_x,
                        pred_box_pixel[3] * scale_y
                    ] if pred_box_pixel is not None else None

                    # 查询图像点击点缩放到可视化尺寸（兼容batch结构，确保为int）
                    ori_query_H, ori_query_W = ori_img_shape
                    if isinstance(ori_query_H, (list, tuple, torch.Tensor, np.ndarray)):
                        ori_query_H = ori_query_H[i]
                    if isinstance(ori_query_W, (list, tuple, torch.Tensor, np.ndarray)):
                        ori_query_W = ori_query_W[i]
                    ori_query_H = int(ori_query_H)
                    ori_query_W = int(ori_query_W)
                    img_H, img_W = qimg.shape[:2]
                    click_x, click_y = click_xy[i] if isinstance(click_xy, (list, tuple, np.ndarray)) else click_xy
                    ori_query_W = max(ori_query_W, 1)  # 避免除零
                    ori_query_H = max(ori_query_H, 1)
                    scale_x = img_W / ori_query_W
                    scale_y = img_H / ori_query_H
                    clickxy_pixel = (click_x * scale_x, click_y * scale_y)
                    # 确保点击区域坐标在图片范围内
                    clickxy_pixel = (
                        np.clip(clickxy_pixel[0], 0, img_W - 1),
                        np.clip(clickxy_pixel[1], 0, img_H - 1)
                    )

                    # 1. 图片resize到可视化分辨率
                    VIS_H, VIS_W = 256, 256  # 可视化窗口分辨率
                    qimg_vis = cv2.resize(qimg, (VIS_W, VIS_H))
                    simg_vis = cv2.resize(simg, (VIS_W, VIS_H))

                    # 2. bbox坐标缩放
                    scale_x = VIS_W / ori_W
                    scale_y = VIS_H / ori_H
                    x1 = gt_box_pixel[0] * scale_x
                    y1 = gt_box_pixel[1] * scale_y
                    x2 = gt_box_pixel[2] * scale_x
                    y2 = gt_box_pixel[3] * scale_y
                    gt_box_pixel_vis = [x1, y1, x2, y2]
                    gt_center_pixel_vis = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # 2.5. 中心点坐标缩放
                    if pred_center_pixel_vis is not None:
                        pred_center_pixel_vis = (pred_center_pixel_vis[0] * scale_x, pred_center_pixel_vis[1] * scale_y)
                    else:
                        pred_center_pixel_vis = None
                    if gt_center_pixel is not None:
                        gt_center_pixel_vis = (gt_center_pixel[0] * scale_x, gt_center_pixel[1] * scale_y)
                    else:
                        gt_center_pixel_vis = None

                    # 3. 点击点坐标缩放（主推理输入严格与训练一致，仅展示层兼容多点）
                    raw_click_xy = click_xy[i].cpu().numpy() if hasattr(click_xy[i], 'cpu') else click_xy[i]
                    click_xy_arr = np.array(raw_click_xy)
                    # 主推理输入：只用第一个点，保证与训练一致
                    if click_xy_arr.ndim == 2 and click_xy_arr.shape[0] > 0:
                        click_xy_main = click_xy_arr[0]
                        all_click_points = click_xy_arr
                    else:
                        click_xy_main = click_xy_arr
                        all_click_points = None
                    click_x, click_y = float(click_xy_main[0]), float(click_xy_main[1])
                    click_x_vis = click_x * scale_x
                    click_y_vis = click_y * scale_y
                    clickxy_pixel_vis = (float(np.clip(click_x_vis, 0, VIS_W - 1)), float(np.clip(click_y_vis, 0, VIS_H - 1)))

                    # 2.6. 预测框坐标缩放
                    if pred_box_pixel_vis is not None:
                        pred_box_pixel_vis = [
                            pred_box_pixel_vis[0] * scale_x,
                            pred_box_pixel_vis[1] * scale_y,
                            pred_box_pixel_vis[2] * scale_x,
                            pred_box_pixel_vis[3] * scale_y
                        ]
                    else:
                        pred_box_pixel_vis = None

                    # 保证mat_clickxy为标准1D float32向量
                    mat_clickxy = np.array(clickxy_pixel_vis, dtype=np.float32).reshape(-1)
                    if mat_clickxy.shape != (2,):
                        mat_clickxy = mat_clickxy.flatten()[:2]

                    # 传递原始像素点击点和原图尺寸到extra_info
                    extra_info = {
                        "all_click_points": all_click_points,
                        "click_xy_pixel": (click_x, click_y),
                        "ori_query_W": ori_W,
                        "ori_query_H": ori_H
                    }
                    # 4. 传入draw_visualization，img_size严格用(VIS_H, VIS_W)
                    draw_visualization(
                        qimg_vis, simg_vis, mat_clickxy,
                        pred_hm, gt_hm,
                        pred_box_pixel_vis, gt_box_pixel_vis,
                        pred_center_pixel_vis, gt_center_pixel_vis,
                        bbox_pred_values,
                        self.output_dir, sample_count+1,
                        (VIS_H, VIS_W),
                        (feat_H, feat_W),
                        extra_info=extra_info
                    )
                    sample_count += 1
    
    def plot_output_distributions(self, ax, heatmap_pred, bbox_pred):
        """绘制输出分布"""
        # 热力图分布
        hm_flat = heatmap_pred.sigmoid().flatten().cpu().numpy()
        bbox_flat = bbox_pred.flatten().cpu().numpy()
        
        # 创建子图
        ax2 = ax.twinx()
        
        # 热力图分布（蓝色）
        ax.hist(hm_flat, bins=50, alpha=0.7, color='blue', label='Heatmap Values')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Heatmap Frequency', color='blue', fontsize=10)
        ax.tick_params(axis='y', labelcolor='blue')
        
        # bbox分布（红色）
        ax2.hist(bbox_flat, bins=50, alpha=0.7, color='red', label='BBox Values')
        ax2.set_ylabel('BBox Frequency', color='red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title('Output Distributions', fontsize=12, fontweight='bold')
        
        # 添加统计信息
        hm_mean, hm_std = hm_flat.mean(), hm_flat.std()
        bbox_mean, bbox_std = bbox_flat.mean(), bbox_flat.std()
        
        stats_text = f'Heatmap: μ={hm_mean:.3f}, σ={hm_std:.3f}\nBBox: μ={bbox_mean:.3f}, σ={bbox_std:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def denormalize_image(self, img_tensor):
        """反归一化图像，兼容4通道（只还原RGB）"""
        # 只取前三个通道
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3, ...]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        return img_denorm.cpu().permute(1, 2, 0).numpy()
    
    def save_sample_details(self, sample_id, details):
        """保存样本详细信息"""
        details_file = os.path.join(self.output_dir, f'sample_{sample_id}_details.txt')
        with open(details_file, 'w', encoding='utf-8') as f:
            f.write(f"样本 {sample_id} 详细信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"真实边界框: {details['gt_box']}\n")
            f.write(f"预测边界框: {details['pred_box']}\n")
            f.write(f"真实中心点: {details['gt_center']}\n")
            f.write(f"预测中心点: {details['pred_center']}\n")
            f.write(f"预测置信度: {details['pred_confidence']:.4f}\n")
            f.write(f"真实置信度: {details['gt_confidence']:.4f}\n")
    
    def print_coord_info(self, name, value, img_W, img_H):
        arr = np.asarray(value).flatten()
        print(f"[DEBUG] {name}: {arr}")
        if np.all((arr >= 0) & (arr <= 1)):
            print(f"  → {name} 可能是归一化坐标 (0~1)")
            print(f"  → 映射到像素: ({arr[0]*img_W:.1f}, {arr[1]*img_H:.1f})")
        elif np.all((arr >= 0) & (arr <= max(img_W, img_H))):
            print(f"  → {name} 可能是像素坐标")
        else:
            print(f"  → {name} 可能是特征图坐标或其它类型")

    def smart_map_point(self, point, img_W, img_H, feat_W=32, feat_H=32):
        arr = np.asarray(point).flatten()
        if np.all((arr >= 0) & (arr <= 1)):
            # 归一化
            return arr[0] * img_W, arr[1] * img_H
        elif np.all((arr >= 0) & (arr <= max(img_W, img_H))):
            # 像素
            return arr[0], arr[1]
        else:
            # 特征图
            scale_x = img_W / feat_W
            scale_y = img_H / feat_H
            return (arr[0] + 0.5) * scale_x, (arr[1] + 0.5) * scale_y

# === 坐标缩放工具函数 ===
def map_box_to_vis(box, ori_W, ori_H, vis_W, vis_H):
    """
    将原图像素坐标映射到可视化窗口坐标
    @param box: [x1, y1, x2, y2] 原图像素坐标
    @param ori_W: 原图宽度
    @param ori_H: 原图高度
    @param vis_W: 可视化窗口宽度
    @param vis_H: 可视化窗口高度
    @return: [x1, y1, x2, y2] 可视化窗口坐标
    """
    x1, y1, x2, y2 = box
    scale_x = vis_W / ori_W
    scale_y = vis_H / ori_H
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

def map_point_to_vis(point, ori_W, ori_H, vis_W, vis_H):
    """
    将原图像素点映射到可视化窗口坐标
    @param point: (x, y) 原图像素坐标
    @return: (x, y) 可视化窗口坐标
    """
    x, y = point
    scale_x = vis_W / ori_W
    scale_y = vis_H / ori_H
    return (x * scale_x, y * scale_y)

def batch_auto_tune_visualization(records_file="auto_tune_records.txt"):
    output_dir = "visualization_outputs/auto_tune/"
    os.makedirs(output_dir, exist_ok=True)
    # 读取实验记录
    with open(records_file, "r", encoding="utf-8") as f:
        records = [ast.literal_eval(line.strip()) for line in f if line.strip()]
    # 批量分析日志
    all_metrics = []
    for rec in records:
        log_path = rec["log"]
        if not os.path.exists(log_path):
            continue
        metrics = parse_log_metrics(log_path)
        all_metrics.append({**rec, **metrics})
        # 绘制loss曲线
        if "loss" in metrics:
            plt.figure()
            plt.plot(metrics["loss"], label="loss")
            plt.title(f"Loss Curve: {rec['savename']}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{rec['savename']}_loss.png"))
            plt.close()
    # 绘制参数对比指标
    plot_metric_compare(all_metrics, output_dir)

def parse_log_metrics(log_path):
    # 简单解析日志，提取loss和Accu50等
    metrics = {"loss": [], "Accu50": [], "Accu25": [], "Mean_iou": [], "Accu_c": []}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "loss:" in line:
                try:
                    loss = float(line.split("loss:")[-1].split()[0])
                    metrics["loss"].append(loss)
                except:
                    pass
            for k in ["Accu50", "Accu25", "Mean_iou", "Accu_c"]:
                if f"{k}:" in line:
                    try:
                        v = float(line.split(f"{k}:")[-1].split()[0])
                        metrics[k].append(v)
                    except:
                        pass
    return metrics

def plot_metric_compare(all_metrics, output_dir):
    # 按参数组合对比Accu50等
    for k in ["Accu50", "Accu25", "Mean_iou", "Accu_c"]:
        plt.figure()
        for rec in all_metrics:
            if rec[k]:
                plt.plot(rec[k], label=f"{rec['savename']}")
        plt.title(f"{k} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"compare_{k}.png"))
        plt.close()

def main():
    """
    主函数 - 直接运行可视化脚本
    权重文件：D:\PythonProject\Cross_View\saved_weights\debug_experiment\best_weights.pth
    输出目录：D:\PythonProject\Cross_View\visualization_outputs\visualization
    """
    print("=== 开始运行可视化脚本 ===")
    
    # 直接设置参数，无需命令行输入
    class Args:
        def __init__(self):
            self.gpu = '0'  # GPU ID
            self.num_workers = 4  # 数据加载的工作进程数
            self.batch_size = 2  # 批次大小
            self.img_size = 1024  # 图像尺寸
            self.data_root = './data'  # 数据集根目录
            self.data_name = 'CVOGL_DroneAerial'  # 数据集名称
            self.checkpoint = r'D:\PythonProject\Cross_View\saved_weights\debug_experiment\best_weights.pth'  # 权重文件路径
            self.savename = 'visualization'  # 输出目录名称
            self.num_samples = 5  # 可视化样本数量
            self.visualize_outputs = True  # 是否可视化模型输出
            self.model = 'swinmoe'  # 模型类型
    
    args = Args()
    
    # 检查权重文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误：权重文件不存在: {args.checkpoint}")
        print("请检查权重文件路径是否正确")
        return
    
    print(f"✓ 权重文件路径: {args.checkpoint}")
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 创建可视化器
    visualizer = ModelVisualizer(args)
    
    # 加载模型
    print("正在加载模型...")
    model = visualizer.load_model(args.checkpoint)
    
    # 加载数据
    print("正在加载数据...")
    data_loader = visualizer.load_data()
    
    # 执行可视化
    if args.visualize_outputs:
        print("开始可视化模型输出...")
        visualizer.visualize_model_outputs(model, data_loader, args.num_samples)
    
    print("=== 可视化完成 ===")
    print(f"结果已保存到: {visualizer.output_dir}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ 脚本运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查以下问题：")
        print("1. 权重文件是否存在")
        print("2. 数据集路径是否正确")
        print("3. 依赖模块是否已安装")
        print("4. GPU是否可用")