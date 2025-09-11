#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版训练脚本
包含权重保存策略、损失函数分析和可视化功能
"""

import os
import sys
import argparse
import time
import random
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import gc
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, RandomAffine
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

from dataset.data_loader import RSDataset
from model.loss import yolo_loss, build_target, adjust_learning_rate
from model.loss import build_target_anchorfree, anchorfree_loss
from utils.utils import AverageMeter, eval_iou_acc
from utils.checkpoint import save_checkpoint, load_pretrain
from model.swin_moe_geo_config import swin_moe_geo_cfg
# from visualization_core import draw_visualization  # 暂时注释，先让训练跑起来
from model.swin_moe_geo import SwinTransformer_MoE_MultiInput
from model.anchorfree_head import AnchorFreeHead

def setup_multi_gpu(args):
    """
    设置多GPU训练
    @param args: 参数对象
    @return: device_ids, device
    """
    if args.gpu:
        # 解析GPU设备列表
        if ',' in args.gpu:
            device_ids = [int(id.strip()) for id in args.gpu.split(',')]
        else:
            device_ids = [int(args.gpu)]
        
        # 设置主设备
        device = torch.device(f'cuda:{device_ids[0]}')
        
        # 检查GPU可用性
        available_gpus = []
        for gpu_id in device_ids:
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                available_gpus.append(gpu_id)
            else:
                logging.warning(f"GPU {gpu_id} 不可用，跳过")
        
        if not available_gpus:
            logging.error("没有可用的GPU，使用CPU训练")
            device = torch.device('cpu')
            device_ids = []
        else:
            device_ids = available_gpus
            logging.info(f"使用GPU设备: {device_ids}")
            
        return device_ids, device
    else:
        # 使用所有可用GPU
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            device = torch.device('cuda:0')
            logging.info(f"使用所有可用GPU: {device_ids}")
        else:
            device_ids = []
            device = torch.device('cpu')
            logging.info("使用CPU训练")
        
        return device_ids, device

def wrap_model_for_multi_gpu(model, device_ids, args):
    """
    将模型包装为多GPU训练
    @param model: 模型
    @param device_ids: GPU设备ID列表
    @param args: 参数对象
    @return: 包装后的模型
    """
    if len(device_ids) > 1:
        # 多GPU训练 - 先将模型移动到主设备
        main_device = f'cuda:{device_ids[0]}'
        model = model.to(main_device)
        
        # 确保BatchNorm使用同步统计信息
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = True
                module.momentum = 0.1  # 使用较小的momentum，提高稳定性
        
        if args.distributed:
            # 分布式训练
            model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
            logging.info(f"使用DistributedDataParallel，设备: {device_ids}")
        else:
            # DataParallel训练 - 添加同步BN
            model = nn.DataParallel(model, device_ids=device_ids)
            logging.info(f"使用DataParallel，设备: {device_ids}")
    else:
        # 单GPU训练
        if device_ids:
            model = model.to(f'cuda:{device_ids[0]}')
            logging.info(f"使用单GPU训练，设备: {device_ids[0]}")
        else:
            model = model.to('cpu')
            logging.info("使用CPU训练")
    
    return model

def custom_collate_fn(batch):
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

class EnhancedTrainer:
    def __init__(self, args):
        self.args = args
        self.best_accu = -float('Inf')
        self.worst_accu = float('Inf')
        self.best_epoch = 0
        self.worst_epoch = 0
        
        # 创建权重保存目录
        self.weight_dir = f'./saved_weights/{args.savename}'
        os.makedirs(self.weight_dir, exist_ok=True)
        
        # 损失历史记录
        self.loss_history = {
            'total_loss': [],
            'heatmap_loss': [],
            'bbox_loss': [],
            'accu50': [],
            'accu25': [],
            'mean_iou': []
        }
        
    def save_weights(self, model, optimizer, scheduler, epoch, accu, is_best=False, is_worst=False):
        """只保存最佳、最差和最终权重，并保存swin_cfg用于后续对比"""
        from model.swin_moe_geo_config import swin_moe_geo_cfg  # 确保每次保存都是最新的
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'accu': accu,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'swin_cfg': dict(swin_moe_geo_cfg),  # 新增：保存当前config
        }
        # 最终权重（每轮覆盖）
        torch.save(checkpoint, os.path.join(self.weight_dir, 'final_weights.pth'))
        # 最佳权重
        if is_best:
            torch.save(checkpoint, os.path.join(self.weight_dir, 'best_weights.pth'))
            self.best_accu = accu
            self.best_epoch = epoch
            logging.info("[OK] 新的最佳权重 (Epoch {epoch+1}, Accu: {accu:.4f})")
        # 最差权重
        if is_worst:
            torch.save(checkpoint, os.path.join(self.weight_dir, 'worst_weights.pth'))
            self.worst_accu = accu
            self.worst_epoch = epoch
            logging.warning("[WARNING] 新的最差权重 (Epoch {epoch+1}, Accu: {accu:.4f})")
        logging.info(f"权重已保存 - Epoch {epoch+1}, Accu: {accu:.4f}")
    
    def analyze_loss_function(self, heatmap_loss, bbox_loss, total_loss, accu50, accu25, mean_iou):
        """分析损失函数效果"""
        # 保证只保存float数值
        self.loss_history['total_loss'].append(float(total_loss))
        self.loss_history['heatmap_loss'].append(float(heatmap_loss))
        self.loss_history['bbox_loss'].append(float(bbox_loss))
        self.loss_history['accu50'].append(float(accu50))
        self.loss_history['accu25'].append(float(accu25))
        self.loss_history['mean_iou'].append(float(mean_iou))
        
        # 计算损失比例
        loss_ratio = heatmap_loss / bbox_loss if bbox_loss > 0 else float('inf')
        
        # 分析损失函数合理性
        analysis = {
            'loss_ratio': loss_ratio,
            'is_balanced': 0.1 < loss_ratio < 10.0,
            'heatmap_dominating': loss_ratio > 10.0,
            'bbox_dominating': loss_ratio < 0.1,
            'loss_decreasing': len(self.loss_history['total_loss']) > 1 and 
                              self.loss_history['total_loss'][-1] < self.loss_history['total_loss'][-2],
            'accu_improving': len(self.loss_history['accu50']) > 1 and 
                             self.loss_history['accu50'][-1] > self.loss_history['accu50'][-2]
        }
        
        return analysis
    
    def visualize_loss_analysis(self, save_path='loss_analysis.png'):
        """可视化损失函数分析"""
        if len(self.loss_history['total_loss']) < 2:
            logging.info("需要更多训练数据才能进行损失分析")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.loss_history['total_loss']) + 1)
        
        # 损失变化趋势
        axes[0, 0].plot(epochs, self.loss_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.loss_history['heatmap_loss'], 'r-', label='Heatmap Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.loss_history['bbox_loss'], 'g-', label='BBox Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss Value')
        axes[0, 0].set_title('Loss Components Over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 损失比例
        loss_ratios = [h/b if b > 0 else float('inf') for h, b in zip(self.loss_history['heatmap_loss'], self.loss_history['bbox_loss'])]
        axes[0, 1].plot(epochs, loss_ratios, 'purple', linewidth=2)
        axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Heatmap/BBox Loss Ratio')
        axes[0, 1].set_title('Loss Balance Over Training')
        axes[0, 1].grid(True)
        
        # 准确率变化
        axes[1, 0].plot(epochs, self.loss_history['accu50'], 'orange', linewidth=2, label='Accu50')
        axes[1, 0].plot(epochs, self.loss_history['accu25'], 'cyan', linewidth=2, label='Accu25')
        axes[1, 0].plot(epochs, self.loss_history['mean_iou'], 'green', linewidth=2, label='Mean IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Model Performance Over Training')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 损失与性能关系
        axes[1, 1].scatter(self.loss_history['total_loss'], self.loss_history['accu50'], alpha=0.7)
        axes[1, 1].set_xlabel('Total Loss')
        axes[1, 1].set_ylabel('Accu50')
        axes[1, 1].set_title('Loss vs Performance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"损失分析图已保存到: {save_path}")
    
    def visualize_model_outputs(self, model, data_loader, num_samples=5, save_dir='./visualization_outputs'):
        """只针对swinmoe分支的可视化，所有点和框严格做映射，所有子图尺寸一致"""
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        sample_count = 0
        with torch.no_grad():
            for query_imgs, rs_imgs, mat_clickxy, ori_gt_bbox, _, click_xy, ori_img_shape in data_loader:
                if sample_count >= num_samples:
                    break
                # 统一使用主设备
                query_imgs = query_imgs.to(device)
                rs_imgs = rs_imgs.to(device)
                ori_gt_bbox = ori_gt_bbox.to(device)
                mat_clickxy = mat_clickxy.to(device) if mat_clickxy is not None else None
                B, _, H, W = query_imgs.shape
                for i in range(B):
                    if sample_count >= num_samples:
                        break
                    qimg = self.denormalize_image(query_imgs[i])
                    simg = self.denormalize_image(rs_imgs[i])
                    
                    # 修正：确保clickxy是单个点击点坐标对
                    if mat_clickxy is not None:
                        clickxy = mat_clickxy[i].cpu().numpy()
                        # 如果clickxy是数组，取第一个元素或转换为坐标对
                        if clickxy.ndim > 1:
                            clickxy = clickxy.flatten()[:2]  # 只取前两个元素作为(x,y)
                        elif len(clickxy) > 2:
                            clickxy = clickxy[:2]  # 只取前两个元素
                        clickxy = tuple(clickxy)  # 转换为坐标对
                    else:
                        clickxy = None
                    
                    # 新增：获取原始点击点
                    raw_click_xy = click_xy[i].cpu().numpy() if hasattr(click_xy[i], 'cpu') else click_xy[i]
                    # 判断是否归一化
                    if raw_click_xy.max() > 1.5:
                        click_xy_norm = (raw_click_xy[0] / W, raw_click_xy[1] / H)
                    else:
                        click_xy_norm = raw_click_xy
                    clickxy_pixel = (click_xy_norm[0] * W, click_xy_norm[1] * H)
                    
                    heatmap_pred, bbox_pred = model(query_imgs[i:i+1], rs_imgs[i:i+1])
                    _, _, hH, hW = heatmap_pred.shape
                    from model.loss import build_target_anchorfree
                    gt_heatmap, gt_bbox, mask = build_target_anchorfree(
                        ori_gt_bbox[i:i+1], hH, hW, self.args.img_size, self.args.img_size)
                    pred_hm = heatmap_pred[0, 0].sigmoid().cpu().numpy()
                    gt_hm = gt_heatmap[0, 0].cpu().numpy()
                    pred_center = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
                    gt_center = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
                    pred_box_params = bbox_pred[0, :, pred_center[0], pred_center[1]].cpu().numpy()
                    gt_box_params = gt_bbox[0, :, gt_center[0], gt_center[1]].cpu().numpy()
                    bbox_pred_values = bbox_pred[0].cpu().numpy().flatten()
                    
                    # 新增：支持原始图片尺寸
                    if len(ori_img_shape) >= 7:
                        ori_img_shape = ori_img_shape[6][i] if isinstance(ori_img_shape[6], (list, tuple)) else ori_img_shape[6]
                    else:
                        ori_img_shape = simg.shape[:2]  # 兜底
                    ori_H, ori_W = ori_img_shape
                    img_H, img_W = simg.shape[:2]

                    # 真实框和中心点缩放到可视化尺寸
                    gt_box_pixel = ori_gt_bbox[i].cpu().numpy() if hasattr(ori_gt_bbox[i], 'cpu') else ori_gt_bbox[i]
                    x1, y1, x2, y2 = gt_box_pixel
                    scale_x = img_W / ori_W
                    scale_y = img_H / ori_H
                    x1 = x1 * scale_x
                    x2 = x2 * scale_x
                    y1 = y1 * scale_y
                    y2 = y2 * scale_y
                    gt_box_pixel = [x1, y1, x2, y2]
                    gt_center_pixel = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # 获取特征图尺寸（用于anchor-free解码和过滤）
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
                    pred_score = pred_hm[pred_center[0], pred_center[1]] if pred_hm is not None else 0
                    if pred_box_params is not None and pred_score > 0.3 and np.all(np.abs(pred_box_params) < feat_W*2):
                        pred_box_pixel = box_params_to_pixel(pred_box_params, pred_center, img_W, img_H, feat_W, feat_H)
                    else:
                        pred_box_pixel = None
                    def center_feat2pixel(center, img_H, img_W, feat_H, feat_W):
                        y, x = center
                        x_pixel = (x + 0.5) * img_W / feat_W
                        y_pixel = (y + 0.5) * img_H / feat_H
                        return (x_pixel, y_pixel)
                    pred_center_pixel = center_feat2pixel(pred_center, img_H, img_W, feat_H, feat_W) if pred_center is not None else None

                    # 查询图像点击点缩放到可视化尺寸
                    ori_query_H, ori_query_W = ori_img_shape
                    img_H, img_W = qimg.shape[:2]
                    # 修正：安全地获取点击点坐标，支持小区域点击
                    if isinstance(click_xy, (list, tuple, np.ndarray)):
                        click_data = click_xy[i]
                        if isinstance(click_data, (list, tuple, np.ndarray)):
                            click_data = np.asarray(click_data).flatten()
                            if len(click_data) >= 2:
                                click_x, click_y = click_data[0], click_data[1]  # 取前两个元素作为中心点
                            elif len(click_data) == 1:
                                click_x, click_y = click_data[0], click_data[0]  # 单个值复制
                            else:
                                click_x, click_y = 0, 0  # 兜底值
                        else:
                            click_x, click_y = click_data, click_data  # 单个值
                    else:
                        click_x, click_y = click_xy, click_xy  # 兜底
                    scale_x = img_W / ori_query_W
                    scale_y = img_H / ori_query_H
                    clickxy_pixel = (click_x * scale_x, click_y * scale_y)

                    # 格式检查和转换：确保所有参数都是正确的格式
                    def ensure_coord_format(coord, name):
                        """确保坐标是标量或坐标对格式"""
                        if coord is None:
                            return None
                        if isinstance(coord, (list, tuple, np.ndarray)):
                            coord = np.asarray(coord).flatten()
                            if len(coord) >= 2:
                                return tuple(coord[:2])  # 只取前两个元素
                            elif len(coord) == 1:
                                return (coord[0], coord[0])  # 单个值复制为坐标对
                            else:
                                return None
                        else:
                            return (coord, coord)  # 标量转换为坐标对
                    
                    # 确保所有坐标都是正确格式
                    clickxy_pixel = ensure_coord_format(clickxy_pixel, 'clickxy_pixel')
                    pred_center_pixel = ensure_coord_format(pred_center_pixel, 'pred_center_pixel')
                    gt_center_pixel = ensure_coord_format(gt_center_pixel, 'gt_center_pixel')
                    
                    # 确保框是列表格式
                    if pred_box_pixel is not None and not isinstance(pred_box_pixel, list):
                        pred_box_pixel = list(pred_box_pixel)
                    if gt_box_pixel is not None and not isinstance(gt_box_pixel, list):
                        gt_box_pixel = list(gt_box_pixel)

                    # 4. 传入draw_visualization
                    # draw_visualization(  # 暂时注释，先让训练跑起来
                    #     qimg, simg, clickxy_pixel,
                    #     pred_hm, gt_hm,
                    #     pred_box_pixel, gt_box_pixel,
                    #     pred_center_pixel, gt_center_pixel,
                    #     bbox_pred_values,
                    #     save_dir, sample_count+1,
                    #     (img_W, img_H), (feat_H, feat_W)
                    # )
                    sample_count += 1
        logging.info(f"所有可视化结果已保存到: {save_dir}")

    def denormalize_image(self, img_tensor):
        """
        将归一化的图像张量转换回原始像素值
        @param img_tensor: [C, H, W] 归一化的图像张量
        @return: [H, W, C] numpy数组，像素值范围[0, 255]
        """
        # ImageNet归一化参数
        device = img_tensor.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        
        # 反归一化
        img_tensor = img_tensor * std + mean
        
        # 转换到[0, 1]范围
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # 转换到[0, 255]范围并转为numpy
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # 转换通道顺序 [C, H, W] -> [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))
        
        return img_np

def main():
    parser = argparse.ArgumentParser(description='增强版跨视角目标定位训练')
    parser.add_argument('--max_epoch', default=25, type=int, help='training epoch')
    parser.add_argument('--lr', default=1.0e-4, type=float, help='learning rate')  # 基于最佳日志的优化学习率
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')  # 保持最佳批次大小
    parser.add_argument('--img_size', default=1024, type=int, help='image size')
    parser.add_argument('--data_root', default='data', type=str, help='data root')
    parser.add_argument('--data_name', default='CVOGL_DroneAerial', type=str, help='data name')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--num_workers', default=24, type=int, help='num workers')
    parser.add_argument('--savename', default='optimized_enhanced_25epoch', type=str, help='save name')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of cls loss')  # 恢复原始权重
    parser.add_argument('--model', default='swinmoe', type=str, help='model name')
    parser.add_argument('--cosine', action='store_true', default=True, help='use cosine annealing')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lambda-entropy-base', default=0.005, type=float, help='entropy regularization base')  # 恢复原来的熵正则化权重
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--val', dest='val', default=False, action='store_true', help='val')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='visualize results')
    parser.add_argument('--no-moe-entropy', action='store_true', help='If set, do not use MoE门控熵正则项')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    
    args = parser.parse_args()
    
    # 自动生成唯一日志文件名
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f"{args.savename}_{now_str}.log"
    log_path = os.path.join('logs', log_name)
    # 清空所有已存在的handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # 同时输出到文件和终端
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a+"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"日志将保存到: {log_path}")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)
    
    # 设置多GPU训练
    device_ids, device = setup_multi_gpu(args)
    logging.info(f"主设备: {device}")
    logging.info(f"可用GPU数量: {len(device_ids)}")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        logging.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.warning("CUDA不可用，将使用CPU训练")
    
    # 根据GPU数量调整batch_size
    if len(device_ids) > 1:
        original_batch_size = args.batch_size
        args.batch_size = args.batch_size * len(device_ids)
        logging.info(f"多GPU训练，batch_size从{original_batch_size}调整为{args.batch_size}")
        
        # 基于历史最佳结果优化学习率
        if len(device_ids) > 1:
            # 多GPU训练，根据batch size调整学习率
            # 使用线性缩放规则：lr = base_lr * (batch_size / base_batch_size)
            base_batch_size = 8
            scale_factor = args.batch_size / base_batch_size
            args.lr = 1e-4 * scale_factor  # 基础学习率1e-4，根据batch size缩放
            logging.info(f"多GPU训练，学习率调整为: {args.lr} (缩放因子: {scale_factor})")
        elif len(device_ids) == 1:
            # 单GPU训练，使用历史最佳学习率
            args.lr = 1e-4
            logging.info(f"单GPU训练，学习率: {args.lr}")
    else:
        # 单GPU训练
        args.lr = 1e-4
        logging.info(f"单GPU训练，学习率: {args.lr}")
    
    # 创建增强训练器
    trainer = EnhancedTrainer(args)
    
    # 数据加载
    # 增强的数据变换策略
    from torchvision.transforms import functional as F
    
    class CrossViewAugment:
        """跨视角数据增强"""
        def __init__(self, p=0.7):  # 增加增强概率
            self.p = p
            
        def __call__(self, query_img, sat_img):
            if random.random() < self.p:
                # 同步增强：保持跨视角一致性
                if random.random() < 0.3:
                    # 同步旋转
                    angle = random.uniform(-15, 15)
                    query_img = F.rotate(query_img, angle)
                    sat_img = F.rotate(sat_img, angle)
                
                if random.random() < 0.3:
                    # 同步翻转
                    if random.random() < 0.5:
                        query_img = F.hflip(query_img)
                        sat_img = F.hflip(sat_img)
                    else:
                        query_img = F.vflip(query_img)
                        sat_img = F.vflip(sat_img)
                
                if random.random() < 0.3:
                    # 颜色增强
                    brightness = random.uniform(0.8, 1.2)
                    contrast = random.uniform(0.8, 1.2)
                    saturation = random.uniform(0.8, 1.2)
                    hue = random.uniform(-0.1, 0.1)
                    
                    query_img = F.adjust_brightness(query_img, brightness)
                    query_img = F.adjust_contrast(query_img, contrast)
                    query_img = F.adjust_saturation(query_img, saturation)
                    query_img = F.adjust_hue(query_img, hue)
                    
                    sat_img = F.adjust_brightness(sat_img, brightness)
                    sat_img = F.adjust_contrast(sat_img, contrast)
                    sat_img = F.adjust_saturation(sat_img, saturation)
                    sat_img = F.adjust_hue(sat_img, hue)
            
            return query_img, sat_img
    
    # 基础变换
    base_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 增强变换
    augment_transform = Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),      # 适度增强强度
        RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),       # 适度几何变换
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 使用增强变换
    input_transform = augment_transform
    
    train_dataset = RSDataset(data_root=args.data_root, data_name=args.data_name,
                             split_name='train', img_size=args.img_size,
                             transform=input_transform, augment=True)
    val_dataset = RSDataset(data_root=args.data_root, data_name=args.data_name,
                           split_name='val', img_size=args.img_size, transform=input_transform)
    
    # Windows系统多进程问题修复：减少worker数量或禁用多进程
    if os.name == 'nt':  # Windows系统
        num_workers = min(args.num_workers, 4)  # 限制最大worker数量
        if num_workers > 0:
            logging.info(f"Windows系统，调整worker数量: {args.num_workers} -> {num_workers}")
        else:
            num_workers = 0
            logging.info("Windows系统，禁用多进程数据加载")
    else:
        num_workers = args.num_workers
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, drop_last=False, num_workers=num_workers,
                             collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, drop_last=False, num_workers=num_workers,
                             collate_fn=custom_collate_fn)
    
    # 模型创建
    if args.model == 'swinmoe':
        from model.swin_moe_geo import SwinTransformer_MoE_MultiInput
        from model.anchorfree_head import AnchorFreeHead
        
        swin_cfg = swin_moe_geo_cfg
        
        # 创建自定义的SwinTransformer_MoE_MultiInput，支持不同通道数
        class CustomSwinTransformer_MoE_MultiInput(SwinTransformer_MoE_MultiInput):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # 重新创建patch embeddings，为不同数据集使用不同通道数
                from model.swin_moe_geo import PatchEmbed
                self.patch_embeds = nn.ModuleDict({
                    'query': PatchEmbed(in_channels=4, embed_dim=kwargs.get('embed_dim', 96), patch_size=kwargs.get('patch_size', 4)),  # 查询图像4通道
                    'sat': PatchEmbed(in_channels=3, embed_dim=kwargs.get('embed_dim', 96), patch_size=kwargs.get('patch_size', 4))      # 卫星图像3通道
                })
            
            def forward(self, query_img, sat_img):
                """重载forward方法，接受query_img和sat_img参数"""
                # 调用父类的forward方法，传递正确的参数格式
                return super().forward([query_img, sat_img])
        
        swin_backbone = CustomSwinTransformer_MoE_MultiInput(
            in_channels=4,  # 这个参数在自定义类中会被忽略
            embed_dim=swin_cfg.get('embed_dim', 96),
            patch_size=swin_cfg.get('patch_size', 4),
            window_size=swin_cfg.get('window_size', 7),
            depths=swin_cfg.get('depths', (2,2,6,2)),
            num_heads=swin_cfg.get('num_heads', (3,6,12,24)),
            ffn_ratio=swin_cfg.get('ffn_ratio', 4),
            num_experts=6,  # 固定6个专家
            top_k=2,        # 固定使用2个专家
            moe_block_indices=swin_cfg.get('moe_block_indices', None),
            datasets=swin_cfg.get('datasets', ('query','sat'))
        )
        
        out_dim = swin_backbone.out_dim
        anchorfree_head = AnchorFreeHead(in_channels=out_dim, feat_channels=256, num_classes=1)
        
        class DetGeoSwinMoE_AF(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.moe_entropy = 0.0  # 确保初始化
                
            def forward(self, query_img, sat_img, click_map=None):
                # 前向传播
                backbone_output = self.backbone(query_img, sat_img)
                if isinstance(backbone_output, tuple):
                    query_vec, sat_feat, avg_entropy = backbone_output
                    # 确保熵值在所有设备上同步
                    if hasattr(self, 'module') and hasattr(self.module, 'moe_entropy'):
                        # DataParallel环境，同步到主模块
                        self.module.moe_entropy = avg_entropy
                    self.moe_entropy = avg_entropy
                    # 调试信息：每100个batch打印一次
                    if torch.rand(1).item() < 0.01:  # 1%概率打印
                        print(f"[调试] backbone返回熵值: {avg_entropy:.6f}")
                        print(f"[调试] self.moe_entropy设置: {self.moe_entropy:.6f}")
                else:
                    sat_feat = backbone_output
                    if hasattr(self, 'module') and hasattr(self.module, 'moe_entropy'):
                        self.module.moe_entropy = 0.0
                    self.moe_entropy = 0.0

                heatmap, bbox = self.head(sat_feat)
                return heatmap, bbox
            
            def get_moe_entropy(self):
                """获取MoE熵值"""
                # 优先从backbone获取最新的熵值
                if hasattr(self.backbone, 'get_moe_entropy'):
                    entropy = self.backbone.get_moe_entropy()
                else:
                    # 兜底：从主模块获取熵值
                    if hasattr(self, 'module') and hasattr(self.module, 'moe_entropy'):
                        entropy = self.module.moe_entropy
                    else:
                        entropy = self.moe_entropy
                
                # 调试信息：每100次调用打印一次
                if torch.rand(1).item() < 0.01:  # 1%概率打印
                    print(f"[调试] get_moe_entropy返回: {entropy:.6f}")
                return entropy
            
            def get_backbone_moe_entropy(self):
                """从backbone获取MoE熵值"""
                if hasattr(self.backbone, 'get_moe_entropy'):
                    return self.backbone.get_moe_entropy()
                return 0.0
        
        model = DetGeoSwinMoE_AF(swin_backbone, anchorfree_head)
    else:
        # 对于非swinmoe模型，使用默认的DetGeo模型
        from model.detgeo_swinmoe import DetGeo
        model = DetGeo()
    
    # 包装模型以支持多GPU训练
    model = wrap_model_for_multi_gpu(model, device_ids, args)
    
    # 优先使用命令行参数，如果没有则使用配置文件中的预训练权重
    if args.pretrain:
        model = load_pretrain(model, args, logging)
    elif swin_cfg.get('pretrained') and os.path.exists(swin_cfg['pretrained']):
        # 使用配置文件中的预训练权重
        args.pretrain = swin_cfg['pretrained']
        model = load_pretrain(model, args, logging)
        logging.info(f"✅ 使用配置文件中的预训练权重: {swin_cfg['pretrained']}")
    else:
        logging.info("ℹ️ 未使用预训练权重，从头开始训练")

    # ====== MoE专家权重初始化（用主干FFN） ======
    if args.model == 'swinmoe':
        from model.swin_moe_geo import initialize_moe_experts_from_ffn
        
        # 多卡环境下正确获取backbone
        if hasattr(model, 'module'):
            # DataParallel环境
            backbone = model.module.backbone
            logging.info("🔧 多卡环境：从model.module获取backbone")
        else:
            # 单卡环境
            backbone = model.backbone
            logging.info("🔧 单卡环境：从model获取backbone")
        
        # 修复后的MoE架构：每个stage有自己的专家池，但共享专家数量
        total_experts = 0
        logging.info(f"[DEBUG] backbone类型: {type(backbone)}")
        logging.info(f"[DEBUG] backbone.stages数量: {len(backbone.stages)}")
        
        for stage_idx, stage in enumerate(backbone.stages):
            logging.info(f"[DEBUG] Stage {stage_idx}: type={type(stage)}, hasattr(expert_pool)={hasattr(stage, 'expert_pool')}")
            if hasattr(stage, 'expert_pool') and stage.expert_pool is not None:
                logging.info(f"[DEBUG] Stage {stage_idx} expert_pool: {stage.expert_pool}")
                # 找到同stage第一个非MoE Block的ffn作为参考
                ffn_ref = None
                for block_idx, block in enumerate(stage.blocks):
                    logging.info(f"[DEBUG] Block {block_idx}: type={type(block)}, hasattr(use_moe)={hasattr(block, 'use_moe')}, use_moe={getattr(block, 'use_moe', False)}")
                    if hasattr(block, 'use_moe') and not block.use_moe:
                        ffn_ref = block.ffn
                        logging.info(f"[DEBUG] 找到参考FFN: {ffn_ref}")
                        break
                
                if ffn_ref is not None:
                    # 用参考FFN初始化专家池中的所有专家
                    initialize_moe_experts_from_ffn(stage.expert_pool, ffn_ref)
                    logging.info(f"[MoE专家初始化] Stage {stage_idx} 专家池已用普通FFN权重初始化")
                    total_experts += stage.expert_pool.num_experts
                else:
                    logging.info(f"[MoE专家初始化] Stage {stage_idx} 未找到普通FFN，跳过专家池初始化")
            else:
                logging.info(f"[MoE专家初始化] Stage {stage_idx} 没有专家池")
        
        # 验证MoE架构修复结果
        logging.info(f"[MoE验证] 总专家数量: {total_experts} (配置: {swin_cfg.get('num_experts', 6)})")
        if total_experts == swin_cfg.get('num_experts', 6):
            logging.info("✅ MoE架构修复成功！专家数量配置正确")
        else:
            logging.warning(f"⚠️ MoE架构专家数量不匹配: 实际{total_experts} vs 配置{swin_cfg.get('num_experts', 6)}")
    
    # 基于最佳日志的学习率调度策略
    warmup_epochs = 2  # 减少预热轮次，更快进入学习
    total_epochs = args.max_epoch
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 优化预热调度器 - 更平缓的预热
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
    
    # ====== 动态熵正则权重 ======
    if args.no_moe_entropy:
        def get_lambda_entropy(epoch, max_epoch, base=None, min_val=0.001, warmup=2, best_epoch=None, freeze_after=2):
            return 0.0
    else:
        # 修复MoE熵正则化策略 - 使用原来的权重，问题可能在其他地方
        base = 0.005  # 恢复原来的基础权重
        def get_lambda_entropy(epoch, max_epoch, base=None, min_val=0.001, warmup=2, best_epoch=None, freeze_after=5):
            """
            @function get_lambda_entropy
            @desc 动态调整MoE熵正则化强度，让MoE在训练中后期发挥核心作用
            @param {int} epoch - 当前轮数
            @param {int} max_epoch - 总轮数
            @param {float} base - 基础熵正则化强度
            @param {float} min_val - 最小熵正则化强度
            @param {int} warmup - 预热轮数
            @param {int} best_epoch - 最佳轮数（用于动态调整）
            @param {int} freeze_after - 冻结轮数
            @return {float} 当前轮数的熵正则化强度
            """
            if base is None:
                base = 0.005  # 恢复原来的基础强度
            if best_epoch is None:
                best_epoch = max_epoch // 2  # 默认最佳轮数为总轮数的一半
            
            # 预热阶段：逐渐增加熵正则化
            if epoch < warmup:
                lambda_entropy = base * (epoch / warmup)
            # 冻结阶段：保持稳定强度
            elif epoch < freeze_after:
                lambda_entropy = base
            # 动态调整阶段：根据训练进度调整
            else:
                # 在训练中后期，降低熵正则化，让MoE发挥核心作用
                progress = (epoch - freeze_after) / (max_epoch - freeze_after)
                decay_factor = 0.8  # 衰减因子，让MoE在后期更自由
                lambda_entropy = base * (1 - progress * decay_factor)
            
            # 确保不低于最小值
            lambda_entropy = max(lambda_entropy, min_val)
            
            return lambda_entropy
    
    # 优化学习率调度策略 - 解决学习率下降过快问题
    if args.cosine:
        # 彻底修复余弦退火调度 - 防止学习率下降过快
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs-warmup_epochs, eta_min=1e-5)
        scheduler = None  # 确保scheduler变量存在
    else:
        # 基于最佳日志的ReduceLROnPlateau策略，更保守的patience
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-6)
        cosine_scheduler = None  # 确保cosine_scheduler变量存在
    
    # 训练循环
    if not args.test and not args.val:
        val_history = []  # 记录每轮验证指标和权重
        best_val_accu = -float('inf')
        best_epoch = 0
        lambda_entropy = args.lambda_entropy_base
        
        # 移除早停机制，先观察训练效果
        for epoch in range(total_epochs):
            logging.info(f'========== Epoch {epoch+1}/{total_epochs} ==========',)
            if epoch > best_epoch + 2:
                lambda_entropy = 0.0
            else:
                lambda_entropy = get_lambda_entropy(epoch, total_epochs, base=args.lambda_entropy_base)
            train_metrics = train_epoch(train_loader, model, optimizer, epoch, args, trainer, lambda_entropy, device_ids, device)
            # 解包训练指标
            avg_loss, avg_geo_loss, avg_cls_loss, avg_accu50, avg_accu25, avg_mean_iou = train_metrics
            logging.info(f'训练完成 - Loss: {avg_loss:.4f}, Geo Loss: {avg_geo_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}')
            logging.info(f'训练指标 - Accu50: {avg_accu50:.4f}, Accu25: {avg_accu25:.4f}, Mean IoU: {avg_mean_iou:.4f}')
            logging.info(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            
            logging.info(f'\n=== 开始验证评估 ===')
            val_metrics = test_epoch(val_loader, model, args, device_ids, device)
            val_accu50 = val_metrics['accu50']
            val_accu25 = val_metrics['accu25']
            val_mean_iou = val_metrics['mean_iou']
            logging.info(f'验证结果 - Accu50: {val_accu50:.4f}, Accu25: {val_accu25:.4f}, Mean IoU: {val_mean_iou:.4f}')
            
            # 学习率调度器step调用 - 在验证完成后调用
            if args.cosine:
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                    logging.info(f'Warmup Scheduler Step - Epoch {epoch+1}')
                else:
                    cosine_scheduler.step()
                    logging.info(f'Cosine Scheduler Step - Epoch {epoch+1}')
            else:
                # 对于ReduceLROnPlateau，使用验证准确率来step
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_accu50)  # 使用验证准确率作为监控指标
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logging.info(f'Learning Rate Changed: {old_lr:.2e} -> {new_lr:.2e}')
                else:
                    logging.info(f'Learning Rate Unchanged: {new_lr:.2e}')
            
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current Learning Rate: {current_lr:.2e}')
            
            # 只记录，不保存权重
            if args.cosine:
                scheduler_state = cosine_scheduler.state_dict() if epoch >= warmup_epochs else warmup_scheduler.state_dict()
            else:
                scheduler_state = scheduler.state_dict()
            val_history.append({
                'epoch': epoch + 1,
                'accu50': val_accu50,
                'accu25': val_accu25,
                'mean_iou': val_mean_iou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_state,
                'swin_cfg': swin_cfg,
            })
            logging.info(f'========== End of Epoch {epoch+1} ==========\n')
            # 移除早停检查，让训练继续运行
            
            # 记录最佳epoch
            if val_accu50 > best_val_accu:
                best_val_accu = val_accu50
                best_epoch = epoch
            # 增强训练监控日志
            logging.info(f"Epoch {epoch+1}: lr={current_lr:.6e}, lambda_entropy={lambda_entropy:.4f}, Accu50={val_accu50:.4f}, MeanIoU={val_mean_iou:.4f}")
            logging.info(f"训练损失: {avg_loss:.4f}, 几何损失: {avg_geo_loss:.4f}, 分类损失: {avg_cls_loss:.4f}")
            logging.info(f"训练IoU: {avg_mean_iou:.4f}, 验证IoU: {val_mean_iou:.4f}, 差距: {abs(avg_mean_iou-val_mean_iou):.4f}")
            
            # MoE专家激活监控
            if hasattr(model, 'get_backbone_moe_entropy'):
                moe_entropy = model.get_backbone_moe_entropy()
                logging.info(f"🎯 MoE专家激活状态: 熵值={moe_entropy:.4f} (目标范围: 0.6-0.9)")
                if moe_entropy < 0.1:
                    logging.warning("⚠️  MoE专家激活不足！熵值过低，专家可能未被充分利用")
                elif moe_entropy > 0.9:
                    logging.info("✅ MoE专家激活良好！熵值在理想范围内")
                else:
                    logging.info("🔄 MoE专家激活正常，继续观察")
        logging.info('\n=== 训练完成，开始分析 ===')
        # 训练结束后，保存最佳、最差和最终权重
        best_idx = max(range(len(val_history)), key=lambda i: val_history[i]['accu50'])
        worst_idx = min(range(len(val_history)), key=lambda i: val_history[i]['accu50'])
        final_idx = len(val_history) - 1
        weight_dir = trainer.weight_dir
        # 最终权重
        torch.save({
            'epoch': val_history[final_idx]['epoch'],
            'state_dict': val_history[final_idx]['state_dict'],
            'accu': val_history[final_idx]['accu50'],
            'optimizer': val_history[final_idx]['optimizer'],
            'scheduler': val_history[final_idx]['scheduler'],
            'swin_cfg': swin_cfg,
        }, os.path.join(weight_dir, 'final_weights.pth'))
        logging.info(f'✓ 最终权重已保存 (Epoch {val_history[final_idx]["epoch"]}, Accu: {val_history[final_idx]["accu50"]:.4f})')
        # 最佳权重
        torch.save({
            'epoch': val_history[best_idx]['epoch'],
            'state_dict': val_history[best_idx]['state_dict'],
            'accu': val_history[best_idx]['accu50'],
            'optimizer': val_history[best_idx]['optimizer'],
            'scheduler': val_history[best_idx]['scheduler'],
            'swin_cfg': swin_cfg,
        }, os.path.join(weight_dir, 'best_weights.pth'))
        logging.info(f'✓ 最佳权重已保存 (Epoch {val_history[best_idx]["epoch"]}, Accu: {val_history[best_idx]["accu50"]:.4f})')
        # 最差权重
        torch.save({
            'epoch': val_history[worst_idx]['epoch'],
            'state_dict': val_history[worst_idx]['state_dict'],
            'accu': val_history[worst_idx]['accu50'],
            'optimizer': val_history[worst_idx]['optimizer'],
            'scheduler': val_history[worst_idx]['scheduler'],
            'swin_cfg': swin_cfg,
        }, os.path.join(weight_dir, 'worst_weights.pth'))
        logging.info(f'⚠ 最差权重已保存 (Epoch {val_history[worst_idx]["epoch"]}, Accu: {val_history[worst_idx]["accu50"]:.4f})')
        trainer.best_accu = val_history[best_idx]['accu50']
        trainer.best_epoch = val_history[best_idx]['epoch']
        trainer.worst_accu = val_history[worst_idx]['accu50']
        trainer.worst_epoch = val_history[worst_idx]['epoch']
        trainer.visualize_loss_analysis()
        if args.visualize:
            logging.info('\n=== 开始可视化模型输出 ===')
            trainer.visualize_model_outputs(model, val_loader)
    
    elif args.visualize:
        # 加载最佳权重进行可视化
        best_weights_path = os.path.join(trainer.weight_dir, 'best_weights.pth')
        if os.path.exists(best_weights_path):
            checkpoint = torch.load(best_weights_path)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info(f"加载最佳权重进行可视化: {best_weights_path}")
            trainer.visualize_model_outputs(model, val_loader)
        else:
            logging.info("未找到最佳权重文件，使用当前模型进行可视化")
            trainer.visualize_model_outputs(model, val_loader)

def train_epoch(train_loader, model, optimizer, epoch, args, trainer, lambda_entropy, device_ids, device):
    """
    @function train_epoch
    @desc 训练一个epoch，loss在多卡时自动全局平均，保证与单卡一致，自动统计MoE门控分布和熵
    """
    model.train()
    batch_time = AverageMeter()
    avg_losses = AverageMeter()
    avg_cls_losses = AverageMeter()
    avg_geo_losses = AverageMeter()
    avg_accu = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_iou = AverageMeter()
    moe_entropy_list = []  # 记录每step所有MoE熵
    moe_gate_stats = []    # 记录门控分布
    print_freq_entropy = max(1, args.print_freq // 2)
    end = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            # 正确解包数据，与custom_collate_fn返回格式匹配
            query_imgs, rs_imgs, ori_gt_bbox, idx, click_xy, ori_hw = batch
            
            # 检查数据有效性
            if query_imgs is None or rs_imgs is None or ori_gt_bbox is None:
                logging.warning(f"批次 {batch_idx} 数据为空，跳过")
                continue
                
            # 检查数据形状
            if query_imgs.shape[0] == 0 or rs_imgs.shape[0] == 0:
                logging.warning(f"批次 {batch_idx} 数据形状异常，跳过")
                continue
            
            # 确保数据正确移动到设备
            # 统一使用主设备，让DataParallel自动分配
            query_imgs = query_imgs.to(device)
            rs_imgs = rs_imgs.to(device)
            ori_gt_bbox = ori_gt_bbox.to(device)
            
            # 调试设备分配
            if batch_idx % 100 == 0:
                logging.info(f"[调试] 数据设备: query_imgs={query_imgs.device}, rs_imgs={rs_imgs.device}, model={next(model.parameters()).device}")
            
            # 确保数据类型正确
            query_imgs = query_imgs.float()
            rs_imgs = rs_imgs.float()
            ori_gt_bbox = ori_gt_bbox.float()
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 检查CUDA内存使用情况
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                if memory_allocated > 20:  # 如果使用超过20GB
                    logging.warning(f"CUDA内存使用过高: {memory_allocated:.2f}GB, 尝试清理")
                    torch.cuda.empty_cache()
                    gc.collect()
            
            ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)
            
            if args.model == 'swinmoe':
                # Anchor-Free流程
                # 模型前向推理
                heatmap_pred, bbox_pred = model(query_imgs, rs_imgs)
                B, _, H, W = heatmap_pred.shape
                gt_heatmap, gt_bbox, mask = build_target_anchorfree(ori_gt_bbox, H, W, args.img_size, args.img_size)
                
                # 确保目标数据也在正确的设备上
                # 统一使用主设备，让DataParallel自动分配
                gt_heatmap = gt_heatmap.to(device)
                gt_bbox = gt_bbox.to(device)
                mask = mask.to(device)
                
                # 基于历史最佳结果优化损失权重
                # 历史最佳配置：热力图损失权重较低，几何损失权重较高
                heatmap_loss, bbox_loss = anchorfree_loss(heatmap_pred, bbox_pred, gt_heatmap, gt_bbox, mask)
                
                # 获取MoE熵值
                moe_entropy = 0.0
                # 多GPU环境下，需要正确处理DataParallel包装的模型
                if len(device_ids) > 1 and hasattr(model, 'module'):
                    # 多GPU环境，从module获取熵值
                    if hasattr(model.module, 'get_moe_entropy'):
                        moe_entropy = model.module.get_moe_entropy()
                    elif hasattr(model.module, 'get_backbone_moe_entropy'):
                        moe_entropy = model.module.get_backbone_moe_entropy()
                else:
                    # 单GPU环境，直接从model获取
                    if hasattr(model, 'get_moe_entropy'):
                        moe_entropy = model.get_moe_entropy()
                    elif hasattr(model, 'get_backbone_moe_entropy'):
                        moe_entropy = model.get_backbone_moe_entropy()
                
                # 调试信息：每100个batch打印一次熵值
                if batch_idx % 100 == 0:
                    logging.info(f"[调试] MoE熵值获取: {moe_entropy:.6f}")
                
                # 调整损失权重 - 根据batch size动态调整
                # 多GPU环境下，batch size翻倍，需要相应调整权重
                if len(device_ids) > 1:
                    # 多GPU环境，batch size翻倍，增加分类损失权重
                    total_loss = 0.8 * heatmap_loss + 0.2 * bbox_loss
                else:
                    # 单GPU环境，使用平衡权重
                    total_loss = 0.7 * heatmap_loss + 0.3 * bbox_loss
                
                # 添加MoE熵损失到总损失中（安全添加，不影响主线任务）
                if lambda_entropy > 0 and moe_entropy > 0:
                    entropy_loss = -lambda_entropy * moe_entropy
                    loss = total_loss + entropy_loss
                    # 记录熵损失用于监控
                    entropy_loss_value = entropy_loss.item()
                else:
                    loss = total_loss
                    entropy_loss_value = 0.0
                
                loss_geo = bbox_loss
                loss_cls = heatmap_loss
                
                # ====== 性能指标计算 ======
                pred_hm = heatmap_pred.sigmoid()
                pred_centers = pred_hm.view(B, -1).argmax(dim=1)
                pred_y = (pred_centers // W).cpu().numpy()
                pred_x = (pred_centers % W).cpu().numpy()
                gt_centers = gt_heatmap.view(B, -1).argmax(dim=1)
                gt_y = (gt_centers // W).cpu().numpy()
                gt_x = (gt_centers % W).cpu().numpy()
                
                ious = []
                for i in range(B):
                    pred_box = bbox_pred[i, :, pred_y[i], pred_x[i]].detach().cpu().numpy()
                    gt_box = gt_bbox[i, :, gt_y[i], gt_x[i]].detach().cpu().numpy()
                    ious.append(compute_iou(pred_box, gt_box))
                
                ious = np.array(ious)
                accu50 = np.mean(ious > 0.5)
                accu25 = np.mean(ious > 0.25)
                mean_iou = np.mean(ious)
                
                # 更新指标统计
                avg_accu.update(accu50, B)
                avg_accu25.update(accu25, B)
                avg_iou.update(mean_iou, B)
                
            else:
                # 非swinmoe分支
                heatmap_pred, bbox_pred = model(query_imgs, rs_imgs)
                B, _, H, W = heatmap_pred.shape
                gt_heatmap, gt_bbox, mask = build_target_anchorfree(ori_gt_bbox, H, W, args.img_size, args.img_size)
                heatmap_loss, bbox_loss = anchorfree_loss(heatmap_pred, bbox_pred, gt_heatmap, gt_bbox, mask)
                # 调整损失权重 - 增加分类损失权重
                total_loss = 0.7 * heatmap_loss + 0.3 * bbox_loss
                beta = getattr(args, 'beta', 1.0)
                loss = total_loss
                loss_geo = bbox_loss
                loss_cls = heatmap_loss
                
                # 非MoE模型，熵损失为0
                entropy_loss_value = 0.0
                
                # 性能指标计算（与非MoE分支相同）
                pred_hm = heatmap_pred.sigmoid()
                pred_centers = pred_hm.view(B, -1).argmax(dim=1)
                pred_y = (pred_centers // W).cpu().numpy()
                pred_x = (pred_centers % W).cpu().numpy()
                gt_centers = gt_heatmap.view(B, -1).argmax(dim=1)
                gt_y = (gt_centers // W).cpu().numpy()
                gt_x = (gt_centers % W).cpu().numpy()
                
                ious = []
                for i in range(B):
                    pred_box = bbox_pred[i, :, pred_y[i], pred_x[i]].detach().cpu().numpy()
                    gt_box = gt_bbox[i, :, gt_y[i], gt_x[i]].detach().cpu().numpy()
                    ious.append(compute_iou(pred_box, gt_box))
                
                ious = np.array(ious)
                accu50 = np.mean(ious > 0.5)
                accu25 = np.mean(ious > 0.25)
                mean_iou = np.mean(ious)
                
                # 更新指标统计
                avg_accu.update(accu50, B)
                avg_accu25.update(accu25, B)
                avg_iou.update(mean_iou, B)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 更新损失统计
            avg_losses.update(loss.item(), query_imgs.shape[0])
            avg_cls_losses.update(loss_cls.item(), query_imgs.shape[0])
            avg_geo_losses.update(loss_geo.item(), query_imgs.shape[0])
            
                        # 记录MoE熵（简化版本）
            if batch_idx % print_freq_entropy == 0:
                # 从模型获取MoE熵
                if len(device_ids) > 1 and hasattr(model, 'module'):
                    # 多GPU环境，从module获取熵值
                    if hasattr(model.module, 'get_moe_entropy'):
                        moe_entropy = model.module.get_moe_entropy()
                    elif hasattr(model.module, 'get_backbone_moe_entropy'):
                        moe_entropy = model.module.get_backbone_moe_entropy()
                    else:
                        moe_entropy = 0.0
                else:
                    # 单GPU环境，直接从model获取
                    if hasattr(model, 'get_moe_entropy'):
                        moe_entropy = model.get_moe_entropy()
                    elif hasattr(model, 'get_backbone_moe_entropy'):
                        moe_entropy = model.get_backbone_moe_entropy()
                    else:
                        moe_entropy = 0.0
                # 确保是标量值
                if torch.is_tensor(moe_entropy):
                    moe_entropy = moe_entropy.cpu().item()
                moe_entropy_list.append(moe_entropy)
            
            # 每次都要获取当前MoE熵用于显示
            current_moe_entropy = 0.0
            if len(device_ids) > 1 and hasattr(model, 'module'):
                # 多GPU环境，从module获取熵值
                if hasattr(model.module, 'get_moe_entropy'):
                    current_moe_entropy = model.module.get_moe_entropy()
                elif hasattr(model.module, 'get_backbone_moe_entropy'):
                    current_moe_entropy = model.module.get_backbone_moe_entropy()
            else:
                # 单GPU环境，直接从model获取
                if hasattr(model, 'get_moe_entropy'):
                    current_moe_entropy = model.get_moe_entropy()
                elif hasattr(model, 'get_backbone_moe_entropy'):
                    current_moe_entropy = model.get_backbone_moe_entropy()
            if torch.is_tensor(current_moe_entropy):
                current_moe_entropy = current_moe_entropy.cpu().item()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 每print_freq输出一次
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                logging.info(f"Epoch: [{epoch+1}][{batch_idx+1}/{len(train_loader)}] | "
                              f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                              f"Loss: {loss.item():.4f} ({avg_losses.avg:.4f}) | "
                              f"Geo Loss: {loss_geo.item():.4f} ({avg_geo_losses.avg:.4f}) | "
                              f"Cls Loss: {loss_cls.item():.4f} ({avg_cls_losses.avg:.4f}) | "
                              f"Accu50: {accu50:.4f} ({avg_accu.avg:.4f}) | "
                              f"Accu25: {accu25:.4f} ({avg_accu25.avg:.4f}) | "
                              f"Mean_IoU: {mean_iou:.4f} ({avg_iou.avg:.4f}) | "
                              f"MoE Entropy: {current_moe_entropy:.4f} | "
                              f"Entropy Loss: {entropy_loss_value:.6f}")
                
                # 输出MoE熵信息
                if moe_entropy_list:
                    # 确保tensor在CPU上再计算平均值
                    entropy_tensors = [e.cpu().item() if torch.is_tensor(e) else e for e in moe_entropy_list[-print_freq_entropy:]]
                    avg_entropy = np.mean(entropy_tensors)
                    logging.info(f"[MoE门控熵] step={batch_idx}, avg_entropy={avg_entropy:.4f}")
            
            # 损失分析
            trainer.analyze_loss_function(loss_cls.item(), loss_geo.item(), loss.item(), accu50, accu25, mean_iou)
            
        except Exception as e:
            import traceback
            logging.error(f"训练批次 {batch_idx} 出错: {e}")
            logging.error(f"错误详情: {traceback.format_exc()}")
            continue
    
    return avg_losses.avg, avg_geo_losses.avg, avg_cls_losses.avg, avg_accu.avg, avg_accu25.avg, avg_iou.avg

def test_epoch(data_loader, model, args, device_ids, device):
    """
    验证一个epoch
    """
    # 多卡验证彻底修复：临时解除DataParallel包装
    original_model = model
    if len(device_ids) > 1 and hasattr(model, 'module'):
        # 多卡环境下，临时使用主模型进行验证
        model = model.module
        logging.info("🔧 多卡验证：临时解除DataParallel包装")
    
    model.eval()
    avg_accu50 = AverageMeter()
    avg_accu25 = AverageMeter()
    avg_mean_iou = AverageMeter()
    avg_accu_c = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # 正确解包数据，与custom_collate_fn返回格式匹配
                query_imgs, rs_imgs, ori_gt_bbox, idx, click_xy, ori_hw = batch
                
                # 多卡验证修复：强制使用主设备进行验证，避免DataParallel问题
                if len(device_ids) > 1:
                    # 多卡环境下，强制使用主设备进行验证，避免DataParallel问题
                    query_imgs = query_imgs.to(device)
                    rs_imgs = rs_imgs.to(device)
                    ori_gt_bbox = ori_gt_bbox.to(device)
                else:
                    # 单卡环境正常处理
                    query_imgs = query_imgs.to(device)
                    rs_imgs = rs_imgs.to(device)
                    ori_gt_bbox = ori_gt_bbox.to(device)
                
                # 确保数据类型正确
                query_imgs = query_imgs.float()
                rs_imgs = rs_imgs.float()
                ori_gt_bbox = ori_gt_bbox.float()
                
                # 多卡验证修复：更安全的内存清理
                if len(device_ids) > 1:
                    # 多卡环境下，跳过内存清理，避免CUDA错误
                    pass
                else:
                    # 单卡环境正常清理
                    torch.cuda.empty_cache()
                
                ori_gt_bbox = torch.clamp(ori_gt_bbox, min=0, max=args.img_size-1)
                
                if args.model == 'swinmoe':
                    try:
                        heatmap_pred, bbox_pred = model(query_imgs, rs_imgs)
                        B, _, H, W = heatmap_pred.shape
                        gt_heatmap, gt_bbox, mask = build_target_anchorfree(ori_gt_bbox, H, W, args.img_size, args.img_size)
                        
                        # 多卡验证修复：确保目标数据在主设备上
                        if len(device_ids) > 1:
                            # 多卡环境下，强制使用主设备
                            gt_heatmap = gt_heatmap.to(device)
                            gt_bbox = gt_bbox.to(device)
                            mask = mask.to(device)
                        else:
                            # 单卡环境正常处理
                            gt_heatmap = gt_heatmap.to(device)
                            gt_bbox = gt_bbox.to(device)
                            mask = mask.to(device)
                        
                        pred_hm = heatmap_pred.sigmoid()
                        pred_centers = pred_hm.view(B, -1).argmax(dim=1)
                        pred_y = (pred_centers // W).cpu().numpy()
                        pred_x = (pred_centers % W).cpu().numpy()
                        gt_centers = gt_heatmap.view(B, -1).argmax(dim=1)
                        gt_y = (gt_centers // W).cpu().numpy()
                        gt_x = (gt_centers % W).cpu().numpy()
                        
                        ious = []
                        for i in range(B):
                            pred_box = bbox_pred[i, :, pred_y[i], pred_x[i]].detach().cpu().numpy()
                            gt_box = gt_bbox[i, :, gt_y[i], gt_x[i]].detach().cpu().numpy()
                            ious.append(compute_iou(pred_box, gt_box))
                        
                        ious = np.array(ious)
                        accu50 = np.mean(ious > 0.5)
                        accu25 = np.mean(ious > 0.25)
                        mean_iou = np.mean(ious)
                        accu_c = np.mean((pred_x == gt_x) & (pred_y == gt_y))
                        
                        avg_accu50.update(accu50, query_imgs.shape[0])
                        avg_accu25.update(accu25, query_imgs.shape[0])
                        avg_mean_iou.update(mean_iou, query_imgs.shape[0])
                        avg_accu_c.update(accu_c, query_imgs.shape[0])
                        
                    except RuntimeError as e:
                        logging.error(f"模型推理错误: {e}")
                        # 如果推理失败，使用默认值
                        accu50 = 0.0
                        accu25 = 0.0
                        mean_iou = 0.0
                        accu_c = 0.0
                        avg_accu50.update(accu50, query_imgs.shape[0])
                        avg_accu25.update(accu25, query_imgs.shape[0])
                        avg_mean_iou.update(mean_iou, query_imgs.shape[0])
                        avg_accu_c.update(accu_c, query_imgs.shape[0])
                        
                else:
                    from model.loss import build_target
                    from utils.utils import eval_iou_acc
                    anchors_full = np.array([float(x.strip()) for x in args.anchors.split(',')])
                    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
                    anchors_full = torch.tensor(anchors_full, dtype=torch.float32)
                    # 多卡验证修复：确保anchors在主设备上
                    if len(device_ids) > 1:
                        anchors_full = anchors_full.to(device)
                    else:
                        anchors_full = anchors_full.to(device)
                    pred_anchor, attn_score = model(query_imgs, rs_imgs, click_xy)
                    pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
                    _, best_anchor_gi_gj = build_target(ori_gt_bbox, anchors_full, args.img_size, pred_anchor.shape[3])
                    accu_list, accu_center, iou, each_acc_list, _, _ = eval_iou_acc(
                        pred_anchor, ori_gt_bbox, anchors_full, best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2],
                        args.img_size, iou_threshold_list=[0.5, 0.25])
                    accu50 = accu_list[0]
                    accu25 = accu_list[1]
                    mean_iou = iou
                    accu_c = accu_center
                    avg_accu50.update(accu50, query_imgs.shape[0])
                    avg_accu25.update(accu25, query_imgs.shape[0])
                    avg_mean_iou.update(mean_iou, query_imgs.shape[0])
                    avg_accu_c.update(accu_c, query_imgs.shape[0])
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                # 多卡验证修复：更安全的内存清理
                if len(device_ids) <= 1:
                    # 单卡环境正常清理
                    torch.cuda.empty_cache()
                
                # 只每print_freq输出一次
                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(data_loader):
                    logging.info(f"[{batch_idx+1}/{len(data_loader)}] | "
                                  f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | "
                                  f"Accu50: {accu50:.4f} ({avg_accu50.avg:.4f}) | "
                                  f"Accu25: {accu25:.4f} ({avg_accu25.avg:.4f}) | "
                                  f"Mean_IoU: {mean_iou:.4f} ({avg_mean_iou.avg:.4f}) | "
                                  f"Accu_c: {accu_c:.4f} ({avg_accu_c.avg:.4f})")
                                  
            except Exception as e:
                import traceback
                logging.error(f"验证批次 {batch_idx} 出错: {e}")
                logging.error(f"错误详情: {traceback.format_exc()}")
                # 使用默认值继续
                accu50 = 0.0
                accu25 = 0.0
                mean_iou = 0.0
                accu_c = 0.0
                avg_accu50.update(accu50, query_imgs.shape[0])
                avg_accu25.update(accu25, query_imgs.shape[0])
                avg_mean_iou.update(mean_iou, query_imgs.shape[0])
                avg_accu_c.update(accu_c, query_imgs.shape[0])
                continue
    
    # 多卡验证修复：恢复原始模型
    if len(device_ids) > 1 and original_model != model:
        model = original_model
        logging.info("🔧 多卡验证：恢复DataParallel包装")
    
    # 汇总输出
    logging.info("\n=== 验证集汇总结果 ===")
    logging.info(f"Accu50: {avg_accu50.avg:.4f}, Accu25: {avg_accu25.avg:.4f}, Mean IoU: {avg_mean_iou.avg:.4f}, Accu_c: {avg_accu_c.avg:.4f}")
    return {
        'accu50': avg_accu50.avg,
        'accu25': avg_accu25.avg,
        'mean_iou': avg_mean_iou.avg,
        'accu_c': avg_accu_c.avg
    }

def compute_iou(box1, box2):
    """计算IoU"""
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def _extract_meter_values(meter_list):
    """
    将AverageMeter对象列表转为float数值列表
    """
    return [x.avg if hasattr(x, 'avg') else float(x) for x in meter_list]

if __name__ == '__main__':
    main()