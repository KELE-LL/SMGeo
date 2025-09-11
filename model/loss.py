# -*- coding:utf8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import bbox_iou, xyxy2xywh

def adjust_learning_rate(args, optimizer, i_iter):
    """
    @function adjust_learning_rate
    @desc 动态调整学习率 - 修改为更平缓的衰减策略
    @param args: 参数对象，需包含lr
    @param optimizer: 优化器
    @param i_iter: 当前迭代次数
    """
    # 修改学习率衰减策略，使其更加平缓
    # 原来: lr = args.lr * ((0.1) ** (i_iter // 10)) - 衰减过快
    # 现在: 使用更平缓的衰减，每20个epoch衰减到原来的0.5倍
    decay_factor = 0.5 ** (i_iter // 20)
    lr = args.lr * decay_factor
    print(("lr", lr))
    for param_idx, param in enumerate(optimizer.param_groups):
        param['lr'] = lr

# the shape of the target is (batch_size, anchor_count, 5, grid_wh, grid_wh)
def yolo_loss(predictions, gt_bboxes, anchors_full, best_anchor_gi_gj, image_wh):
    """
    @function yolo_loss
    @desc YOLO anchor-based损失函数
    @param predictions: [B, anchor_count, 5, H, W]
    @param gt_bboxes: [B, 4]
    @param anchors_full: [anchor_count, 2]
    @param best_anchor_gi_gj: [B, 3]
    @param image_wh: int
    @return: loss_bbox, loss_confidence
    """
    batch_size, grid_stride = predictions.shape[0], image_wh // predictions.shape[3]
    best_anchor, gi, gj = best_anchor_gi_gj[:, 0], best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2]
    scaled_anchors = anchors_full / grid_stride
    mseloss = torch.nn.MSELoss(reduction='mean')
    celoss_confidence = torch.nn.CrossEntropyLoss(reduction='mean')
    #celoss_cls = torch.nn.CrossEntropyLoss(size_average=True)

    selected_predictions = predictions[range(batch_size), best_anchor, :, gj, gi]

    #---bbox loss---
    pred_bboxes = torch.zeros_like(gt_bboxes)
    pred_bboxes[:, 0:2] = selected_predictions[:, 0:2].sigmoid()
    pred_bboxes[:, 2:4] = selected_predictions[:, 2:4]
    
    loss_x = mseloss(pred_bboxes[:,0], gt_bboxes[:,0])
    loss_y = mseloss(pred_bboxes[:,1], gt_bboxes[:,1])
    loss_w = mseloss(pred_bboxes[:,2], gt_bboxes[:,2])
    loss_h = mseloss(pred_bboxes[:,3], gt_bboxes[:,3])

    loss_bbox = loss_x + loss_y + loss_w + loss_h

    #---confidence loss---
    pred_confidences = predictions[:,:,4,:,:]
    gt_confidences = torch.zeros_like(pred_confidences)
    gt_confidences[range(batch_size), best_anchor, gj, gi] = 1
    pred_confidences, gt_confidences = pred_confidences.reshape(batch_size, -1), \
                    gt_confidences.reshape(batch_size, -1)
    loss_confidence = celoss_confidence(pred_confidences, gt_confidences.max(1)[1])

    return loss_bbox, loss_confidence

#target_coord:batch_size, 5
def build_target(ori_gt_bboxes, anchors_full, image_wh, grid_wh):
    """
    @function build_target
    @desc 构建YOLO anchor-based目标
    @param ori_gt_bboxes: [B, 5]
    @param anchors_full: [anchor_count, 2]
    @param image_wh: int
    @param grid_wh: int
    @return: target_coord, best_anchor_gi_gj
    """
    #the default value of coord_dim is 5
    batch_size, coord_dim, grid_stride, anchor_count = ori_gt_bboxes.shape[0], ori_gt_bboxes.shape[1], image_wh//grid_wh, anchors_full.shape[0]
    
    gt_bboxes = xyxy2xywh(ori_gt_bboxes)
    gt_bboxes = (gt_bboxes/image_wh) * grid_wh
    scaled_anchors = anchors_full/grid_stride

    gxy = gt_bboxes[:, 0:2]
    gwh = gt_bboxes[:, 2:4]
    gij = gxy.long()

    #get the best anchor for each target bbox
    gt_bboxes_tmp, scaled_anchors_tmp = torch.zeros_like(gt_bboxes), torch.zeros((anchor_count, coord_dim), device=gt_bboxes.device)
    gt_bboxes_tmp[:, 2:4] = gwh
    gt_bboxes_tmp = gt_bboxes_tmp.unsqueeze(1).repeat(1, anchor_count, 1).view(-1, coord_dim)
    scaled_anchors_tmp[:, 2:4] = scaled_anchors
    scaled_anchors_tmp = scaled_anchors_tmp.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, coord_dim)
    anchor_ious = bbox_iou(gt_bboxes_tmp, scaled_anchors_tmp).view(batch_size, -1)
    best_anchor=torch.argmax(anchor_ious, dim=1)
    
    twh = torch.log(gwh / scaled_anchors[best_anchor] + 1e-16)
    #print((gxy.dtype, gij.dtype, twh.dtype, gwh.dtype, scaled_anchors.dtype, 'inner'))
    #print((gxy.shape, gij.shape, twh.shape, gwh.shape), flush=True)
    #print(('gxy,gij,twh', gxy, gij, twh), flush=True)
    return torch.cat((gxy - gij, twh), 1), torch.cat((best_anchor.unsqueeze(1), gij), 1)

def build_target_anchorfree(gt_bboxes, feat_h, feat_w, img_h, img_w, sigma=2):
    """
    @function build_target_anchorfree
    @desc 将gt bbox映射到特征图坐标，生成中心点热力图和bbox参数
    @param gt_bboxes: [B, 4]，原图坐标
    @param feat_h: 特征图高
    @param feat_w: 特征图宽
    @param img_h: 原图高
    @param img_w: 原图宽
    @param sigma: 高斯半径
    @return: heatmap [B, 1, H, W], bbox_target [B, 4, H, W], mask [B, 1, H, W]
    """
    # print("DEBUG build_target_anchorfree gt_bboxes shape:", gt_bboxes.shape)
    # for i, b in enumerate(gt_bboxes):
    #     print(f"  build_target_anchorfree sample {i} bbox: {b}, len={len(b)}")
    B = gt_bboxes.shape[0]
    heatmap = torch.zeros((B, 1, feat_h, feat_w), device=gt_bboxes.device)
    bbox_target = torch.zeros((B, 4, feat_h, feat_w), device=gt_bboxes.device)
    mask = torch.zeros((B, 1, feat_h, feat_w), device=gt_bboxes.device)
    for i in range(B):
        x1, y1, x2, y2 = gt_bboxes[i]
        cx = (x1 + x2) / 2 / img_w * feat_w
        cy = (y1 + y2) / 2 / img_h * feat_h
        w = (x2 - x1) / img_w * feat_w
        h = (y2 - y1) / img_h * feat_h
        cx_int, cy_int = int(cx), int(cy)
        # 生成高斯热力图
        for dx in range(-sigma*2, sigma*2+1):
            for dy in range(-sigma*2, sigma*2+1):
                nx, ny = cx_int + dx, cy_int + dy
                if 0 <= nx < feat_w and 0 <= ny < feat_h:
                    d2 = dx*dx + dy*dy
                    value = math.exp(-d2/(2*sigma**2))
                    heatmap[i, 0, ny, nx] = max(heatmap[i, 0, ny, nx], value)
                    bbox_target[i, 0, ny, nx] = cx - nx
                    bbox_target[i, 1, ny, nx] = cy - ny
                    bbox_target[i, 2, ny, nx] = w
                    bbox_target[i, 3, ny, nx] = h
                    mask[i, 0, ny, nx] = 1
    return heatmap, bbox_target, mask

def anchorfree_loss(pred_heatmap, pred_bbox, gt_heatmap, gt_bbox, mask):
    """
    @function anchorfree_loss - 重新设计版本
    @desc 解决损失数值范围不平衡和下降缓慢的问题
    @param pred_heatmap: [B, 1, H, W]
    @param pred_bbox: [B, 4, H, W]
    @param gt_heatmap: [B, 1, H, W]
    @param gt_bbox: [B, 4, H, W]
    @param mask: [B, 1, H, W]
    @return: heatmap_loss, bbox_loss
    """
    # 确保所有张量都在同一设备上
    device = pred_heatmap.device
    gt_heatmap = gt_heatmap.to(device)
    pred_bbox = pred_bbox.to(device)
    gt_bbox = gt_bbox.to(device)
    mask = mask.to(device)
    
    # ====== 重新设计分类损失 ======
    pred_hm = pred_heatmap.sigmoid().clamp(1e-4, 1 - 1e-4)
    
    # 1. 简化Focal Loss，减少数值范围
    gamma = 2.0
    alpha = 0.25
    beta = 0.75
    
    pos_inds = gt_heatmap.eq(1)
    neg_inds = gt_heatmap.lt(1)
    
    # 正样本损失 - 简化版本
    if pos_inds.sum() > 0:
        loss_pos = -torch.log(pred_hm)[pos_inds] * torch.pow(1 - pred_hm[pos_inds], gamma) * alpha
        pos_loss = loss_pos.mean()  # 使用mean而不是sum
    else:
        pos_loss = torch.tensor(0.0, device=device)
    
    # 负样本损失 - 大幅简化，移除导致数值过大的项
    if neg_inds.sum() > 0:
        # 移除 torch.pow(1 - gt_heatmap[neg_inds], 4) 这个导致数值过大的项
        loss_neg = -torch.log(1 - pred_hm)[neg_inds] * torch.pow(pred_hm[neg_inds], gamma) * beta
        neg_loss = loss_neg.mean()  # 使用mean而不是sum
    else:
        neg_loss = torch.tensor(0.0, device=device)
    
    # 分类损失 - 平衡正负样本
    heatmap_loss = pos_loss + neg_loss
    
    # 2. 归一化分类损失到合理范围
    heatmap_loss = torch.clamp(heatmap_loss, 0, 10.0)  # 限制最大值
    
    # ====== 重新设计几何损失 ======
    # 分离中心点和宽高
    pred_center = pred_bbox[:, :2, :, :]  # [x, y]
    pred_wh = pred_bbox[:, 2:, :, :]      # [w, h]
    gt_center = gt_bbox[:, :2, :, :]
    gt_wh = gt_bbox[:, 2:, :, :]

    # mask for center and wh
    mask_center = mask.expand_as(pred_center)  # [B, 2, H, W]
    mask_wh = mask.expand_as(pred_wh)          # [B, 2, H, W]

    # 3. 改进几何损失计算
    if mask_center.sum() > 0:
        # 中心点损失 - 使用L1损失，数值更稳定
        center_loss = F.l1_loss(pred_center * mask_center, gt_center * mask_center, reduction='sum') / (mask_center.sum() + 1e-4)
        # 归一化到合理范围
        center_loss = torch.clamp(center_loss, 0, 5.0)
    else:
        center_loss = torch.tensor(0.0, device=device)

    if mask_wh.sum() > 0:
        # 宽高损失 - 使用L1损失
        wh_loss = F.l1_loss(pred_wh * mask_wh, gt_wh * mask_wh, reduction='sum') / (mask_wh.sum() + 1e-4)
        # 归一化到合理范围
        wh_loss = torch.clamp(wh_loss, 0, 5.0)
    else:
        wh_loss = torch.tensor(0.0, device=device)

    # 4. 组合几何损失 - 调整权重比例
    bbox_loss = 0.7 * center_loss + 0.3 * wh_loss
    
    # 5. 最终归一化，确保两个损失在相似范围内
    # 目标：让分类损失和几何损失都在0-10范围内
    heatmap_loss = heatmap_loss * 2.0  # 适当放大分类损失
    bbox_loss = bbox_loss * 3.0        # 适当放大几何损失
    
    return heatmap_loss, bbox_loss

def eval_anchorfree_acc(heatmap_pred, bbox_pred, gt_heatmap, gt_bbox, mask, iou_threshold=0.5):
    """
    @function eval_anchorfree_acc
    @desc Anchor-Free评估：中心点准确率、IoU、Accu_c
    @param heatmap_pred: [B, 1, H, W]，预测中心点热力图
    @param bbox_pred: [B, 4, H, W]，预测bbox参数
    @param gt_heatmap: [B, 1, H, W]，标签中心点热力图
    @param gt_bbox: [B, 4, H, W]，标签bbox参数
    @param mask: [B, 1, H, W]，正样本mask
    @param iou_threshold: IoU阈值，默认0.5
    @return: accu, mean_iou, accu_c
    """
    pred_hm = heatmap_pred.sigmoid()
    B, _, H, W = pred_hm.shape
    pred_centers = (pred_hm.view(B, -1) > 0.3).float()
    # 若无点超过阈值，取最大值点
    for i in range(B):
        if pred_centers[i].sum() == 0:
            pred_centers[i][pred_hm.view(B, -1)[i].argmax()] = 1
    pred_yx = pred_centers.nonzero(as_tuple=False)
    # 只保留每个样本的第一个中心点
    pred_y = (pred_yx[:, 1] // W).cpu().numpy()
    pred_x = (pred_yx[:, 1] % W).cpu().numpy()
    gt_centers = gt_heatmap.view(B, -1).argmax(dim=1)
    gt_y = (gt_centers // W).cpu().numpy()
    gt_x = (gt_centers % W).cpu().numpy()
    accu_c = np.mean((pred_x == gt_x) & (pred_y == gt_y))
    ious = []
    for i in range(B):
        pred_box = bbox_pred[i, :, pred_y[i], pred_x[i]].detach().cpu().numpy()  # [4]
        gt_box = gt_bbox[i, :, gt_y[i], gt_x[i]].detach().cpu().numpy()          # [4]
        # 需实现compute_iou函数
        ious.append(compute_iou(pred_box, gt_box))
    mean_iou = np.mean(ious)
    accu = np.mean([iou > iou_threshold for iou in ious])
    return float(accu), float(mean_iou), float(accu_c)

def compute_iou(box1, box2):
    """
    @function compute_iou
    @desc 计算两个bbox的IoU，box格式[x, y, w, h]，中心点+宽高
    @param box1: [4]
    @param box2: [4]
    @return: iou
    """
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

# ========== MoE门控熵正则 ==========
def moe_entropy_loss(gate_entropy, entropy_weight=0.01):
    """
    @function moe_entropy_loss
    @desc MoE门控分布的熵正则损失，鼓励门控分布均匀
    @param gate_entropy: 门控分布熵（标量或张量）
    @param entropy_weight: 熵正则权重
    @return: 熵正则损失
    """
    return -entropy_weight * gate_entropy

# 用法示例：
# heatmap_loss, bbox_loss = anchorfree_loss(...)
# moe_entropy = ... # 从MoE主干forward返回
# total_loss = heatmap_loss + bbox_loss + moe_entropy_loss(moe_entropy, entropy_weight=0.01)