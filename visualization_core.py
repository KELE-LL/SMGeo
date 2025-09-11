import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.ndimage import gaussian_filter

# =====================
# 统一可视化核心函数
# =====================

def to_pixel_point(coord, coord_type, img_size, feat_size):
    """
    按指定类型将点坐标转换为像素坐标 (x_pixel, y_pixel)
    @param coord: (y, x) or (x, y)
    @param coord_type: 'pixel'/'feat'/'norm'
    @return: (x_pixel, y_pixel)
    """
    img_H, img_W = img_size
    hH, hW = feat_size
    arr = np.asarray(coord).flatten()
    if coord_type == 'norm':
        print(f"[DEBUG] 输入点 {arr} 类型: norm (归一化)")
        return arr[0] * img_W, arr[1] * img_H
    elif coord_type == 'feat':
        print(f"[DEBUG] 输入点 {arr} 类型: feat (特征图)")
        return (arr[1] + 0.5) * (img_W / hW), (arr[0] + 0.5) * (img_H / hH)
    elif coord_type == 'pixel':
        print(f"[DEBUG] 输入点 {arr} 类型: pixel (像素)")
        return arr[0], arr[1]
    else:
        raise ValueError(f'Unknown coord_type: {coord_type}')

def to_pixel_box(box, box_type, img_size, feat_size):
    """
    按指定类型将框坐标转换为像素坐标 (x1, y1, x2, y2)
    支持(cx,cy,w,h)或(x1,y1,x2,y2)格式
    @param box: [cx,cy,w,h] or [x1,y1,x2,y2]
    @param box_type: 'pixel'/'feat'/'norm'
    @return: [x1, y1, x2, y2] (像素坐标)
    """
    img_H, img_W = img_size
    hH, hW = feat_size
    arr = np.asarray(box).flatten()
    if box_type == 'norm':
        print(f"[DEBUG] 输入框 {arr} 类型: norm (归一化(cx,cy,w,h))")
        cx, cy, w, h = arr
        cx, w = cx * img_W, w * img_W
        cy, h = cy * img_H, h * img_H
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]
    elif box_type == 'feat':
        print(f"[DEBUG] 输入框 {arr} 类型: feat (特征图(cx,cy,w,h))")
        cx, cy, w, h = arr
        x1 = (cx - w/2 + 0.5) * (img_W / hW)
        y1 = (cy - h/2 + 0.5) * (img_H / hH)
        x2 = (cx + w/2 + 0.5) * (img_W / hW)
        y2 = (cy + h/2 + 0.5) * (img_H / hH)
        return [x1, y1, x2, y2]
    elif box_type == 'pixel':
        print(f"[DEBUG] 输入框 {arr} 类型: pixel (像素(cx,cy,w,h))")
        cx, cy, w, h = arr
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]
    else:
        raise ValueError(f'Unknown box_type: {box_type}')

def assert_pixel_coord(coord, img_size, name):
    """
    断言输入为像素坐标且在图片范围内
    @param coord: 标量、1D或2D坐标
    @param img_size: (H, W)
    @param name: 名称
    """
    arr = np.asarray(coord).flatten()
    img_H, img_W = img_size
    assert np.all((arr >= 0) & (arr <= max(img_H, img_W))), f"{name}像素坐标越界: {arr}，图片尺寸: {img_size}"
    print(f"[ASSERT] {name}: {arr} in [0, {img_W}], [0, {img_H}]")

def draw_visualization(
    query_img, sat_img, mat_clickxy,
    pred_heatmap, gt_heatmap,
    pred_box, gt_box,
    pred_center, gt_center,
    bbox_pred_values,
    save_dir, sample_idx,
    img_size,
    heatmap_size,
    extra_info=None
):
    """
    可视化入口：所有输入（框、中心点、点击点）都必须是像素坐标，入口做断言校验，内部直接绘图。
    @param query_img: (C,H,W) tensor or ndarray, 已反归一化
    @param sat_img: (C,H,W) tensor or ndarray, 已反归一化
    @param mat_clickxy: (2,) 像素坐标
    """
    # 优化热力图显示效果
    def enhance_heatmap(heatmap, gamma=2.0, target_size=512):
        """增强热力图对比度，使其更加清晰和平滑"""
        
        # 1. 上采样到更高分辨率
        import cv2
        h, w = heatmap.shape
        heatmap_upsampled = cv2.resize(heatmap, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        # 2. 应用更强的多次高斯平滑，使热力图更自然平滑
        heatmap_smooth = gaussian_filter(heatmap_upsampled, sigma=4.0)
        heatmap_smooth = gaussian_filter(heatmap_smooth, sigma=3.0)
        heatmap_smooth = gaussian_filter(heatmap_smooth, sigma=2.0)
        heatmap_smooth = gaussian_filter(heatmap_smooth, sigma=1.5)
        
        # 3. 应用更强的gamma校正增强对比度
        enhanced = np.power(heatmap_smooth, gamma)
        
        # 4. 归一化到0-1范围
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
        
        # 5. 应用更强的边缘模糊，让热力图更柔和
        enhanced = gaussian_filter(enhanced, sigma=2.0)
        
        # 6. 应用适度的对比度增强，让温度差异更明显但保持舒适
        enhanced = np.clip(enhanced * 1.3, 0, 1)
        
        return enhanced
    
    # 增强预测热力图和真实热力图
    pred_heatmap_enhanced = enhance_heatmap(pred_heatmap, gamma=3.0, target_size=512)
    gt_heatmap_enhanced = enhance_heatmap(gt_heatmap, gamma=2.5, target_size=512)
    os.makedirs(save_dir, exist_ok=True)
    # 反归一化
    if torch.is_tensor(query_img):
        query_img = query_img.cpu().numpy().transpose(1,2,0)
    if torch.is_tensor(sat_img):
        sat_img = sat_img.cpu().numpy().transpose(1,2,0)
    query_img = np.clip(query_img, 0, 1)
    sat_img = np.clip(sat_img, 0, 1)
    # 支持非正方形
    if isinstance(img_size, int):
        img_H = img_W = img_size
    else:
        img_H, img_W = img_size
    if isinstance(heatmap_size, int):
        hH = hW = heatmap_size
    else:
        hH, hW = heatmap_size

    # =====================
    # 入口断言校验
    # =====================
    if mat_clickxy is not None:
        assert_pixel_coord(mat_clickxy, (img_H, img_W), 'mat_clickxy')
    if gt_box is not None:
        assert_pixel_coord(gt_box, (img_H, img_W), 'gt_box')
    if pred_box is not None:
        assert_pixel_coord(pred_box, (img_H, img_W), 'pred_box')
    if gt_center is not None:
        assert_pixel_coord(gt_center, (img_H, img_W), 'gt_center')
    if pred_center is not None:
        assert_pixel_coord(pred_center, (img_H, img_W), 'pred_center')

    # =====================
    # 可视化
    # =====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # 1. 查询图像
    ax = axes[0,0]
    ax.imshow(query_img, extent=[0, img_W, img_H, 0])
    legend_handles = []
    # 只保留图例为红色点，不再显示点击点
    red_dot = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Query Click Region')
    legend_handles.append(red_dot)
    ax.set_title('Query Image (Ground/Drone)', fontsize=14, fontweight='bold')
    ax.axis([0, img_W, img_H, 0])
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=12)
    # 2. 卫星图像
    ax = axes[0,1]
    ax.imshow(sat_img, extent=[0, img_W, img_H, 0])
    ax.set_title('Satellite Image', fontsize=14, fontweight='bold')
    ax.axis([0, img_W, img_H, 0])
    # 3. 预测热力图
    ax = axes[0,2]
    # 使用更高分辨率的增强热力图
    h_enhanced, w_enhanced = pred_heatmap_enhanced.shape
    im1 = ax.imshow(pred_heatmap_enhanced, cmap='viridis', vmin=0, vmax=1, extent=[0, w_enhanced, h_enhanced, 0])
    legend_handles = []
    if pred_center is not None:
        # 调整中心点坐标到增强后的尺寸
        pred_x_enhanced = pred_center[0] * (w_enhanced / hW)
        pred_y_enhanced = pred_center[1] * (h_enhanced / hH)
        h2 = ax.scatter(pred_x_enhanced, pred_y_enhanced, c='lime', marker='+', s=150, linewidths=2, label='Pred Center', zorder=10)
        legend_handles.append(h2)
    ax.set_title('Predicted Heatmap', fontsize=14, fontweight='bold')
    ax.axis([0, w_enhanced, h_enhanced, 0])
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=12)
    # 4. 真实热力图
    ax = axes[1,0]
    # 使用更高分辨率的增强热力图
    h_enhanced, w_enhanced = gt_heatmap_enhanced.shape
    im2 = ax.imshow(gt_heatmap_enhanced, cmap='viridis', vmin=0, vmax=1, extent=[0, w_enhanced, h_enhanced, 0])
    ax.set_title('Ground Truth Heatmap', fontsize=14, fontweight='bold')
    ax.axis([0, w_enhanced, h_enhanced, 0])
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    # 5. 检测结果对比
    ax = axes[1,1]
    ax.imshow(sat_img, extent=[0, img_W, img_H, 0])
    legend_handles = []
    # 真实框
    if gt_box is not None:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
        gt_w_img = gt_x2 - gt_x1
        gt_h_img = gt_y2 - gt_y1
        rect_gt = Rectangle((gt_x1, gt_y1), gt_w_img, gt_h_img, linewidth=3, edgecolor='red', facecolor='none', label='Ground Truth', linestyle='-')
        ax.add_patch(rect_gt)
        legend_handles.append(rect_gt)
        # 真实框中心点
        gt_cx = (gt_x1 + gt_x2) / 2
        gt_cy = (gt_y1 + gt_y2) / 2
        h3 = ax.scatter(gt_cx, gt_cy, c='red', marker='x', s=100, label='GT Center', zorder=10)
        legend_handles.append(h3)
    # 预测框
    if pred_box is not None:
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
        pred_w_img = pred_x2 - pred_x1
        pred_h_img = pred_y2 - pred_y1
        rect_pred = Rectangle((pred_x1, pred_y1), pred_w_img, pred_h_img, linewidth=3, edgecolor='green', facecolor='none', label='Prediction', linestyle='--')
        ax.add_patch(rect_pred)
        legend_handles.append(rect_pred)
        # 预测框中心点
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        h4 = ax.scatter(pred_cx, pred_cy, c='green', marker='+', s=100, label='Pred Center', zorder=10)
        legend_handles.append(h4)
    ax.set_title('Detection Results Comparison', fontsize=14, fontweight='bold')
    ax.axis([0, img_W, img_H, 0])
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=12)
    # 6. 输出分布分析
    ax = axes[1,2]
    hm_flat = pred_heatmap.flatten()
    bbox_flat = bbox_pred_values.flatten() if bbox_pred_values is not None else np.array([])
    ax2 = ax.twinx()
    n1, bins1, _ = ax.hist(hm_flat, bins=50, alpha=0.7, color='blue', label='Heatmap Values')
    ax.set_xlabel('Heatmap/BBox Value', fontsize=10)
    ax.set_ylabel('Heatmap Frequency', color='blue', fontsize=10)
    ax.tick_params(axis='y', labelcolor='blue')
    if bbox_flat.size > 0:
        n2, bins2, _ = ax2.hist(bbox_flat, bins=50, alpha=0.7, color='red', label='BBox Values')
        ax2.set_ylabel('BBox Frequency', color='red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Output Distributions', fontsize=12, fontweight='bold')
    hm_mean, hm_std = hm_flat.mean(), hm_flat.std()
    bbox_mean, bbox_std = (bbox_flat.mean(), bbox_flat.std()) if bbox_flat.size > 0 else (0,0)
    stats_text = f'Heatmap: μ={hm_mean:.3f}, σ={hm_std:.3f}\nBBox: μ={bbox_mean:.3f}, σ={bbox_std:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    
    # 保存综合分析图
    save_path = os.path.join(save_dir, f'sample_{sample_idx}_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分别保存每张图，按原图尺寸清晰展示
    save_individual_images(
        query_img, sat_img, pred_heatmap_enhanced, gt_heatmap_enhanced,
        pred_box, gt_box, pred_center, gt_center,
        save_dir, sample_idx, img_size
    )
    
    details_file = os.path.join(save_dir, f'sample_{sample_idx}_details.txt')
    with open(details_file, 'w', encoding='utf-8') as f:
        f.write(f"样本 {sample_idx} 详细信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"点击点: {mat_clickxy}\n")
        f.write(f"真实中心: {gt_center}\n")
        f.write(f"预测中心: {pred_center}\n")
        f.write(f"真实框: {gt_box}\n")
        f.write(f"预测框: {pred_box}\n")
        f.write(f"热力图均值: {hm_mean:.4f}, 方差: {hm_std:.4f}\n")
        f.write(f"BBox均值: {bbox_mean:.4f}, 方差: {bbox_std:.4f}\n")
        if extra_info is not None:
            f.write(str(extra_info)+"\n")
    print(f"✓ 样本{sample_idx}可视化已保存: {save_path}")
    print(f"✓ 样本{sample_idx}单独图片已保存到: {save_dir}") 

def save_individual_images(
    query_img, sat_img, pred_heatmap, gt_heatmap,
    pred_box, gt_box, pred_center, gt_center,
    save_dir, sample_idx, img_size
):
    """分别保存每张图片，按原图尺寸清晰展示"""
    import cv2
    
    # 获取原图尺寸
    if isinstance(img_size, int):
        img_H = img_W = img_size
    else:
        img_H, img_W = img_size
    
    # 1. 保存查询图像（保持原图尺寸）
    query_save_path = os.path.join(save_dir, f'sample_{sample_idx}_query.png')
    # 不进行缩放，直接保存原图尺寸，不添加任何标记
    query_img_original = query_img
    cv2.imwrite(query_save_path, query_img_original * 255)
    
    # 2. 保存卫星图像（保持原图尺寸）
    sat_save_path = os.path.join(save_dir, f'sample_{sample_idx}_satellite.png')
    # 不进行缩放，直接保存原图尺寸
    sat_img_original = sat_img
    cv2.imwrite(sat_save_path, sat_img_original * 255)
    
    # 3. 保存预测热力图
    pred_heatmap_save_path = os.path.join(save_dir, f'sample_{sample_idx}_pred_heatmap.png')
    # 使用plasma颜色映射，更专业的热力图效果
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # 创建高分辨率热力图
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(pred_heatmap, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    
    # 添加中心点标记（只显示绿色十字，不添加文字）
    if pred_center is not None:
        h_heatmap, w_heatmap = pred_heatmap.shape
        pred_x_heatmap = pred_center[0] * (w_heatmap / img_W)
        pred_y_heatmap = pred_center[1] * (h_heatmap / img_H)
        ax.scatter(pred_x_heatmap, pred_y_heatmap, c='lime', marker='+', s=200, linewidths=2, zorder=10)
    
    ax.set_title(f'Predicted Heatmap - Sample {sample_idx}', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(pred_heatmap_save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 4. 保存真实热力图
    gt_heatmap_save_path = os.path.join(save_dir, f'sample_{sample_idx}_gt_heatmap.png')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(gt_heatmap, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    
    # 添加中心点标记（只显示红色X，不添加文字）
    if gt_center is not None:
        h_heatmap, w_heatmap = gt_heatmap.shape
        gt_x_heatmap = gt_center[0] * (w_heatmap / img_W)
        gt_y_heatmap = gt_center[1] * (h_heatmap / img_H)
        ax.scatter(gt_x_heatmap, gt_y_heatmap, c='red', marker='x', s=200, linewidths=2, zorder=10)
    
    ax.set_title(f'Ground Truth Heatmap - Sample {sample_idx}', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(gt_heatmap_save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 5. 保存检测结果对比图
    detection_save_path = os.path.join(save_dir, f'sample_{sample_idx}_detection.png')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(sat_img, extent=[0, img_W, img_H, 0])
    
    # 绘制真实框
    if gt_box is not None:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        rect_gt = plt.Rectangle((gt_x1, gt_y1), gt_w, gt_h, linewidth=3, 
                               edgecolor='red', facecolor='none', linestyle='-', label='Ground Truth')
        ax.add_patch(rect_gt)
        
        # 真实框中心点
        gt_cx = (gt_x1 + gt_x2) / 2
        gt_cy = (gt_y1 + gt_y2) / 2
        ax.scatter(gt_cx, gt_cy, c='red', marker='x', s=200, linewidths=3, zorder=10, label='GT Center')
    
    # 绘制预测框
    if pred_box is not None:
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        rect_pred = plt.Rectangle((pred_x1, pred_y1), pred_w, pred_h, linewidth=3, 
                                 edgecolor='green', facecolor='none', linestyle='--', label='Prediction')
        ax.add_patch(rect_pred)
        
        # 预测框中心点
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        ax.scatter(pred_cx, pred_cy, c='green', marker='+', s=200, linewidths=3, zorder=10, label='Pred Center')
    
    ax.set_title(f'Detection Results - Sample {sample_idx}', fontsize=16, fontweight='bold', pad=20)
    ax.axis([0, img_W, img_H, 0])
    ax.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(detection_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - 查询图像: {query_save_path}")
    print(f"  - 卫星图像: {sat_save_path}")
    print(f"  - 预测热力图: {pred_heatmap_save_path}")
    print(f"  - 真实热力图: {gt_heatmap_save_path}")
    print(f"  - 检测结果: {detection_save_path}") 