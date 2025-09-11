#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file pair_bbox_test.py
@desc 生成Query和Reference图像对，验证bbox对应关系
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def load_dataset_info(data_root='data/CVOGL_DroneAerial'):
    """加载数据集信息"""
    train_path = os.path.join(data_root, 'CVOGL_DroneAerial_train.pth')
    val_path = os.path.join(data_root, 'CVOGL_DroneAerial_val.pth')
    
    train_data = torch.load(train_path, weights_only=True)
    val_data = torch.load(val_path, weights_only=True)
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    
    return train_data, val_data

def parse_data_item(item):
    """解析数据项"""
    sample_id = item[0]
    query_img_name = item[1]
    ref_img_name = item[2]
    query_size = item[3]  # (W, H)
    ref_size = item[4]    # (W, H)
    query_bbox = item[5]  # [xmin, ymin, xmax, ymax] - Query图上的水平目标框
    ref_polygon = item[6] # [x1, y1, x2, y2, x3, y3, x4, y4] - Reference图上的多边形边界
    class_name = item[7]
    
    return {
        'sample_id': sample_id,
        'query_img_name': query_img_name,
        'ref_img_name': ref_img_name,
        'query_size': query_size,
        'ref_size': ref_size,
        'query_bbox': query_bbox,
        'ref_polygon': ref_polygon,
        'class_name': class_name
    }

def convert_polygon_to_bbox(polygon):
    """将8值多边形转换为4值bbox"""
    if len(polygon) == 8:
        x_coords = polygon[::2]  # 所有x坐标
        y_coords = polygon[1::2]  # 所有y坐标
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        return [xmin, ymin, xmax, ymax]
    elif len(polygon) == 4:
        return polygon
    else:
        raise ValueError(f"多边形坐标格式错误: {polygon}")

def draw_bbox_on_image(img, bbox, label, color=(255, 0, 0), thickness=3):
    """在图像上绘制bbox和中心点标记"""
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = bbox
    
    # 绘制矩形框
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=thickness)
    
    # 计算框的中心点
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    
    # 在中心点绘制红色的"×"符号
    cross_size = 8  # "×"符号的大小
    # 绘制左上到右下的斜线
    draw.line([(center_x - cross_size, center_y - cross_size), 
               (center_x + cross_size, center_y + cross_size)], 
              fill=color, width=2)
    # 绘制右上到左下的斜线
    draw.line([(center_x + cross_size, center_y - cross_size), 
               (center_x - cross_size, center_y + cross_size)], 
              fill=color, width=2)
    
    # 绘制标签
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((xmin, ymin-25), label, fill=color, font=font)
    return img

def test_sample_pair(data_root, sample_info, save_dir='pair_test_results'):
    """测试单个样本的Query和Reference图像对"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建图像路径
    query_img_path = os.path.join(data_root, 'query', sample_info['query_img_name'])
    ref_img_path = os.path.join(data_root, 'satellite', sample_info['ref_img_name'])
    
    # 检查文件是否存在
    if not os.path.exists(query_img_path):
        print(f"Query图像不存在: {query_img_path}")
        return None
    
    if not os.path.exists(ref_img_path):
        print(f"Reference图像不存在: {ref_img_path}")
        return None
    
    # 加载图像
    try:
        query_img = Image.open(query_img_path).convert('RGB')
        ref_img = Image.open(ref_img_path).convert('RGB')
    except Exception as e:
        print(f"无法加载图像: {e}")
        return None
    
    # 获取原始bbox和多边形
    query_bbox = sample_info['query_bbox']
    ref_polygon = sample_info['ref_polygon']
    
    # 将多边形转换为bbox
    ref_bbox = convert_polygon_to_bbox(ref_polygon)
    
    # 在Query图像上绘制bbox（红色，不显示标签）
    query_img_with_bbox = draw_bbox_on_image(
        query_img.copy(), 
        query_bbox, 
        "",  # 不显示标签
        (255, 0, 0)
    )
    
    # 在Reference图像上绘制bbox（红色，不显示标签）
    ref_img_with_bbox = draw_bbox_on_image(
        ref_img.copy(), 
        ref_bbox, 
        "",  # 不显示标签
        (255, 0, 0)  # 改为红色
    )
    
    # 保存结果
    sample_id = sample_info['sample_id']
    
    # 保存Query图像（文件名包含类别和bbox信息）
    query_filename = f"sample_{sample_id}_query_{sample_info['class_name']}_bbox_{query_bbox[0]}_{query_bbox[1]}_{query_bbox[2]}_{query_bbox[3]}.png"
    query_save_path = os.path.join(save_dir, query_filename)
    query_img_with_bbox.save(query_save_path)
    
    # 保存Reference图像（文件名包含类别和bbox信息）
    ref_filename = f"sample_{sample_id}_reference_{sample_info['class_name']}_bbox_{ref_bbox[0]}_{ref_bbox[1]}_{ref_bbox[2]}_{ref_bbox[3]}.png"
    ref_save_path = os.path.join(save_dir, ref_filename)
    ref_img_with_bbox.save(ref_save_path)
    
    # 保存卫星图原图（无任何标注）
    ref_original_filename = f"sample_{sample_id}_satellite_original_{sample_info['class_name']}.png"
    ref_original_save_path = os.path.join(save_dir, ref_original_filename)
    ref_img.save(ref_original_save_path)
    
    print(f"样本 {sample_id}:")
    print(f"  Query图像: {query_save_path}")
    print(f"  Reference图像: {ref_save_path}")
    print(f"  卫星图原图: {ref_original_save_path}")
    print(f"  类别: {sample_info['class_name']}")
    print(f"  Query bbox: {query_bbox}")
    print(f"  Ref bbox: {ref_bbox}")
    print(f"  Query尺寸: {sample_info['query_size']}")
    print(f"  Ref尺寸: {sample_info['ref_size']}")
    print()
    
    return {
        'sample_id': sample_id,
        'query_bbox': query_bbox,
        'ref_bbox': ref_bbox,
        'query_size': sample_info['query_size'],
        'ref_size': sample_info['ref_size'],
        'class_name': sample_info['class_name']
    }

def main():
    """主函数"""
    print("=== Query-Reference图像对测试 ===")
    
    # 加载数据集
    data_root = 'data/CVOGL_DroneAerial'
    train_data, val_data = load_dataset_info(data_root)
    
    # 选择5个样本进行测试
    test_samples = []
    
    # 从训练集选择3个
    train_indices = random.sample(range(len(train_data)), 3)
    for idx in train_indices:
        sample_info = parse_data_item(train_data[idx])
        test_samples.append(sample_info)
    
    # 从验证集选择2个
    val_indices = random.sample(range(len(val_data)), 2)
    for idx in val_indices:
        sample_info = parse_data_item(val_data[idx])
        test_samples.append(sample_info)
    
    # 测试每个样本
    results = []
    for i, sample_info in enumerate(test_samples):
        print(f"--- 测试样本 {i+1}/5 ---")
        result = test_sample_pair(data_root, sample_info)
        if result:
            results.append(result)
    
    # 总结
    print(f"=== 测试完成 ===")
    print(f"成功测试了 {len(results)} 个样本")
    print(f"结果保存在 'pair_test_results' 目录")
    print("请检查生成的图像对，验证Query和Reference图像中的bbox是否对应！")

if __name__ == "__main__":
    main() 