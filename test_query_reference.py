#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file test_query_reference.py
@desc 测试Query-Reference图像对专家激活可视化脚本
"""

import os
import sys

def test_import():
    """测试导入"""
    try:
        import expert_activation_visualization
        print("✓ 成功导入 expert_activation_visualization 模块")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_dataset_files():
    """测试数据集文件是否存在"""
    data_root = 'data/CVOGL_DroneAerial'
    train_path = os.path.join(data_root, 'CVOGL_DroneAerial_train.pth')
    val_path = os.path.join(data_root, 'CVOGL_DroneAerial_val.pth')
    
    if os.path.exists(train_path):
        print(f"✓ 训练集文件存在: {train_path}")
    else:
        print(f"✗ 训练集文件不存在: {train_path}")
        return False
    
    if os.path.exists(val_path):
        print(f"✓ 验证集文件存在: {val_path}")
    else:
        print(f"✗ 验证集文件不存在: {val_path}")
        return False
    
    return True

def test_image_directories():
    """测试图像目录是否存在"""
    query_dir = 'data/CVOGL_DroneAerial/query'
    reference_dir = 'data/CVOGL_DroneAerial/satellite'
    
    if os.path.exists(query_dir):
        print(f"✓ Query图像目录存在: {query_dir}")
        query_files = [f for f in os.listdir(query_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  Query图像数量: {len(query_files)}")
    else:
        print(f"✗ Query图像目录不存在: {query_dir}")
        return False
    
    if os.path.exists(reference_dir):
        print(f"✓ Reference图像目录存在: {reference_dir}")
        reference_files = [f for f in os.listdir(reference_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  Reference图像数量: {len(reference_files)}")
    else:
        print(f"✗ Reference图像目录不存在: {reference_dir}")
        return False
    
    return True

def test_model_weights():
    """测试模型权重文件是否存在"""
    weights_path = "D:/PythonProject/Cross_View/saved_weights/debug_experiment/best_weights.pth"
    
    if os.path.exists(weights_path):
        print(f"✓ 模型权重文件存在: {weights_path}")
        return True
    else:
        print(f"✗ 模型权重文件不存在: {weights_path}")
        return False

def main():
    """主函数"""
    print("=== Query-Reference图像对专家激活可视化测试 ===")
    print()
    
    # 测试导入
    if not test_import():
        print("导入测试失败，请检查代码")
        return
    
    print()
    
    # 测试数据集文件
    if not test_dataset_files():
        print("数据集文件测试失败")
        return
    
    print()
    
    # 测试图像目录
    if not test_image_directories():
        print("图像目录测试失败")
        return
    
    print()
    
    # 测试模型权重
    if not test_model_weights():
        print("模型权重测试失败")
        return
    
    print()
    print("=== 所有测试通过！ ===")
    print("现在可以运行完整的专家激活可视化脚本了")
    print("运行命令: python expert_activation_visualization.py")

if __name__ == '__main__':
    main()


