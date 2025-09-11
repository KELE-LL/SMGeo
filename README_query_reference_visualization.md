# Query-Reference图像对MoE专家激活特征可视化

## 功能说明

本脚本用于可视化Query-Reference图像对的MoE专家激活特征，支持同时处理两种类型的图像并输出对比结果。

## 主要特性

- **双图像处理**: 同时处理Query图像（4通道）和Reference图像（3通道）
- **逐步输出流程**: 按照用户要求的顺序逐步显示各阶段结果
- **10列布局输出**: 原图对比(2列) + Query(原图+3个stage) + Reference(原图+3个stage)
- **专家激活可视化**: 展示不同stage的专家激活情况
- **智能图像匹配**: 从数据集文件中获取正确的Query-Reference图像对
- **独立特征提取**: Query和Reference图像分别使用各自的专家激活信息

## 问题修复

### 之前的问题
在之前的版本中，Query和Reference图像的专家激活特征极其相似，这是因为：
1. **错误的键值选择**: 无论处理Query还是Reference图像，都使用了相同的`'sat'`门控信息
2. **特征提取不一致**: 两个分支没有真正独立处理，导致专家激活结果相同

### 修复方案
1. **正确选择门控信息**: 根据`image_type`参数选择对应的门控信息
   - Query图像: 使用`stage_gates.get('query', [])`
   - Reference图像: 使用`stage_gates.get('sat', [])`
2. **独立特征处理**: 确保每个分支使用自己的专家激活信息
3. **调试信息增强**: 添加详细的调试输出，便于验证修复效果

## 输出流程

脚本按照以下步骤逐步处理每个图像对：

### 步骤1: 显示Query原图
- 保存Query原始图像
- 文件名: `sample{X}_sampleID{Y}_query_{class_name}_original.png`

### 步骤2: 显示Query各stage专家激活
- 生成Query图像的Stage 1、2、3专家激活可视化
- 使用Query分支的专家门控信息
- 文件名: `sample{X}_sampleID{Y}_query_{class_name}_stage{Z}_experts.jpg`

### 步骤3: 显示Reference原图
- 保存Reference原始图像
- 文件名: `sample{X}_sampleID{Y}_reference_{class_name}_original.png`

### 步骤4: 显示Reference各stage专家激活
- 生成Reference图像的Stage 1、2、3专家激活可视化
- 使用Reference分支的专家门控信息
- 文件名: `sample{X}_sampleID{Y}_reference_{class_name}_stage{Z}_experts.jpg`

### 步骤5: 保存统一对比图
- 将所有结果整合到一张8列对比图中
- 文件名: `sample{X}_sampleID{Y}_{class_name}_query_reference_comparison.png`

## 输出格式

### 单个图像文件
- **Query原图**: `sample{X}_sampleID{Y}_query_{class_name}_original.png`
- **Reference原图**: `sample{X}_sampleID{Y}_reference_{class_name}_original.png`
- **Query专家激活**: `sample{X}_sampleID{Y}_query_{class_name}_stage{Z}_experts.jpg`
- **Reference专家激活**: `sample{X}_sampleID{Y}_reference_{class_name}_stage{Z}_experts.jpg`

### 统一对比图
- 文件名: `sample{X}_sampleID{Y}_{class_name}_query_reference_comparison.png`
- 布局: 8列
  - 列1-4: Query图像（原图 + Stage 1 + Stage 2 + Stage 3）
  - 列5-8: Reference图像（原图 + Stage 1 + Stage 2 + Stage 3）

### 汇总可视化
- 文件名: `summary_visualization.png`
- 布局: 3行10列（前3个样本的完整对比）
  - 列1-2: 原图对比（Query | Reference）
  - 列3-6: Query专家激活（原图 + Stage 1 + Stage 2 + Stage 3）
  - 列7-10: Reference专家激活（原图 + Stage 1 + Stage 2 + Stage 3）

## 使用方法

### 1. 配置路径
在`main()`函数中设置以下路径：

```python
weights_path = "path/to/your/best_weights.pth"
query_data_path = "path/to/query/images"      # Query图像目录
reference_data_path = "path/to/reference/images"  # Reference图像目录
output_path = "path/to/output"                # 输出目录
```

### 2. 运行脚本
```bash
python expert_activation_visualization.py
```

### 3. 查看结果
结果将按照以下顺序逐步生成：
1. Query原图
2. Query各stage专家激活（使用Query分支门控信息）
3. Reference原图
4. Reference各stage专家激活（使用Reference分支门控信息）
5. 统一对比图
6. 汇总可视化

## 图像对匹配策略

脚本从数据集文件（`CVOGL_DroneAerial_train.pth`和`CVOGL_DroneAerial_val.pth`）中获取正确的Query-Reference图像对：

1. 加载训练集和验证集数据
2. 解析每个样本的Query和Reference图像文件名
3. 确保图像对是真正对应的，而不是随机匹配
4. 包含完整的样本信息：样本ID、类别、bbox坐标等

## 专家颜色映射

- **专家1**: 黄色 [255, 255, 0]
- **专家2**: 绿色 [0, 255, 0]  
- **专家3**: 青色 [0, 255, 255]
- **专家4**: 蓝色 [0, 0, 255]
- **专家5**: 红色 [255, 0, 0]
- **专家6**: 洋红色 [255, 0, 255]

## 注意事项

1. 确保Query图像目录和Reference图像目录都存在
2. 确保数据集文件（.pth文件）存在且可访问
3. Query图像会自动转换为4通道（如果原为3通道）
4. Reference图像保持3通道
5. 输出图像尺寸为1024x1024
6. 支持JPG、JPEG、PNG、BMP格式
7. **重要**: 现在Query和Reference图像会使用各自的专家激活信息，结果应该明显不同

## 技术细节

- 使用Swin Transformer MoE模型
- 支持6个专家，top-2选择
- 处理Stage 1、2、3的专家激活（Stage 0无MoE）
- 网格粒度: Stage 1(128x128), Stage 2(64x64), Stage 3(32x32)
- 透明度: Stage 1(35%), Stage 2(35%), Stage 3(35%)
- **关键修复**: Query和Reference分支独立处理，使用各自的专家门控信息

## 输出文件命名规则

所有输出文件都包含以下信息：
- `sample{X}`: 样本索引
- `sampleID{Y}`: 样本ID（从数据集文件中获取）
- `{image_type}`: 图像类型（query或reference）
- `{class_name}`: 类别名称
- `stage{Z}`: 阶段编号（1、2、3）
- `experts`: 专家激活标识
- `original`: 原图标识

## 验证修复效果

运行修复后的脚本，你应该能看到：
1. Query和Reference图像的专家激活特征明显不同
2. 调试信息显示分别使用了Query和Reference的门控信息
3. 每个分支的专家激活统计信息独立显示

## 专家激活准确性验证

为了确保专家激活的准确性，脚本现在包含完整的验证功能：

### **验证指标**

#### **1. 结构相似性 (SSIM)**
- **SSIM > 0.95**: 警告 - Query和Reference可能过于相似
- **SSIM < 0.3**: 良好 - Query和Reference差异明显
- **0.3 ≤ SSIM ≤ 0.95**: 正常 - Query和Reference有一定差异

#### **2. 均方误差 (MSE)**
- 计算Query和Reference图像之间的像素级差异
- 数值越大表示差异越明显

#### **3. 专家颜色分布差异**
- 分析每个专家在Query和Reference中的激活比例
- 使用KL散度计算分布差异
- **差异 < 0.1**: 警告 - 可能存在异常
- **差异 > 0.8**: 良好 - 激活模式明显不同
- **0.1 ≤ 差异 ≤ 0.8**: 正常 - 有一定差异

#### **4. 激活模式分析**
- 统计不同激活模式的数量
- 检测是否存在过于单一的激活模式
- 分析专家选择的多样性

### **验证输出示例**

```
[VALIDATION] 样本 1 专家激活准确性验证
============================================================

--- Stage 1 验证 ---
  Stage 1 结构相似性 (SSIM): 0.2345
    [INFO] SSIM较低 (0.2345)，Query和Reference差异明显
  Stage 1 均方误差 (MSE): 156.78
  Stage 1 专家颜色分布:
    Query: {0: 0.25, 1: 0.30, 2: 0.15, 3: 0.20, 4: 0.05, 5: 0.05}
    Reference: {0: 0.15, 1: 0.20, 2: 0.30, 3: 0.25, 4: 0.05, 5: 0.05}
    Query-Reference颜色分布差异: 0.7234
    [INFO] 颜色分布差异较大 (0.7234)，激活模式明显不同
```

### **如何判断准确性**

#### **✅ 准确性高的表现**
1. **SSIM < 0.5**: Query和Reference结构差异明显
2. **MSE > 100**: 像素级差异足够大
3. **颜色分布差异 > 0.5**: 专家激活模式明显不同
4. **激活模式多样化**: 不同网格有不同的专家组合

#### **⚠️ 需要关注的异常**
1. **SSIM > 0.9**: 可能使用了相同的门控信息
2. **颜色分布差异 < 0.2**: 专家激活过于相似
3. **激活模式单一**: 所有网格都激活相同的专家
4. **专家分布极不均匀**: 某个专家占比过高

### **验证步骤**

脚本现在包含6个步骤：
1. **显示Query原图**
2. **显示Query各stage专家激活**
3. **显示Reference原图**
4. **显示Reference各stage专家激活**
5. **保存统一对比图**
6. **验证专家激活准确性** ← 新增

### **依赖要求**

验证功能需要安装`scikit-image`库：
```bash
pip install scikit-image
```

如果没有安装，验证功能会跳过SSIM计算，但其他指标仍然可用。
