# SMGeo
This repository is the official implementation of ArXiv "SMGeo: Cross-View Object Geo-Localization with Grid-Level Mixture-of-Experts"
## 📌 Abstract

Cross-view object geo-localization aims to precisely pinpoint the same object across large-scale satellite imagery based on drone images. Due to significant differences in viewpoint and scale, coupled with complex background interference, traditional multi-stage *retrieval–matching* pipelines are prone to cumulative errors. To address these challenges, we present **SMGeo**, a promptable end-to-end transformer-based model for object geo-localization. The proposed model supports **click-based prompting** and can output object geo-localization results **in real time**, enabling interactive use. SMGeo adopts a fully transformer-based architecture, utilizing a **Swin Transformer** for joint feature encoding of both drone and satellite imagery, together with an **anchor-free transformer detection head** for direct coordinate regression. To better capture both **inter-view** and **intra-view** dependencies, we further introduce a **grid-level sparse Mixture-of-Experts (GMoE)** module into the cross-view encoder. This design allows the network to adaptively activate specialized experts according to the **content, scale, and source** of each spatial grid. In addition, the anchor-free detection head predicts object locations via **heatmap-based supervision** on reference images, avoiding the scale bias and matching complexity introduced by predefined anchor boxes.
---

## 🛠 Preparation

### 1. Dataset

We use the **CVOGL** dataset for cross-view object geo-localization experiments.

- **CVOGL Dataset**:  
  [Google Drive Download](https://drive.google.com/file/d/1WCwnK_rrU--ZOIQtmaKdR0TXcmtzU4cf/view)

Please download the dataset and organize it under the `data/` directory.

---

### 2. Backbone Pretrained Weights

SMGeo adopts **Swin Transformer** as the backbone network.

- **Official Swin Transformer Repository**:  
  [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

---

### 3. Provided Model Weights

We provide:
- A **Swin-Tiny pretrained model**
- A **SMGeo model trained for X epochs**

These files are shared via Baidu Netdisk:

- **SMGeo Model Weights**  
  Link: https://pan.baidu.com/s/1aq_VjzaGtX5-qIUmCi0zIg  
  Password: `vykd`

Please download and place the weights in the designated checkpoint directory before training or inference.

---

## 🚀 Training, Testing and Inference

### Training Script

The main training script is:

```bash
enhanced_training.py
