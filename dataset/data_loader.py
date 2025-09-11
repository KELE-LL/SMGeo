# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from shapely.geometry import Polygon
from PIL import Image

cv2.setNumThreads(0)

class DatasetNotFoundError(Exception):
    pass

class MyAugment:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0)
        ])

    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                                cv2.LUT(sat, lut_sat),
                                cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh, imgw, _ = img.shape
        x, y, w, h = (bbox[0]+bbox[2])/2/imgw, (bbox[1]+bbox[3])/2/imgh, (bbox[2]-bbox[0])/imgw, (bbox[3]-bbox[1])/imgh
        img = self.transform(image=img)['image']

        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1 - y

        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1 - x

        # Restore to pixel coords
        x, y, w, h = x*imgw, y*imgh, w*imgw, h*imgh

        # Optional random crop (略，同原逻辑)
        # …（此处可保留原 MyAugment 中的 crop 逻辑，不影响 bbox 类型）…

        # 计算并返回新的 bbox
        new_bbox = [(x - w/2), (y - h/2), (x + w/2), (y + h/2)]
        return img, np.array(new_bbox, dtype=int)

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img,
                  (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min),
                  BOX_COLOR, -1)
    cv2.putText(img,
                text=class_name,
                org=(x_min, y_min - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35,
                color=TEXT_COLOR,
                lineType=cv2.LINE_AA)
    return img

class RSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL', split_name='train',
                 img_size=1024, transform=None, augment=False):
        self.data_root = data_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment = augment
        self.myaugment = MyAugment()

        # 加载数据列表
        if self.data_name == 'CVOGL_DroneAerial':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, f'{self.data_name}_{split_name}.pth')
            self.data_list = torch.load(data_path, weights_only=True)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 256)
        elif self.data_name == 'CVOGL_SVI':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, f'{self.data_name}_{split_name}.pth')
            self.data_list = torch.load(data_path, weights_only=True)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 512)
        else:
            raise DatasetNotFoundError(f'Unknown dataset {self.data_name}')

        # 动态构建类别到 ID 的映射
        all_classes = sorted({item[7] for item in self.data_list})
        self.cls_to_id = {name: idx for idx, name in enumerate(all_classes)}

        # 定义带 label_fields 的 Albumentations 变换
        self.rs_transform = A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.2),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.OneOf([A.Blur(p=0.4), A.MedianBlur(p=0.3)], p=0.5),
                A.OneOf([A.RandomBrightnessContrast(p=0.4), A.CLAHE(p=0.3)], p=0.5),
                A.ToGray(p=0.2),
                A.RandomGamma(p=0.3),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id'])
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = self.data_list[idx]

        # 只用第6个字段作为bbox，必须为[x1, y1, x2, y2]
        bbox = np.array(bbox, dtype=np.float32)
        if bbox.shape[0] != 4:
            raise ValueError(f"[卫星图像bbox错误] idx={idx}, bbox={bbox}，应为[x1, y1, x2, y2]")

        # 读取原始卫星图像尺寸
        ori_rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        ori_H, ori_W = ori_rsimg.shape[:2]

        # 读取并转为 RGB（numpy array）
        queryimg = cv2.cvtColor(
            cv2.imread(os.path.join(self.queryimg_dir, queryimg_name)), cv2.COLOR_BGR2RGB)
        rsimg = cv2.cvtColor(
            cv2.imread(os.path.join(self.rsimg_dir, rsimg_name)), cv2.COLOR_BGR2RGB)

        # 只对卫星图像做增强（保持numpy array）
        if self.augment:
            bboxes = [bbox.tolist()]
            labels = [self.cls_to_id[cls_name]]
            rs_transformed = self.rs_transform(
                image=rsimg,
                bboxes=bboxes,
                category_id=labels
            )
            rsimg = rs_transformed['image']
            bbox = np.array(rs_transformed['bboxes'][0], dtype=np.float32)
            if bbox.shape[0] != 4:
                raise ValueError(f"[增强后卫星图像bbox错误] idx={idx}, bbox={bbox}，应为[x1, y1, x2, y2]")

        # 归一化 & 转 tensor（此时转为PIL Image）
        if self.transform is not None:
            from PIL import Image
            queryimg = Image.fromarray(queryimg)
            rsimg = Image.fromarray(rsimg)
            queryimg = self.transform(queryimg)
            rsimg = self.transform(rsimg)

        # 兼容多点点击：如果click_xy为多点，取均值
        click_xy_arr = np.array(click_xy)
        if click_xy_arr.ndim == 2 and click_xy_arr.shape[0] > 1:
            click_xy_mean = click_xy_arr.mean(axis=0)
        else:
            click_xy_mean = click_xy_arr
        click_h, click_w = int(click_xy_mean[1]), int(click_xy_mean[0])
        qh, qw = self.query_featuremap_hw
        mat_clickhw = np.zeros((qh, qw), dtype=np.float32)
        click_h_dist = [(i - click_h)**2 for i in range(qh)]
        click_w_dist = [(j - click_w)**2 for j in range(qw)]
        norm = np.sqrt(qh*qh + qw*qw)
        for i in range(qh):
            for j in range(qw):
                val = 1 - (np.sqrt(click_h_dist[i] + click_w_dist[j]) / norm)
                mat_clickhw[i, j] = val * val
        # --- 新增：拼接点击点热力图为4通道 ---
        mat_clickhw_tensor = torch.from_numpy(mat_clickhw).unsqueeze(0)
        if queryimg.shape[1:] == mat_clickhw_tensor.shape[1:]:
            queryimg_4ch = torch.cat([queryimg, mat_clickhw_tensor], dim=0)
        else:
            mat_clickhw_tensor = torch.nn.functional.interpolate(mat_clickhw_tensor.unsqueeze(0), size=queryimg.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
            queryimg_4ch = torch.cat([queryimg, mat_clickhw_tensor], dim=0)
        # --- 返回4通道查询图像 ---
        return queryimg_4ch, rsimg, torch.tensor(bbox, dtype=torch.float32), idx, torch.tensor(click_xy_mean, dtype=torch.float32), torch.tensor([ori_H, ori_W], dtype=torch.float32)
