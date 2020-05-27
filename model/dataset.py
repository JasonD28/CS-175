import os
import numpy as np
import pandas as pd
import torch
import pydicom
import cv2

class PneumoniaDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "stage_2_train_images"))))
        self.masks = pd.read_csv(os.path.join(root, "stage_2_train_labels.csv"))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "stage_2_train_images", self.imgs[index])
        dicom = pydicom.read_file(img_path)
        image = dicom.pixel_array
        img_label = self.masks.loc[self.masks['patientId'] == self.imgs[index][:-3]]
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        if img_label['Target'] == 1:
            width = img_label['width']
            height = img_label['height']
            x = img_label['x']
            y = img_label['y']

            cv2.rectangle(mask, (x, y), (x + width, y + height), 1, -1)
            mask[mask > 0] = 1
            masks = masks = torch.tensor([mask], dtype=torch.uint8)
            
            boxes = [[x, y, x + width, y + height]]
            boxes = torch.tensor(boxes, dtype=torch.float32)

            labels = torch.ones((1,), dtype=torch.int64)
            target = {
                'boxes': boxes,
                'masks': masks,
                'labels': labels
            }
        else:
            target = {
                'boxes': None,
                'masks': None,
                'labels': None
            }

        return image, target




