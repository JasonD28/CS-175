import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import cv2
import sys
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, root, transforms, mode='train'):
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.imgs = list(sorted(os.listdir(os.path.join(root, "stage_2_"+self.mode+"_images"))))
        self.labels = pd.read_csv(os.path.join(root, self.mode+".csv"))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "stage_2_"+self.mode+"_images", self.imgs[index])
        dicom = pydicom.read_file(img_path)
        image = dicom.pixel_array
        image = image[..., np.newaxis]
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

        img_label = self.labels.loc[self.labels['patientId'] == self.imgs[index][:-4]]

        target = {
            'image_id': torch.tensor([index]),
            'iscrowd': torch.zeros((img_label.shape[0],), dtype=torch.int64)
        }
        if self.mode == 'train':
            boxes = []
            labels = []
            area = []
            for _, row in img_label.iterrows():
                if int(row['Target']) == 1:
                    width = int(row['width'])
                    height = int(row['height'])
                    x = int(row['x'])
                    y = int(row['y'])

                    boxes.append([x, y, x + width, y + height])
                    labels.append(2)
                    area.append(height * width)
                else:
                    boxes.append([0,1,2,3])
                    area.append(4)
                    if row['class'] == 'Normal':
                        labels.append(0)
                    else:
                        labels.append(1)
                target['position'] = torch.tensor(0) if row['ViewPosition'] == 'PA' else torch.tensor(1)
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['area'] = torch.tensor(area, dtype=torch.int64)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)

        return image, target

    def __len__(self):
        return len(self.imgs)




