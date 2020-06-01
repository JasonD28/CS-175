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
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "stage_2_train_images"))))
        self.masks = pd.read_csv(os.path.join(root, "stage_2_train_labels.csv"))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "stage_2_train_images", self.imgs[index])
        dicom = pydicom.read_file(img_path)
        image = dicom.pixel_array
        image = image[..., np.newaxis]
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

        img_label = self.masks.loc[self.masks['patientId'] == self.imgs[index][:-4]].iloc[0]
        mask = np.zeros((1, 1024, 1024), dtype=np.uint8)

        target = {
            'image_id': torch.tensor([index]),
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }

        if int(img_label['Target']) == 1:
            width = int(img_label['width'])
            height = int(img_label['height'])
            x = int(img_label['x'])
            y = int(img_label['y'])

            cv2.rectangle(mask, (x, y), (x + width, y + height), 1, -1)
            mask[mask > 0] = 1
            masks = torch.tensor(mask, dtype=torch.uint8)

            boxes = [x, y, x + width, y + height]
            boxes = torch.tensor([boxes], dtype=torch.float32)

            labels = torch.ones((1,), dtype=torch.int64)

            target['boxes'] = boxes
            target['masks'] = masks
            target['labels'] = labels
            target['area'] = torch.tensor([height * width], dtype=torch.int64)
        else:
            target['boxes'] = torch.tensor([[0,1,2,3]], dtype=torch.float32)
            target['masks'] = torch.tensor(mask, dtype=torch.uint8)
            target['labels'] = torch.zeros((1,), dtype=torch.int64)
            target['area'] = torch.tensor([4], dtype=torch.int64)
        return image, target

    def __len__(self):
        return len(self.imgs)




