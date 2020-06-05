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

        img_label = self.labels.loc[self.labels['patientId'] == self.imgs[index][:-4]].iloc[0]

        target = {
            'image_id': torch.tensor([index]),
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }
        if self.mode == 'train':
            if int(img_label['Target']) == 1:
                width = int(img_label['width'])
                height = int(img_label['height'])
                x = int(img_label['x'])
                y = int(img_label['y'])

                boxes = [x, y, x + width, y + height]
                boxes = torch.tensor([boxes], dtype=torch.float32)

                labels = torch.tensor([2], dtype=torch.int64)

                target['boxes'] = boxes
                target['area'] = torch.tensor([height * width], dtype=torch.int64)
                target['labels'] = labels
            else:
                target['boxes'] = torch.tensor([[0,1,2,3]], dtype=torch.float32)
                target['area'] = torch.tensor([4], dtype=torch.int64)
                if img_label['class'] == 'Normal':
                    target['labels'] = torch.tensor([0], dtype=torch.int64)
                else:
                    target['labels'] = torch.tensor([1], dtype=torch.int64)
        return image, target

    def __len__(self):
        return len(self.imgs)




