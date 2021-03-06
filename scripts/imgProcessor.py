import os
import random
import numpy as np

import pydicom

from skimage.transform import resize


class imgProcessor:
    def __init__(self, folder, files, boxes=None, shuffle=False, aug = False, predict=False):
        self.folder = folder
        self.files = files
        self.boxes = boxes
        self.img_size = 512
        self.shuffle = shuffle
        self.aug = aug
        self.pred = predict

    def load_files(self, file):
        img = pydicom.dcmread(os.path.join(self.folder, file)).pixel_array
        msk = np.zeros(img.shape)

        pid = file.split(".")[0]
        if pid in self.boxes:
            for location in self.boxes(pid):
                x,y,w,h = location
                msk[y:y+h,x:x+w] = 1
        if self.aug and random.randint(1,10) > 5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
                              
        img = resize(img, (self.img_size, self.img_size), mode='reflect')
        msk = resize(msk, (self.img_size, self.img_size), mode='reflect') > 0.5
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        
        return img, msk

    def load_pred(self, file):
        img = pydicom.dcmread(os.path.join(self.folder, file)).pixel_array
        img = resize(img, (self.img_size, self.img_size), mode='reflect')
        img = np.expand_dims(img, -1)
        return img

    def get_imgs(self, ind):
        if self.pred:
            img = [self.load_pred(file) for file in self.files]
            imgs = np.array(img)
            return  imgs, self.files
        else:
            img = [self.load_files(file) for file in self.files]
            imgs, msks = zip(*img)
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_shuffle(self):
        if self.shuffle:
            random.shuffle(self.files)

    def __len__(self, batch_size):
        if self.pred:
            return int(np.ceil(len(self.files) / batch_size))
        else:
            return int(len(self.files) / batch_size)
