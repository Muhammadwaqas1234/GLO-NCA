from src.datasets.Dataset_Base import Dataset_Base
import cv2
import numpy as np

class Dataset_3D(Dataset_Base):
    r"""Base class for 3D datasets."""
    def __init__(self, slice=None, resize=True): 
        self.slice = slice
        self.count = 42
        super().__init__(resize)

    def getImagePaths(self):
        r"""List of all images in the dataset."""
        return self.images_list

    def getItemByName(self, name):
        r"""Get an item by its name."""
        idx = self.images_list.index(name)
        return self.__getitem__(idx)
    
    def resize_image(self, img, isLabel):
        r"""Get an item by index."""
        if not isLabel:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        return img
    
    def preprocessing(self, img, isLabel=False):
        r"""Preprocess an image slice."""
        if not isLabel:
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Keep a single phase.
        if len(img.shape) > 2:
            img = img[:, :, 0] 

        img = np.expand_dims(img, axis=-1)

        if isLabel:
            img[..., 0][img[...,0] != 0] = 1

        return img

    def __getitem__(self, idx):
        r"""Placeholder overridden by subclasses."""
        return None, None, None