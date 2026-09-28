from torch.utils.data import Dataset
from src.datasets.Data_Instance import Data_Container

class Dataset_Base(Dataset):
    r"""Base class for all datasets."""
    def __init__(self, resize=True): 
        self.resize = resize
        self.count = 42
        self.data = Data_Container()

    def set_size(self, size):
        r"""Set the image size."""
        self.size = tuple(size)

    def set_experiment(self, experiment):
        r"""Attach the experiment."""
        self.exp = experiment

    def setPaths(self, images_path, images_list, labels_path, labels_list):
        r"""Set the image and label paths and file lists."""
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = len(self.images_list)

    def getImagePaths(self):
        r"""List of all images in the dataset."""
        return self.images_list

    def __len__(self):
        r"""Number of items in the dataset."""
        return self.length

    def getItemByName(self, name):
        r"""Get an item by its name."""
        idx = self.images_list.index(name)
        return self.__getitem__(idx)

    def getFilesInPath(self, path):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this!")

    def setState(self, state):
        self.state = state