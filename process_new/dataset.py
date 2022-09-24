import os
import os.path
import numpy as np
import pickle
import torch
# # use below packages as you like
import torchvision.transforms as tfs
from torchvision.transforms import functional as F

# from . import functional as F

from PIL import Image


# import math
# import numbers
# import random
# import warnings
# from collections.abc import Sequence
# from typing import Tuple, List, Optional
#
# import torch
# from PIL import Image
# from torch import Tensor
#
# try:
#     import accimage
# except ImportError:
#     accimage = None
# from . import functional as F


# import cv2

class RandomR(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        super(RandomR, self).__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        random = torch.rand(1)
        if random < 0.25:
            return img
            # return torch.Tensor
        elif random < 0.5:
            return F.rotate(img, 90.0)
        elif random < 0.75:
            F.rotate(img, 180.0)
        return F.rotate(img, 270.0)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SET(torch.utils.data.Dataset):

    def __init__(self, train=True):
        super(SET, self).__init__()

        self.base_folder = './SET2'
        # self.train_data_list = ["data1-1", "data1-2", "data1-3", "data1-4", "data1-5", "data2-1", "data2-2", "data2-3",
        #                         "data2-4", "data2-5"]
        # # , "data3-1", "data3-2", "data3-3",
        # #                     "data3-4", "data3-5"]
        # self.train_label_list = ["label1-1", "label1-2", "label1-3", "label1-4", "label1-5", "label2-1", "label2-2",
        #                          "label2-3",
        #                          "label2-4", "label2-5"]
        # # ,"label3-1", "label3-2", "label3-3",
        # #                      "label3-4", "label3-5"]
        # self.test_data_list = ["test_data1"]
        # self.test_label_list = ["test_label1"]

        self.train_data_list = ["data1-1"]
        self.train_label_list = ["label1-1"]
        self.test_data_list = ["data1-2"]
        self.test_label_list = ["label1-2"]


        # self.meta = {
        #     'filename': 'batches.meta',
        #     'key': 'label_names'
        # }

        self.train = train  # training set or test set
        if self.train:
            data_list = self.train_data_list
            label_list = self.train_label_list
        else:
            data_list = self.test_data_list
            label_list = self.test_label_list

        self.data = np.empty((0, 5, 8, 8))
        self.targets = []

        # now load the picked numpy arrays
        for file_name in data_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as d:
                load_data = pickle.load(d)
                self.data = np.vstack((self.data, load_data))
        for file_name in label_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                load_label = pickle.load(f)
                self.targets.extend(load_label)

        self.data = self.data.reshape(-1, 5, 8, 8)
        # self.data = self.data.transpose((0, 2, 3, 1)).astype(np.float32)  # convert to 8*8*3

        print(len(self.data))

    #     self._load_meta()
    #
    # def _load_meta(self):
    #     path = os.path.join(self.base_folder, self.meta['filename'])
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        label = np.zeros((1, 8, 8))
        label[0, int(target / 8), target % 8] = 1.
        # image data augmentation
        # img = Image.fromarray((img*255).astype(np.uint8))
        comb = np.vstack((img, label))

        comb = torch.from_numpy(comb)
        comb = tfs.RandomHorizontalFlip(p=0.5)(comb)
        comb = tfs.RandomVerticalFlip(p=0.5)(comb)
        # comb = tfs.RandomRotation((-90, 0, 90, 180))(comb)
        comb = RandomR()(comb)
        comb = comb.detach().numpy()

        # print("comb:")
        # print(comb.shape)
        # print("\n")

        img = comb[:5]
        label = np.argmax(comb[5])
        # img = np.array(img)
        img = img.astype(np.float32)

        # img = img.transpose(2, 0, 1)

        return img, label

    def __len__(self):
        return len(self.data)
