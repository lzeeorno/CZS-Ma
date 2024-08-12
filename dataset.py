import numpy as np
import cv2  #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import argparse
from scipy.ndimage import zoom
from PIL import Image
from utilities.utils import mask_to_onehot
from skimage.transform import resize
'''
150 = organ
255 = tumor
0   = background 
'''
palette = [[0], [150], [255]]  # one-hot的颜色表
num_classes = len(palette)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper', default=200)
    parser.add_argument('--lower', default=-200)
    parser.add_argument('--img_size', default=512)
    parser.add_argument('--num_class', default=3)
    args = parser.parse_args()

    return args


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #load npy file
        npimage = np.load(img_path, allow_pickle=True)
        npmask = np.load(mask_path, allow_pickle=True)

        #(3,384,384) -> (384,384,3)
        npimage = npimage.transpose((2, 0, 1))
        #turn tumor as 1 like background when do liver seg
        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1
        # build a array (448, 448, 2), 1 layer save liver labels, 1 layer save tumor label
        nplabel = np.empty((448, 448, 2))
        nplabel[:, :, 0] = liver_label
        nplabel[:, :, 1] = tumor_label
        nplabel = nplabel.transpose((2, 0, 1))

        # npimage = np.abs(npimage.astype("float32"))
        # nplabel = np.abs(nplabel.astype("float32"))
        npimage = np.abs(npimage.astype("complex64"))
        nplabel = np.abs(nplabel.astype("complex64"))
        return npimage, nplabel

'''
npy Dataset
'''
class Dataset_ssl_lits2017(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #load npy file
        npimage = np.load(img_path, allow_pickle=True)
        npmask = np.load(mask_path, allow_pickle=True)

        #(3,512,512) -> (512,512,3)
        npimage = npimage.transpose((2, 0, 1))
        #encoding mask
        bg_label = npmask.copy()
        # liver_label[npmask == 0] = 0
        bg_label[npmask != 0] = 0

        liver_label = npmask.copy()
        # liver_label[npmask == 0] = 0
        liver_label[npmask != 1] = 0
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        # tumor_label[npmask == 1] = 0
        tumor_label[npmask != 2] = 0
        tumor_label[npmask == 2] = 1

        # bg_label_counts = np.bincount(bg_label.flatten())
        # for pixel_value, count in enumerate(bg_label_counts):
        #     print(f"bg_label像素值 {pixel_value} 出现的次数：{count}")
        #
        # liver_label_counts = np.bincount(liver_label.flatten())
        # for pixel_value, count in enumerate(liver_label_counts):
        #     print(f"liver_label像素值 {pixel_value} 出现的次数：{count}")
        #
        # tumor_label_counts = np.bincount(tumor_label.flatten())
        # for pixel_value, count in enumerate(tumor_label_counts):
        #     print(f"tumor_label像素值 {pixel_value} 出现的次数：{count}")

        # build a array, 1 layer save liver labels, 1 layer save tumor label
        nplabel = np.empty((512, 512, 3))
        nplabel[:, :, 0] = bg_label
        nplabel[:, :, 1] = liver_label
        nplabel[:, :, 2] = tumor_label


        # nplabel[:, :, 0] = bg_label
        # nplabel[:, :, 1] = liver_label
        # nplabel[:, :, 2] = tumor_label
        nplabel = nplabel.transpose((2, 0, 1))

        npimage = np.abs(npimage.astype("complex64"))
        # nplabel = np.abs(nplabel.astype("complex64"))
        nplabel = nplabel.astype("float64")
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(nplabel.shape))
        # exit()

        return npimage, nplabel

class Dataset_ssl_lits2017_unlabeled(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        #load npy file
        npimage = np.load(img_path, allow_pickle=True)
        #(3,512,512) -> (512,512,3)
        npimage = npimage.transpose((2, 0, 1))
        npimage = np.abs(npimage.astype("complex64"))
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(nplabel.shape))
        # exit()
        return npimage

class Dataset_ssl_lits2017_png(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.palette = palette
        self.num_classes = len(palette)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        ct = Image.open(img_path)
        seg = Image.open(mask_path)
        npimage = np.array(ct)
        npmask = np.array(seg)

        # 将灰度图像的形状从 (512, 512) 增纬为 (512, 512, 1)
        # npimage = npimage[:, :, np.newaxis]
        npimage = np.expand_dims(npimage, axis=2)   #[512,512,1]
        # npimage = np.concatenate([npimage, npimage], axis=2) #[512,512,2]
        npmask = np.expand_dims(npmask, axis=2)     #[512,512,1]
        npmask = mask_to_onehot(npmask, self.palette)  #[512, 512, 3]

        npmask = npmask.transpose([2, 0, 1])
        npimage = npimage.transpose([2, 0, 1])

        npimage = npimage.astype("float32")
        npmask = npmask.astype("float32")
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(npmask.shape))
        # exit()

        return npimage, npmask

class Dataset_synapse_png(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        '''
        [150,spleen脾][255,right kidney右肾][100,left kidney左肾][200,Gallbladder胆囊]
        [180,liver肝脏][220,stomach胃][130,aorta主动脉][160,pancreas胰脏]
        '''
        self.palette = [[0], [130], [200], [100], [255], [180], [160], [150], [220]]
        self.num_classes = len(self.palette)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        ct = Image.open(img_path)
        seg = Image.open(mask_path)
        npimage = np.array(ct)
        npmask = np.array(seg)

        # 将灰度图像的形状从 (512, 512) 增纬为 (512, 512, 1)
        # npimage = npimage[:, :, np.newaxis]
        npimage = np.expand_dims(npimage, axis=2)   #[512,512,1]
        # npimage = np.concatenate([npimage, npimage], axis=2) #[512,512,2]
        npmask = np.expand_dims(npmask, axis=2)     #[512,512,1]
        npmask = mask_to_onehot(npmask, self.palette)  #[512, 512, 9]

        npmask = npmask.transpose([2, 0, 1])
        npimage = npimage.transpose([2, 0, 1])

        npimage = npimage.astype("float32")
        npmask = npmask.astype("float32")
        # print('dataset check shape')
        # print("ct.size:{}".format(npimage.shape))
        # print("seg.size:{}".format(npmask.shape))
        # exit()
        #
        return npimage, npmask
