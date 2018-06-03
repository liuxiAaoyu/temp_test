import collections
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import dataset
import os
import numpy as np
import json
from glob import glob
import cv2
import matplotlib.pyplot as plt
import random

image_list = []


class image_list:
    def __init__(self, root, split_size):
        images = collections.defaultdict(dict)
        jsonfiles = glob(os.path.join(root, '*.json'))
        for jsonfile in jsonfiles:
            json_gt = [json.loads(line) for line in open(jsonfile)]
            for gt in json_gt:
                gt_lanes = gt['lanes']
                y_samples = gt['h_samples']
                name, ext = os.path.splitext(gt['raw_file'])
                images[name]['base'] = gt['raw_file']
                gt_lanes_vis = [[(x, y) for (x, y) in zip(
                    lane, y_samples) if x >= 0] for lane in gt_lanes]
                images[name]['gt'] = gt_lanes_vis
        image_list = list(images.values())
        random.shuffle(image_list)
        train_size = int(len(image_list)*split_size)
        self.train_list = image_list[:train_size]
        self.valid_list = image_list[train_size:]


class TuSimpleDataset(dataset.Dataset):
    """
    A dataset for loading images (with masks) stored as `xyz.jpg` and `xyz_mask.png`.

    Parameters
    ----------
    root : str
        Path to root directory.
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::
        transform = lambda data, label: (data.astype(np.float32)/255, label)
    """

    def __init__(self, root, image_list, transform=None):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self.__count = 0
        self._image_list = image_list

    def __getitem__(self, idx):
        assert 'base' in self._image_list[idx], "Couldn't find base image for:" + \
            _iamge_list[idx]['base']
        base_filepath = os.path.join(self._root, self._image_list[idx]['base'])
        base = mx.image.imread(base_filepath)
        assert 'gt' in self._image_list[idx], "Couldn't find ground truth for:" + \
            _iamge_list[idx]['base']
        mask = np.zeros(list(base.shape)[:2]+[1], np.uint8)
        width = np.linspace(2, 30, 511)
        for i, lane in enumerate(self._image_list[idx]['gt'], start=1):
            #cv2.polylines(mask, np.int32([lane]), isClosed=False, color=(255,255,255), thickness=15)
            if len(lane) == 0:
                continue
            # lane_l = [(x-width[index] if x-width[index] >= 0 else 0, y)
            #           for index, (x, y) in enumerate(lane)]
            lane_l = [(x-(width[y-200] if y >= 200 else 2), y)
                      for index, (x, y) in enumerate(lane)]
            lane_r = [(x+(width[y-200] if y >= 200 else 2), y)
                      for index, (x, y) in enumerate(lane)]
            lane_r.reverse()
            cv2.fillPoly(mask, np.int32([lane_l+lane_r]), (1))
        mask_nd = mx.nd.array(mask)
        if self._transform is not None:
            return self._transform(base, mask_nd)
        else:
            return base, mask_nd

    def __len__(self):
        return len(self._image_list)


def positional_augmentation(joint):
    # Random crop
    crop_height = 600
    crop_width = 800
    # Watch out: weight before height in size param!
    # aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
    # aug_joint = aug(joint)
    # # Deterministic resize
    # resize_size = 600
    # aug = mx.image.ResizeAug(resize_size)
    # aug_joint = aug(aug_joint)
    # Add more translation/scale/rotation augmentations here...
    aug = mx.image.RandomSizedCropAug(size=(crop_width, crop_height), area=(0.1,1.5), ratio=(0.2,1.8), interp=0)
    aug_joint = aug(joint)
    return aug_joint


def color_augmentation(base):
    # Only applied to the base image, and not the mask layers.
    aug = mx.image.ColorJitterAug(brightness=0.5, contrast=0.5, saturation=0.5)
    aug_base = aug(base)
    # Add more color augmentations here...

    return aug_base


def joint_transform(base, mask):
    ### Convert types
    base = mx.image.color_normalize(base.astype('float32')/255,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    #base = base.astype('float32')/255
    mask = mask.astype('float32')

    ### Join
    # Concatinate on channels dim, to obtain an 6 channel image
    # (3 channels for the base image, plus 3 channels for the mask)
    base_channels = base.shape[2]  # so we know where to split later on
    joint = mx.nd.concat(base, mask, dim=2)

    ### Augmentation Part 1: positional
    aug_joint = positional_augmentation(joint)
    ### Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]

    ### Augmentation Part 2: color
    aug_base = color_augmentation(aug_base)

    aug_base = mx.nd.transpose(aug_base, (2,0,1))
    aug_mask = mx.nd.transpose(aug_mask, (2,0,1))
    aug_mask = aug_mask.flatten()

    return aug_base, aug_mask

def joint_transform_valid(base, mask):
    ### Convert types
    base = mx.image.color_normalize(base.astype('float32')/255,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    #base = base.astype('float32')/255
    mask = mask.astype('float32')

    ### Join
    # Concatinate on channels dim, to obtain an 6 channel image
    # (3 channels for the base image, plus 3 channels for the mask)
    base_channels = base.shape[2]  # so we know where to split later on
    joint = mx.nd.concat(base, mask, dim=2)

    ### Augmentation Part 1: positional
    crop_height = 600
    crop_width = 800
    # Watch out: weight before height in size param!
    #aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
    #aug_joint = aug(joint)
    aug_joint = positional_augmentation(joint)
    ### Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]

    aug_base = mx.nd.transpose(aug_base, (2,0,1))
    aug_mask = mx.nd.transpose(aug_mask, (2,0,1))
    aug_mask = aug_mask.flatten()

    return aug_base, aug_mask


def plot_mx_arrays(arrays):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    plt.subplots(figsize=(12, 4))
    for idx, array in enumerate(arrays):
        #assert array.shape[2] == 3, "RGB Channel should be last"
        array = mx.nd.transpose(array, (1,2,0))
        if array.shape[2] == 1:
            array = mx.ndarray.concat(array, array, array, dim=2)
        plt.subplot(1, 2, idx+1)
        #print((array.clip(0, 255)/255))
        plt.imshow((array.clip(0, 255)/255).asnumpy())


if __name__ == '__main__':
    image_dir = '/media/xiaoyu/Document/data/TuSimple_Lane/train_set'
    image_list = image_list(image_dir, 0.9)
    dataset = TuSimpleDataset(image_dir, image_list.train_list, joint_transform)
    print(dataset.__len__())
    for i in range(5):
        index = random.randint(0, dataset.__len__())
        sample = dataset.__getitem__(index)
        sample_base = sample[0].astype('float32')
        sample_mask = sample[1].astype('float32')
        plot_mx_arrays([sample_base*255, sample_mask*255])
        plt.show()
