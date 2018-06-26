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
import math
from PIL import Image

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

def positional_augmentation(base, mask , crop_height, crop_width,area=(0.5,2.0),ratio=(0.6,1.4)):
    # Random crop
    size=(crop_width, crop_height)
    
    
    
    h, w, _ = base.shape
    src_area = h * w
    target_area = random.uniform(area[0], area[1]) * src_area
    new_ratio = random.uniform(*ratio)

    new_w = int(round(np.sqrt(target_area * new_ratio)))
    new_h = int(round(np.sqrt(target_area / new_ratio)))

    if random.random() < 0.5:
        new_h, new_w = new_w, new_h

    if new_w > w:
        new_w=w
    if new_h > h:
        new_h=h
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    base = mx.image.fixed_crop(base, x0, y0, new_w, new_h, size, interp=2)
    mask = mx.image.fixed_crop(mask, x0, y0, new_w, new_h, size, interp=0)
    return base, mask



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

    ### Augmentation Part 1: positional
    crop_height = 360
    crop_width = 680
    aug_base, aug_mask = positional_augmentation(base, mask, crop_height, crop_width)

    aug = gluon.data.vision.transforms.Resize((int(aug_base.shape[1]/2), int(aug_base.shape[0]/2)), interpolation=0)
    aug_mask = aug(aug_mask)

    stride = 8
    feat_height = int(math.ceil(float(crop_height) / stride))
    feat_width = int(math.ceil(float(crop_width) / stride))
    cell_width = 2
    aug_mask = aug_mask.reshape((feat_height, int(stride / cell_width), feat_width, int(stride / cell_width)))
    aug_mask = mx.nd.transpose(aug_mask, (1, 3, 0, 2))
    aug_mask = aug_mask.reshape((-1, feat_height, feat_width))
    aug_mask = aug_mask.reshape(-1)

    ### Augmentation Part 2: color
    aug_base = color_augmentation(aug_base)

    aug_base = mx.nd.transpose(aug_base, (2,0,1))
    # aug_mask = mx.nd.transpose(aug_mask, (2,0,1))
    # aug_mask = aug_mask.flatten()

    return aug_base, aug_mask

def joint_transform_valid(base, mask):
    ### Convert types
    base = mx.image.color_normalize(base.astype('float32')/255,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    # base = mx.image.color_normalize(base.astype('float32'),
    #                                 mean=mx.nd.array([122.675, 116.669, 104.008])
    #                                 )
                                    
    #base = base.astype('float32')/255
    mask = mask.astype('float32')

    ### Augmentation Part 1: positional
    crop_height = 360#base.shape[0]
    crop_width = 680#base.shape[1]
    # Watch out: weight before height in size param!
    aug_base, aug_mask = positional_augmentation(base, mask, crop_height, crop_width,area=(1.0,1.0),ratio=(1.0,1.0))

    
    aug = gluon.data.vision.transforms.Resize((int(crop_width/2), int(crop_height/2)), interpolation=0)
    aug_mask = aug(aug_mask)

    stride = 8
    feat_height = int(math.ceil(float(crop_height) / stride))
    feat_width = int(math.ceil(float(crop_width) / stride))
    cell_width = 2
    aug_mask = aug_mask.reshape((feat_height, int(stride / cell_width), feat_width, int(stride / cell_width)))
    aug_mask = mx.nd.transpose(aug_mask, (1, 3, 0, 2))
    aug_mask = aug_mask.reshape((-1, feat_height, feat_width))
    aug_mask = aug_mask.reshape(-1)

    aug_base = mx.nd.transpose(aug_base, (2,0,1))
    # aug_mask = mx.nd.transpose(aug_mask, (2,0,1))
    # aug_mask = aug_mask.flatten()

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


# def plot_mx_arrays_val(arrays):
#     """
#     Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
#     """
#     plt.subplots(figsize=(12, 4))
#     array = mx.nd.transpose(arrays[0], (1,2,0))
#     if array.shape[2] == 1:
#         array = mx.ndarray.concat(array, array, array, dim=2)
#     plt.subplot(1, 3, 1)
#     plt.imshow((array.clip(0, 255)/255).asnumpy())

#     array = mx.nd.squeeze(arrays[1])
#     #if array.shape[2] == 1:
#     #    array = mx.ndarray.concat(array, array, array, dim=2)
#     raw_labels = array.clip(0, 255).asnumpy()
#     result_img = Image.fromarray(colorize(raw_labels))#.resize([512,512])
#     plt.subplot(1, 3, 2)
#     plt.imshow(result_img)

#     array = mx.nd.squeeze(arrays[2])
#     #if array.shape[2] == 1:
#     #    array = mx.ndarray.concat(array, array, array, dim=2)
#     raw_labels = array.clip(0, 255).asnumpy()
#     result_img = Image.fromarray(colorize(raw_labels))#.resize([512,512])
#     plt.subplot(1, 3, 3)
#     plt.imshow(result_img)
#     plt.show()


if __name__ == '__main__':
    image_dir = '/media/ihorse/Data/tmp/tusimple/train_set'
    image_list = image_list(image_dir, 0.9)
    dataset = TuSimpleDataset(image_dir, image_list.train_list, joint_transform)
    print(dataset.__len__())
    for i in range(5):
        index = random.randint(0, dataset.__len__())
        sample = dataset.__getitem__(index)
        sample_base = sample[0].astype('float32')
        sample_mask = sample[1].astype('float32')
        
        crop_height, crop_width = sample_base.shape[1], sample_base.shape[2]
        stride = 8
        test_width = (int(crop_width) / stride) * stride
        test_height = (int(crop_height) / stride) * stride
        feat_width = int(test_width / stride)
        feat_height = int(test_height / stride)
        # re-arrange duc results
        labels = sample_mask.reshape((1, int(8/2), int(8/2),
                                    feat_height, feat_width))
        labels = mx.nd.transpose(labels, (0, 3, 1, 4, 2))
        labels = labels.reshape((1, int(test_height / 2), int(test_width / 2)))

        #labels = labels[:, :test_height, :test_width]
        labels = mx.nd.transpose(labels, [1, 2, 0])
        #labels = gluon.data.vision.transforms.Resize((test_width, test_height))(labels)
        sample_mask = mx.nd.transpose(labels, [2, 0, 1])

        #sample_mask = sample_mask.reshape(int(sample_base.shape[-2]/2),int(sample_base.shape[-1]/2))
        plot_mx_arrays([sample_base*255, sample_mask*255])
        plt.show()