import collections
import mxnet as mx
from mxnet.gluon.data import dataset
import os
import numpy as np
import json
from glob import glob
import cv2
import matplotlib.pyplot as plt
import random


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

    def __init__(self, root, transform=None):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.jph', '.jpeg', '.png']
        self.__count = 0
        self._list_images(self._root)

    def _list_images(self, root):
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
        self._image_list = list(images.values())

    def __getitem__(self, idx):
        assert 'base' in self._image_list[idx], "Couldn't find base image for:" + \
            iamge_list[idx]['base']
        base_filepath = os.path.join(self._root, self._image_list[idx]['base'])
        base = mx.image.imread(base_filepath)
        assert 'gt' in self._image_list[idx], "Couldn't find ground truth for:" + \
            iamge_list[idx]['base']
        mask = np.zeros(list(base.shape)[:3], np.uint8)
        width = np.linspace(2, 30, 511)
        for i, lane in enumerate(self._image_list[idx]['gt'], start=1):
            #cv2.polylines(mask, np.int32([lane]), isClosed=False, color=(255,255,255), thickness=15)
            if len(lane) == 0:
                continue
            # lane_l = [(x-width[index] if x-width[index] >= 0 else 0, y)
            #           for index, (x, y) in enumerate(lane)]
            lane_l = [(x-(width[y-200] if y>=200 else 2), y) for index, (x, y) in enumerate(lane)]
            lane_r = [(x+(width[y-200] if y>=200 else 2), y) for index, (x, y) in enumerate(lane)]
            lane_r.reverse()
            cv2.fillPoly(mask, np.int32([lane_l+lane_r]), (255, 255, 255))
        mask_nd = mx.nd.array(mask)
        if self._transform is not None:
            return self._transform(base, mask_nd)
        else:
            return base, mask_nd

    def __len__(self):
        return len(self._image_list)


def plot_mx_arrays(arrays):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    plt.subplots(figsize=(12, 4))
    for idx, array in enumerate(arrays):
        assert array.shape[2] == 3, "RGB Channel should be last"
        plt.subplot(1, 2, idx+1)
        print((array.clip(0, 255)/255))
        plt.imshow((array.clip(0, 255)/255).asnumpy())


if __name__ == '__main__':
    image_dir = '/media/xiaoyu/Document/data/TuSimple_Lane/train_set'
    dataset = TuSimpleDataset(image_dir)

    for i in range(20):
        index = random.randint(0, dataset.__len__())
        sample = dataset.__getitem__(index)
        sample_base = sample[0].astype('float32')
        sample_mask = sample[1].astype('float32')
        plot_mx_arrays([sample_base, sample_mask])
        plt.show()
