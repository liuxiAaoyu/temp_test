import collections
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from collections import namedtuple
from PIL import Image

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',
    # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])


# Our extended list of label types. Our train id is compatible with Cityscapes
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , (180,100,180) ),
    Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , (250,170,100) ),
    Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , (220,220,250) ),
    Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
    Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , (220,220,100) ),
    Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
    Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , (220,220,220) ),
    Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , (250,170,250) ),
    Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
]



class Dataset(dataset.Dataset):
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

    def __init__(self, root, TrainOrValid, transform=None):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self.__count = 0
        self._TrainOrValid = TrainOrValid
        self._list_images(self._root)

    def _list_images(self, root):
        self.items = []
        images = os.path.join(root, 'images')
        labels = os.path.join(root, 'labels')
        iamge_path = os.path.join(images,self._TrainOrValid)
        label_path = os.path.join(labels,self._TrainOrValid)
        for filename in sorted(os.listdir(iamge_path)):
            name_id = os.path.splitext(filename)[0]
            filename = os.path.join(iamge_path,filename)
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in self._exts:
                warnings.warn('Ignoring %s of type %s. Only support %s'%(
                    filename, ext, ', '.join(self._exts)))
                continue
            labelname = os.path.join(label_path,name_id+'_train_id.png')
            if os.path.exists(labelname):
                self.items.append((filename, labelname))
            else:
                warnings.warn('Cannot find label file %s'%(
                    labelname))

    def __getitem__(self, idx):
        base = mx.image.imread(self.items[idx][0])
        mask = mx.image.imread(self.items[idx][1],flag=0)
        #mask = mask.reshape([mask.shape[0],mask.shape[1],-1])
        if self._transform is not None:
            return self._transform(base, mask)
        else:
            return base, mask

    def __len__(self):
        return len(self.items)


def positional_augmentation(joint):
    # Random crop
    crop_height = 512
    crop_width = 512
    # Watch out: weight before height in size param!
    # aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
    # aug_joint = aug(joint)
    # # Deterministic resize
    # resize_size = 600
    # aug = mx.image.ResizeAug(resize_size)
    # aug_joint = aug(aug_joint)
    # Add more translation/scale/rotation augmentations here...
    aug = mx.image.RandomSizedCropAug(size=(crop_width, crop_height), area=(0.5,1.5), ratio=(0.6,1.4), interp=0)
    aug_joint = aug(joint)
    return aug_joint


def color_augmentation(base):
    # Only applied to the base image, and not the mask layers.
    aug = mx.image.ColorJitterAug(brightness=0.2, contrast=0.2, saturation=0.2)
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
    #aug = mx.image.RandomCropAug(size=(512, 512))
    #aug_joint = aug(joint)
    ### Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]

    aug = gluon.data.vision.transforms.Resize((256,256), interpolation=0)
    aug_mask = aug(aug_mask)

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
    crop_height = 512
    crop_width = 512
    # Watch out: weight before height in size param!
    aug = mx.image.RandomCropAug(size=(crop_width, crop_height))
    aug_joint = aug(joint)
    #aug_joint = positional_augmentation(joint)
    ### Split
    aug_base = aug_joint[:, :, :base_channels]
    aug_mask = aug_joint[:, :, base_channels:]
    
    aug = gluon.data.vision.transforms.Resize((256,256), interpolation=0)
    aug_mask = aug(aug_mask)

    aug_base = mx.nd.transpose(aug_base, (2,0,1))
    aug_mask = mx.nd.transpose(aug_mask, (2,0,1))
    aug_mask = aug_mask.flatten()

    return aug_base, aug_mask


def get_palette():
    # get palette
    trainId2colors = {label.trainId: label.color for label in labels}
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def colorize(labels):
    """
    colorize the labels with predefined palette
    :param labels: labels organized in their train ids
    :return: a segmented result of colorful image as numpy array in RGB order
    """
    # label
    result_img = Image.fromarray(labels.astype('uint8')).convert('P')
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))


def plot_mx_arrays(arrays):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    plt.subplots(figsize=(12, 4))
    array = mx.nd.transpose(arrays[0], (1,2,0))
    if array.shape[2] == 1:
        array = mx.ndarray.concat(array, array, array, dim=2)
    plt.subplot(1, 2, 1)
    plt.imshow((array.clip(0, 255)/255).asnumpy())

    array = mx.nd.squeeze(arrays[1])
    #if array.shape[2] == 1:
    #    array = mx.ndarray.concat(array, array, array, dim=2)
    raw_labels = array.clip(0, 255).asnumpy()
    result_img = Image.fromarray(colorize(raw_labels))#.resize([512,512])
    plt.subplot(1, 2, 2)
    plt.imshow(result_img)


def plot_mx_arrays_val(arrays):
    """
    Array expected to be height x width x 3 (channels), and values are floats between 0 and 255.
    """
    plt.subplots(figsize=(12, 4))
    array = mx.nd.transpose(arrays[0], (1,2,0))
    if array.shape[2] == 1:
        array = mx.ndarray.concat(array, array, array, dim=2)
    plt.subplot(1, 3, 1)
    plt.imshow((array.clip(0, 255)/255).asnumpy())

    array = mx.nd.squeeze(arrays[1])
    #if array.shape[2] == 1:
    #    array = mx.ndarray.concat(array, array, array, dim=2)
    raw_labels = array.clip(0, 255).asnumpy()
    result_img = Image.fromarray(colorize(raw_labels))#.resize([512,512])
    plt.subplot(1, 3, 2)
    plt.imshow(result_img)

    array = mx.nd.squeeze(arrays[2])
    #if array.shape[2] == 1:
    #    array = mx.ndarray.concat(array, array, array, dim=2)
    raw_labels = array.clip(0, 255).asnumpy()
    result_img = Image.fromarray(colorize(raw_labels))#.resize([512,512])
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.show()


if __name__ == '__main__':
    image_dir = '/media/ihorse/Data/tmp/bdd100k/bdd100k/seg'
    dataset = Dataset(image_dir, 'train', joint_transform)
    print(dataset.__len__())
    for i in range(5):
        index = random.randint(0, dataset.__len__())
        sample = dataset.__getitem__(index)
        sample_base = sample[0].astype('float32')
        sample_mask = sample[1].astype('float32')
        sample_mask = sample_mask.reshape(256,256)
        plot_mx_arrays([sample_base*255, sample_mask])
        plt.show()

