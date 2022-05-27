import torch.utils.data as data
import torch
import torchvision
from PIL import Image
import os
import os.path
import numpy as np
import json
from torchvision import transforms
from torchvision.transforms import functional as F
import warnings
import random
import math
import copy
import numbers
from utils.augment import *
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_bbox_dict(root):
    print('loading from ground truth bbox')
    name_idx_dict = {}
    with open(os.path.join(root, 'images.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, name = fileline[0], fileline[1]
            name_idx_dict[name] = idx

    idx_bbox_dict = {}
    with open(os.path.join(root, 'bounding_boxes.txt')) as f:
        filelines = f.readlines()
        for fileline in filelines:
            fileline = fileline.strip('\n').split()
            idx, bbox = fileline[0], list(map(float, fileline[1:]))
            idx_bbox_dict[idx] = bbox

    name_bbox_dict = {}
    for name in name_idx_dict.keys():
        name_bbox_dict[name] = idx_bbox_dict[name_idx_dict[name]]

    return name_bbox_dict

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    return pil_loader(path)


def load_train_bbox(label_dict,bbox_dir):
    #bbox_dir = 'ImageNet/Projection/VGG16-448'
    final_dict = {}
    for i in range(200):
        now_name = label_dict[i]
        now_json_file = os.path.join(bbox_dir,now_name+"_bbox.json")
        with open(now_json_file, 'r') as fp:
            name_bbox_dict = json.load(fp)
        final_dict[i] = name_bbox_dict
    return final_dict

def get_bounding_boxes(gt_dir):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(gt_dir) as f:
        for line in f.readlines():
            #print(line)
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes

class CUBDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ddt_path, gt_path, input_size=256, crop_size=224,train=True, transform=None, target_transform=None, loader=default_loader, ext=8):
        from torchvision.datasets import ImageFolder
        self.train = train
        self.input_size = input_size
        self.crop_size = crop_size
        self.gt_path = gt_path
        self.ext = ext
        if self.train:
            self.ddt_path = ddt_path
            self.img_dataset = ImageFolder(os.path.join(root,'train'))
        else:
            self.ddt_path = None
            self.img_dataset = ImageFolder(os.path.join(root,'val'))
        if len(self.img_dataset) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.label_class_dict = {}
        self.train = train

        for k, v in self.img_dataset.class_to_idx.items():
            self.label_class_dict[v] = k
        if self.train:
            #load train bbox
            self.bbox_dict = load_train_bbox(self.label_class_dict, self.ddt_path)
        else:
            #load test bbox
            self.bbox_dict = get_bounding_boxes(self.gt_path)

        self.img_dataset = self.img_dataset.imgs

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.img_dataset[index]
        image_id = os.path.join(path.split('/')[-2],path.split('/')[-1])

        img = self.loader(path)
        if self.train:
            bbox = self.bbox_dict[target][path]
            # bbox = self.bbox_dict[image_id][0]
        else:
            # bbox = self.bbox_dict[path]
            # for multiple boxes, we pick the first box
            bbox = self.bbox_dict[image_id][0]

        w,h = img.size

        bbox = np.array(bbox, dtype='float32')

        if self.train:
            #convert from x, y, w, h to x1, y1, x2, y2, note that integer coordinates are used (e.g. [10, 43, 244, 369])
            bbox[0] = bbox[0]
            bbox[2] = bbox[0] + bbox[2]
            bbox[1] = bbox[1]
            bbox[3] = bbox[1] + bbox[3]
    
            img_i, bbox_i = RandomResizedBBoxCrop((self.crop_size))(img, bbox)
            img, bbox = RandomHorizontalFlipBBox()(img_i, bbox_i)

        else:
            img_i, bbox_i = ResizedBBoxCrop((self.input_size,self.input_size))(img, bbox)
            img, bbox = CenterBBoxCrop((self.crop_size))(img_i, bbox_i)

        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, bbox

    def __len__(self):
        return len(self.img_dataset)
