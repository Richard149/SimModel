import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'] 

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]

###############################################################
################################################################
# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
## 手机拍摄的图片，需要旋转矫正
## 横着拍的照片，实际是竖着的 + exif信息
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

################################################################
#################LoadImg########按照长边等比例缩放#################
################最后的size max== 640， min < 640##################
################################################################
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]

    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    
    assert np.max(img.shape[:2]) == self.img_size, 'Image not resized to correct size, max shape: %g' % np.max(img.shape[:2])
    assert np.min(img.shape[:2]) <= self.img_size, 'Image not resized to correct size, min shape: %g' % np.min(img.shape[:2])
    
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

########################################################################
def letterboxTo640_640(img, new_shape=(640, 640), color=(114, 114, 114)):

    shape = img.shape[:2]  # current shape [height, width] one == 640

    dw, dh = new_shape[1] - shape[1], new_shape[0] - shape[0]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    assert img.shape[0] == img.shape[1] == 640, 'Image not resized to correct size, shape: %g' % img.shape[:2]
    
    return img, (1,1), (dh, dw)  # img, ratio, pad

###############################################################################
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


##################################
### 默认输入imgsize = 640 * 640####
##################################
class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16):
        
        self.img_size = img_size

        ### find all images file paths
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (path, e))
        

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels

        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert nf > 0, f'No labels found in {cache_path}. Can not train without labels.'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update

        self.label_files = img2label_paths(cache.keys())  # update


        n = len(self.img_files)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int32)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # [0,0,0,...,1,1,1,...,2,2,2,...]
        self.n = n
        self.indices = range(n) # [0,1,2,...,n-1]

        self.imgs = [None] * n

    ### 序列化 dict
    ### [hash]      ---> hash
    ### [img_path]  ---> [label, shape]
    ### [results]   ---> [nf, nm, ne, nc, n]  --- 没啥用，一个统计信息
    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = f"Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'WARNING: No labels found in {path}.')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logging.info(f"New cache created: {path}")
        return x

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, index):

        index = self.indices[index]

        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)


        # Letterbox,no scale, just pad to 640x640
        img, ratio, pad = letterboxTo640_640(img)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # originhw, scale_ratio-hw, padhw

        ratio = (h / h0, w / w0)


        # Load labels------>>>xywh in normalized format
        labels = []
        x = self.labels[index]
        if x.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = ratio[1] * w0 * (x[:, 1] - x[:, 3] / 2) + pad[1]  # pad width
            labels[:, 2] = ratio[0] * h0 * (x[:, 2] - x[:, 4] / 2) + pad[0]  # pad height
            labels[:, 3] = ratio[1] * w0 * (x[:, 1] + x[:, 3] / 2) + pad[1]
            labels[:, 4] = ratio[0] * h0 * (x[:, 2] + x[:, 4] / 2) + pad[0]

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1


        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
#### drawing bounding boxes
## img ---->> (3, 640, 640)
## labels ---->> (nL, 6)
def drawing_boundbox(img, label):
    # img: torch tensor (3, H, W) -> numpy (H, W, 3)

    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0)).copy()
        # 如果是 float 类型，转 uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img).astype(np.uint8)
    
    # 原有的画框代码...
    for obj in label:
        cls, cx, cy, w, h = int(obj[1]), obj[2], obj[3], obj[4], obj[5]
        # img_h, img_w = img.shape[:2]
        cx = cx * 640
        cy = cy * 640
        w = w * 640
        h = h * 640
        
        color = (0, 255, 0)
        cv2.rectangle(img, (int((cx - w / 2)), int((cy - h / 2))), (int((cx + w / 2)), int((cy + h / 2))), color, thickness=1)
    
    #RGB to BGR
    img = img[:, :, ::-1]
    return img

def drawing_img(img):
    # img: torch tensor (3, H, W) -> numpy (H, W, 3)

    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0)).copy()
        # 如果是 float 类型，转 uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img).astype(np.uint8)

    #RGB to BGR
    img = img[:, :, ::-1]
    return img

######################### dataloader #################################################################
class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
    

def create_dataloader(path, batch_size=16):

    dataset = LoadImagesAndLabels(path)

    batch_size = min(batch_size, len(dataset))

    # 不使用分布式，使用默认的 sampler
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = InfiniteDataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


if __name__ == "__main__":

    # dataset = LoadImagesAndLabels(path='/home/lijiaxin/coco128/images/train2017', img_size=640, batch_size=16)
    # for i in range(len(dataset)):

    #     img, label, path, shapes = dataset[i]
    #     #print(f"img shape: {img.shape}, label shape: {label.shape}, path: {path}, shapes: {shapes}")
    #     show_img = drawing_boundbox(img, label)
    #     cv2.imwrite(f"output/img_{i}.jpg", show_img)
    #     print(f"img_{i}.jpg saved.")
    dataloader, dataset  = create_dataloader(path='/home/lijiaxin/coco128/images/train2017')
    for i, (img, label, path, shapes) in enumerate(dataloader):

        print(f"img shape: {img.shape}, label shape: {label.shape}")
        for n in range(len(img)):
            draw_img = img[n, :, :, :].squeeze()
            #print(f"draw_img shape: {draw_img.shape}")
            # 如果 n 是单个值
            label_index = torch.tensor(n)

            # 筛选 class_id == n 的所有记录
            mask = label[:, 0] == label_index  # 布尔掩码
            #print(f"mask shape: {mask.shape}")
            filtered_labels = label[mask]      # shape: (M, 5)

            show_img = drawing_img(draw_img)
            #show_img = drawing_boundbox(draw_img, filtered_labels)
            cv2.imwrite(f"output/img_{i}_{n}.jpg", show_img)
            print(f"img_{i}_{n}.jpg saved.")
        break

