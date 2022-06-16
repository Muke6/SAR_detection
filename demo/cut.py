from glob import glob
import cv2
from tqdm import tqdm
import numpy as np
import shapely
import os.path as osp
import random
import time
import os

img_path = '/media/ymhj/data/gxy/yolov5-6.0/coco/val2017/0528Predict/'
save_path = '/media/ymhj/data/gxy/yolov5-6.0/coco/val2017/'
imgs = glob(img_path+'*.tif')
print(imgs)
for img in imgs:
    image = cv2.imread(img)
    w = image.shape[1]
    h = image.shape[0]
    for i in range(0,w,2048):
        print(i)