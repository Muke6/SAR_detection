# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from types import new_class
from unittest import result
import numpy as np
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from glob import glob
import cv2

import multiprocessing  
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default='/media/ymhj/data/gxy/yolov5-6.0/coco/val2017/crop/', help='Image file')
    parser.add_argument('--config',default='/media/ymhj/data/kxp/work/mmdetection-master/work_dirs/crossdet_r101_fpn_1x_coco/crossdet_r101_fpn_1x_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='/media/ymhj/data/kxp/work/mmdetection-master/work_dirs/crossdet_r101_fpn_1x_coco/epoch_100.pth' ,help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--iou', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args
def iou_area(box1,box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  
    iou = a1 / min(s1,s2) #iou = a1/ (s1 + s2 - a1)
    return iou,s1,s2
def caculate(result):
    i = 0
    dele = {}
    for class_res in result:
        j = 0
        i += 1
        for box1 in class_res[:-1]:
            
            if(box1[-1]>=args.score_thr):
                c2 = 0
                for class2 in result[i:]:
                    n = 0
                    for box2 in class2:
                        if(box2[-1]>=args.score_thr):
                            iou,s1,s2 = iou_area(box1[:4],box2[:4])
                            if(iou>=args.iou):
                                if(abs(s1-s2)<=12105):
                                    if(box1[-1]>=box2[-1]):
                                        dele[(i+c2,n)]=1
                                    else:
                                        dele[(i-1,j)]=1
                                else:
                                    if(s1>=s2):
                                        dele[(i+c2,n)]=1
                                    else:
                                        dele[(i-1,j)]=1
                        n+=1
                    c2+=1
            j+=1
        
    dele_res = list(dele.keys())
    
    return dele_res
def caculate_class(result):
    
    i = 0
    dele = {}
    for class_res in result:
        j=0
        for box1 in class_res[:-1]:
            j+=1
            c = 0
            if(box1[-1]>=args.score_thr):
                for box2 in class_res[j:]:
                    if(box2[-1]>=args.score_thr):
                        iou,s1,s2 = iou_area(box1[:4],box2[:4])

                        if(iou>=args.iou):
                            if(abs(s1-s2)<=12105):
                                if(box1[-1]>=box2[-1]):
                                    dele[(i,j+c)]=1
                                else:
                                    dele[(i,j-1)]=1
                            else:
                                if(s1>=s2):
                                    dele[(i,j+c)]=1
                                else:
                                    dele[(i,j-1)]=1
                    c+=1
        i+=1
    
    dele_res = list(dele.keys()) 
                 
    return dele_res
def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    imgs = glob(args.img + '/*tif')
    # print(imgs)
    cpu_core_num = multiprocessing.cpu_count()
    processes=int(cpu_core_num / 2) + 3 
    pool = multiprocessing.Pool(processes)
    for img in imgs:
        result = inference_detector(model, img)
        dele_re1 = pool.apply_async(caculate_class, (result,))
        dele_res1 = dele_re1.get()
        dele_ =[[],[],[]]
        # print(dele_res1)
        for jj in dele_res1:
            if(jj[1] <= result[jj[0]].shape[0]):
                dele_[jj[0]].append(jj[1])
        for x in range(len(dele_)):
            result[x] = np.delete(result[x],dele_[x],0)
        dele_re = pool.apply_async(caculate, (result,))
        dele_res = dele_re.get()
        dele_1 =[[],[],[]]
        for jj in dele_res:
            if(jj[1] <= result[jj[0]].shape[0]):
                dele_1[jj[0]].append(jj[1])
        # print(dele_1)
        for x in range(len(dele_1)):
            result[x] = np.delete(result[x],dele_1[x],0)
        # dele_res = dele_res+dele_res1
        # print(dele_res)
        # for ii in dele_res:
        #     if(ii[1] <= result[ii[0]].shape[0]):
        #         result[ii[0]] = np.delete(result[ii[0]],ii[1],0)
            
        show_result_pyplot(model, img, result, score_thr=args.score_thr)
    pool.close()
    pool.join()

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    imgs = glob(args.img+'/*tif')
    #print(imgs)
    for img in imgs:
        tasks = asyncio.create_task(async_inference_detector(model, img))
        result = await asyncio.gather(tasks)
        # print(result)
        # exit()
    # show the results

        show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
