# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from types import new_class
from unittest import result
import numpy as np

from mmdet.apis import ( inference_detector,
                        init_detector, show_result_pyplot)
from glob import glob
import cv2
import itertools
import multiprocessing  


 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default='/media/ymhj/data/gxy/yolov5_work/coco/0528Predict/', help='Image file')
    parser.add_argument('--out',default='./result/', help='Image file')
    parser.add_argument('--config',default='work_dirs/work_cascade/best.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/work_cascade/epoch_100.pth' ,help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--iou', type=float, default=0.4, help='bbox score threshold')
    parser.add_argument(
        '--crop_size',
        default=2048,
        help='crop size.')
    parser.add_argument(
        '--overlap',
        default=200,
        help='overlop_size.')
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
def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img
def crop(img_path, crop_size, overlap_size):
    result1 = {}
    img = cv2.imread(img_path)
    # crop_json = {}
    if img.shape[0] == crop_size:
        return 
        # crop_json.update({save_name: gt_dict})
    elif img.shape[0] < crop_size:
        print(img, f'<{crop_size}!!')
    else:
        crop_x = np.arange(0, img.shape[1] - crop_size + 1, crop_size - overlap_size)
        crop_x = np.unique(np.array(crop_x.tolist() + [img.shape[1] - crop_size]))
        crop_y = np.arange(0, img.shape[0] - crop_size + 1, crop_size - overlap_size)
        crop_y = np.unique(np.array(crop_y.tolist() + [img.shape[0] - crop_size]))
        for crop_bbox in itertools.product(crop_y, crop_x):
            crop_img = img[crop_bbox[0]:crop_bbox[0] + crop_size, crop_bbox[1]:crop_bbox[1] + crop_size] 
            result1[(crop_bbox[0], crop_bbox[1])] = crop_img
            result_y0x0=list(result1.keys())
            result_img = list(result1.values())
    return   result_y0x0,result_img
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
        img_data = cv2.imread(img)
        outfile = args.out+img.split('/')[-1]
        result = inference_detector(model, img_data)
        # dele_re1 = pool.apply_async(caculate_class, (result,))
        # dele_res1 = dele_re1.get()
        # dele_ =[[],[],[]]
        # # print(dele_res1)
        # for jj in dele_res1:
        #     if(jj[1] <= result[jj[0]].shape[0]):
        #         dele_[jj[0]].append(jj[1])
        # for x in range(len(dele_)):
        #     result[x] = np.delete(result[x],dele_[x],0)
        # dele_re = pool.apply_async(caculate, (result,))
        # dele_res = dele_re.get()
        # dele_1 =[[],[],[]]
        # for jj in dele_res:
        #     if(jj[1] <= result[jj[0]].shape[0]):
        #         dele_1[jj[0]].append(jj[1])
        # # print(dele_1)
        # for x in range(len(dele_1)):
        #     result[x] = np.delete(result[x],dele_1[x],0)
        res_y0x0,res_img  = crop(img, args.crop_size, args.overlap)
        for i in range(len(res_img)):
            result_crop = inference_detector(model, res_img[i])
            dele_re10 = pool.apply_async(caculate_class, (result_crop,))
            dele_res10 = dele_re10.get()
            dele_ =[[],[],[]]
            # print(dele_res1)
            for jj in dele_res10:
                if(jj[1] <= result_crop[jj[0]].shape[0]):
                    dele_[jj[0]].append(jj[1])
            for x in range(len(dele_)):
                result_crop[x] = np.delete(result_crop[x],dele_[x],0)
            dele_re = pool.apply_async(caculate, (result_crop,))
            dele_res = dele_re.get()
            dele_1 =[[],[],[]]
            for jj in dele_res:
                if(jj[1] <= result[jj[0]].shape[0]):
                    dele_1[jj[0]].append(jj[1])
            for x in range(len(dele_1)):
                result_crop[x] = np.delete(result_crop[x],dele_1[x],0)
                for xx in  result_crop[x]:
                    if(xx[-1]>=args.score_thr):
                        xx[0]+=res_y0x0[i][1]
                        xx[1]+=res_y0x0[i][0]
                        xx[2]+=res_y0x0[i][1]
                        xx[3]+=res_y0x0[i][0]
            for x in range(len(result)):
                result[x]=np.vstack((result[x],result_crop[x])) 

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
        image = show_result_pyplot(model, img, result,score_thr=args.score_thr)
        cv2.imwrite("{}/{}".format(args.out, img.split('/')[-1]), image)
    pool.close()
    pool.join()




if __name__ == '__main__':
    args = parse_args()
    
    main(args)
