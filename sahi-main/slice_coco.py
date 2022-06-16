# from sahi.slicing import slice_coco
# import os
# os.getcwd()
from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# coco_dict = load_json("/media/ymhj/data/kxp/zkxt/mmdetection-master/sahi-main/data/ann/instances_val2017.json")

# f, axarr = plt.subplots(1, 1, figsize=(12, 12))
# read image
# img_ind = 0
# img = Image.open("demo_data/" + coco_dict["images"][img_ind]["file_name"]).convert('RGBA')
# # iterate over all annotations
# for ann_ind in range(len(coco_dict["annotations"])):
#     # convert coco bbox to pil bbox
#     xywh = coco_dict["annotations"][ann_ind]["bbox"]
#     xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
#     # visualize bbox over image
#     ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
# axarr.imshow(img)

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="/media/ymhj/data/kxp/zkxt/mmdetection-master/sahi-main/data/ann/instances_val2017.json",
    image_dir="/media/ymhj/data/kxp/zkxt/mmdetection-master/data/coco/train2017/",
    output_coco_annotation_file_name="/media/ymhj/data/kxp/zkxt/mmdetection-master/sahi-main/data/sliced_coco.json",
    ignore_negative_samples=False,
    output_dir="/media/ymhj/data/kxp/zkxt/mmdetection-master/sahi-main/data/imgs/",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True
)

# f, axarr = plt.subplots(4, 5, figsize=(13,13))
# img_ind = 0
# for ind1 in range(4):
#     for ind2 in range(5):
#         img = Image.open("demo_data/sliced/" + coco_dict["images"][img_ind]["file_name"])
#         axarr[ind1, ind2].imshow(img)
#         img_ind += 1
#
# f, axarr = plt.subplots(4, 5, figsize=(13,13))
# img_ind = 0
# for row_ind in range(4):
#     for column_ind in range(5):
#         # read image
#         img = Image.open("demo_data/sliced/" + coco_dict["images"][img_ind]["file_name"]).convert('RGBA')
#         # iterate over all annotations
#         for ann_ind in range(len(coco_dict["annotations"])):
#             # find annotations that belong the selected image
#             if coco_dict["annotations"][ann_ind]["image_id"] == coco_dict["images"][img_ind]["id"]:
#                 # convert coco bbox to pil bbox
#                 xywh = coco_dict["annotations"][ann_ind]["bbox"]
#                 xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
#                 # visualize bbox over image
#                 ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
#             axarr[row_ind, column_ind].imshow(img)
#         img_ind += 1