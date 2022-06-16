import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm

def load_json_gt(json_path):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    category_dict = {cate['id']: cate['name'] for cate in json_dict['categories']}
    img_dict = {img['file_name']: {'width': img['width'],
                                   'height': img['height'],
                                   'gts': []
                                   } for img in json_dict['images']}
    img_id = {img['id']: img['file_name'] for img in json_dict['images']}
    for anno in json_dict['annotations']:
        bbox = [anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]]
        img_dict[img_id[anno['image_id']]]['gts'].append([category_dict[anno['category_id']], bbox])
    return img_dict, json_dict['categories']


def merge(img_path, small_size, big_size, gt_dict, category_dict, save_path):
    category_ids = {cate['name']: cate['id'] for cate in category_dict}
    images_dict = []
    annotations_dict = []

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num = 0
    while True:
        if num * small_size >= big_size:
            break
        else:
            num += 1
    img_list = os.listdir(img_path)
    img_type = '.' + img_list[0].split('.')[-1]
    groups = len(img_list) / num ** 2 if len(img_list) // num ** 2 == len(img_list) / num ** 2 else \
        len(img_list) // num ** 2 + 1
    img_list += random.sample(img_list, groups * num ** 2 - len(img_list))
    random.shuffle(img_list)
    img_list_groups = [img_list[int(i * len(img_list) / num ** 2):int((i + 1) * len(img_list) / num ** 2)] for i in
                       range(num ** 2)]
    with tqdm(total=groups) as bar:
        for index in range(groups):
            save_name = str(len(os.listdir(save_path))) + img_type
            imgs = [i[index] for i in img_list_groups]
            imgs_array = {i: cv2.imread(os.path.join(img_path, imgs[i])) for i in range(len(imgs))}
            imgs_json = {i: gt_dict[imgs[i]] for i in range(len(imgs))}
            merge_imgs = np.concatenate([np.concatenate([imgs_array[j] for j in range(int(i * num), ((i + 1) * num))], 1)
                                         for i in range(num)], 0)
            cv2.imwrite(os.path.join(save_path, save_name), merge_imgs)
            merge_json_dict = []
            for i in range(num):
                for j in range(int(i * num), ((i + 1) * num)):
                    for obj in imgs_json[j]['gts']:
                        merge_json_dict.append([obj[0], [obj[1][0] + (j - int(i * num)) * small_size,
                                                         obj[1][1] + i * small_size,
                                                         obj[1][2] + (j - int(i * num)) * small_size,
                                                         obj[1][3] + i * small_size]])
            images_dict.append({'file_name': save_name, 'width': merge_imgs.shape[1], 'height': merge_imgs.shape[0],
                                'id': len(images_dict)})
            for obj in merge_json_dict:
                annotations_dict.append({'image_id': len(images_dict) - 1, 'category_id': category_ids[obj[0]],
                                         'bbox': [obj[1][0], obj[1][1], obj[1][2] - obj[1][0], obj[1][3] - obj[1][1]],
                                         'area': (obj[1][2] - obj[1][0]) * (obj[1][3] - obj[1][1]), 'iscrowd': 0,
                                         'id': len(annotations_dict)})
            # res_imgs = merge_imgs.copy()
            # for obj in merge_json_dict:
            #     cv2.rectangle(res_imgs, (obj[1][0], obj[1][1]), (obj[1][2], obj[1][3]), (255, 0, 0), 2, 2)
            # cv2.imwrite('./results/' + save_name, res_imgs)
            bar.update(1)

    with open(f'{small_size}to{big_size}.json', 'w') as f:
        json.dump({'images': images_dict, 'annotations': annotations_dict, 'categories': category_dict}, f)


if __name__ == '__main__':
    gt_dict, category_dict = load_json_gt('./trainData/instances_train.json')
    merge(img_path='./trainData/1024/', small_size=1024, big_size=2048,
          gt_dict=gt_dict, category_dict=category_dict, save_path='./merge/')
