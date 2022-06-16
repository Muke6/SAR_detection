import cv2
import json
import torch
import os.path
import argparse
import itertools
import numpy as np
from tqdm import tqdm


def save2json(output_path, results):
    if not os.path.exists(os.path.join(output_path, 'jsons')):
        os.mkdir(os.path.join(output_path, 'jsons'))
    json_results = []
    for img_name in results:
        for obj in results[img_name]['results']:
            json_results.append({'file_name': img_name, 'category': obj[0],
                                 'score': obj[1], 'bbox': obj[2], 'id': len(json_results)})
    with open(os.path.join(output_path, 'jsons', 'results.json'), 'w') as f:
        json.dump(json_results, f)
    return 0

    with tqdm(total=len(os.listdir(input_path))) as bar:
        for img_name in os.listdir(input_path):
            bar.update(1)
            img_path = os.path.join(input_path, img_name)
            im0s = cv2.imread(img_path)

            imgs = cv2.imread(img_path)
            crop_x = np.unique(np.clip(
                np.arange(0, im0s.shape[1], imgsz[1] - olsz), a_min=0, a_max=im0s.shape[1] - imgsz[1]))
            crop_y = np.unique(np.clip(
                np.arange(0, im0s.shape[0], imgsz[0] - olsz), a_min=0, a_max=im0s.shape[0] - imgsz[0]))
            id_crop = [bbox for bbox in itertools.product(crop_y, crop_x)]
            imgs = np.stack(
                [imgs[bbox[0]:bbox[0] + imgsz[0], bbox[1]:bbox[1] + imgsz[1]]
                 for bbox in itertools.product(crop_y, crop_x)], 0)
            imgs = np.stack(imgs, 0)
            imgs = np.stack([letterbox(img, imgsz, stride=stride, auto=True)[0] for img in imgs], 0)
            imgs = imgs.transpose((0, 3, 1, 2))  # [::-1]
            imgs = torch.from_numpy(np.ascontiguousarray(imgs)).float().to(device) / 255.0
            imgs = imgs[None] if len(imgs.shape) == 3 else imgs
            preds = []
            batch_num = imgs.shape[0] // batchsz if imgs.shape[0] // batchsz == imgs.shape[0] / batchsz else \
                imgs.shape[0] // batchsz + 1
            for batch in range(batch_num):
                pred = model(imgs[batch * batchsz:(batch + 1) * batchsz], augment=augment, visualize=False)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)
                for b in range(len(pred)):
                    pred[b][:, 0], pred[b][:, 1] = pred[b][:, 0] + id_crop[batch * batchsz + b][1], \
                                                   pred[b][:, 1] + id_crop[batch * batchsz + b][0]
                    pred[b][:, 2], pred[b][:, 3] = pred[b][:, 2] + id_crop[batch * batchsz + b][1], \
                                                   pred[b][:, 3] + id_crop[batch * batchsz + b][0]
                    preds.append(pred[b])
            preds = torch.cat(preds, 0).unsqueeze(0)
            preds = preds[:, preds[0, :, 4] >= 0.5]
            nms_id = torchvision.ops.batched_nms(preds[0, :, :4], preds[0, :, 4], preds[0, :, 5], 0.2)
            preds = preds[0, nms_id]
            result = {os.path.split(img_path)[-1]:
                          {'path': img_path, 'height': im0s.shape[0], 'width': im0s.shape[1],
                           'results': [[obj_list[int(obj[5])], obj[4].item(), obj[0:4].tolist()] for obj in
                                       preds]}}
            results.update(result)
        view_img(output_path, result, obj_list, if_save=True, if_show=False)
        if save_txt:
            save2txt(output_path, result)
        if save_xml:
            save2xml(output_path, result)
    if save_json:
        save2json(output_path, results)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/images/')
    parser.add_argument('--output_path', type=str, default='test/')
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp/82.pt', help='model path(s)')
    parser.add_argument('--imgsz', type=int, default=[1024], help='inference size h,w')
    parser.add_argument('--batchsz', type=int, default=3, help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--crop_det', action='store_true', default=True)
    parser.add_argument('--olsz', type=int, default=320)
    parser.add_argument('--save_txt', default=False, action='store_true')
    parser.add_argument('--save_xml', default=True, action='store_true')
    parser.add_argument('--save_json', default=False, action='store_true')
    parser.add_argument('--augment', action='store_true', default=True, help='augmented inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
