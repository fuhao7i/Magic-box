import os
import json
import glob
import shutil
import tqdm
import cv2
import numpy as np
from PIL import Image


def img2seg(json_path, image_path, save_path):

    with open(json_path,'r') as f:
        load_dict = json.load(f)

    images = load_dict['images']
    # print(images)
    for index in tqdm.tqdm(images, desc='transforming: '):
        image_id = index['id']
        file_name = index['file_name']

        bboxes = [i['bbox'] for i in load_dict['annotations'] if i['image_id'] == image_id]

        img = cv2.imread(image_path + file_name)
        mask = np.ones([img.shape[0], img.shape[1]], dtype=np.uint8) * 1

        for bbox in bboxes:

            x0 = bbox[0]
            y0 = bbox[1]

            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            mask[y0:y1, x0:x1] = 0

        gt = mask
        ann = Image.fromarray(mask, mode='P')
        ann.save(save_path + file_name.replace('.jpg', '.png'))
        gt = gt * 110
        gt = Image.fromarray(gt)
        gt.save('/user34/fuhao/datasets/urpc2020/coco/gt/' + file_name)



if __name__ == '__main__':


    json_path = '/user34/fuhao/datasets/urpc2020/coco/annotations/instances_test.json'
    image_path = '/user34/fuhao/datasets/urpc2020/coco/ann_ori/test/'
    save_path = '/user34/fuhao/datasets/urpc2020/coco/ann/test/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img2seg(json_path, image_path, save_path)