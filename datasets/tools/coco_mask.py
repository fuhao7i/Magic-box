import cv2
import json
import numpy as np

from PIL import Image
from PIL import ImageDraw

image_path = '/home/fuchenping/chen-mmlab/nas/fuhao/joyboy/mask_datasets/URPC2021/coco/images/test/'

save_path = '/home/fuchenping/chen-mmlab/nas/fuhao/joyboy/mask_datasets/URPC2021/coco/masked/test/'

json_path = '/home/fuchenping/chen-mmlab/nas/fuhao/joyboy/mask_datasets/URPC2021/coco/annotations/instances_test.json'
with open(json_path,'r') as f:
    load_dict = json.load(f)

print(load_dict.keys())
images = load_dict['images']
# print(images)
for index in images:
    image_id = index['id']
    file_name = index['file_name']

    bboxes = [i['bbox'] for i in load_dict['annotations'] if i['image_id'] == image_id]

    # img = Image.open('c000001.jpg')

    img = cv2.imread(image_path + file_name)
    mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    for bbox in bboxes:

        x0 = bbox[0]
        y0 = bbox[1]

        x1 = bbox[0] + bbox[2]
        y1 = bbox[1] + bbox[3]

        mask[y0:y1, x0:x1] = 255

        # draw = ImageDraw.Draw(img)
        # draw.rectangle([x0,y0,x1,y1],fill=(0,0,0,0))
        # del draw

    img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    cv2.imwrite(save_path + file_name, img)
# img.save('1.jpg')
