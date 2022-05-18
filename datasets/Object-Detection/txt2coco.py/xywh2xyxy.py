"""
YOLO 格式的数据集转化为 COCO 格式的数据集
--root_dir 输入根路径
--save_path 保存文件的名字(没有random_split时使用)
--random_split 有则会随机划分数据集，然后再分别保存为3个文件。
"""

import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str,default='./', help="path to save json files")
parser.add_argument('--images', default='Image/', help='images path')
parser.add_argument('--labels', default='Annotation/', help='labels path')

parser.add_argument('--random_split', default=True, action='store_true', help="random split the dataset, default ratio is 8:1:1")
args = parser.parse_args()

def train_test_val_split(img_paths,ratio_train=0.8,ratio_test=0.2):

    train_img, test_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    print("NUMS of train:val:test = {}:{}".format(len(train_img), len(test_img)))
    return train_img, test_img


def yolo2coco(root_path, random_split):
    originLabelsDir = os.path.join(root_path, args.labels)                                        
    originImagesDir = os.path.join(root_path, args.images)
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    # images dir name
    indexes = os.listdir(originImagesDir)

    if random_split:
        # 用于保存所有数据的图片信息和标注信息
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        # 建立类别标签和数字id的对应关系, 类别id从0开始。
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        train_img, test_img = train_test_val_split(indexes)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(root_path, args.images) + index)
        height, width, _ = im.shape
        if random_split:
            # 切换dataset的引用对象，从而划分数据集
                if index in train_img:
                    dataset = train_dataset
                elif index in test_img:
                    dataset = test_dataset
        # 添加图像的信息
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                if label.strip() == '':
                    continue
                label = label.strip().split(',')
                # print(txtFile, label)
                x = float(label[0])
                y = float(label[1])
                w = float(label[2])
                h = float(label[3])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                # x1 = (x - w / 2) 
                # y1 = (y - h / 2) 
                # x2 = (x + w / 2) 
                # y2 = (y + h / 2)
                x1 = x
                y1 = y
                x2 = w - x
                y2 = h - y
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                # cls_id = int(label[0])   
                cls_id = 0
                width = x2
                height = y2
                # width = max(0, x2 - x1)
                # height = max(0, y2 - y1)
                # :)
                # img = cv2.imread(os.path.join(root_path, args.images) + index)
                # cv2.imwrite('./' + index, img)
                # # 画矩形框 距离靠左靠上的位置
                # pt1 = (x1, y1) #左边，上边   #数1 ， 数2
                # pt2 = (x2, y2) #右边，下边  #数1+数3，数2+数4

                # cv2.rectangle(img, (int(x1), int(y1)), (int(x1+x2), int(y1+y2)), (0, 255, 0), 2)

                # a = cls_id #类别名称
                # b = 7.77 #置信度
                # font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
                # imgzi = cv2.putText(img, '{} {:.3f}'.format(a,b), (int(x1), int(y1)-15), font, 1, (0, 255, 255), 4)
                #                  # 图像，      文字内容，      坐标(右上角坐标)，字体， 大小，  颜色，    字体厚度
                # cv2.imwrite('./' + index, img)
                # raise
                # :(
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if random_split:
        for phase in ['train','test']:
            json_name = os.path.join(root_path, '{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, '{}'.format('train.json'))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    root_path = args.root_dir
    assert os.path.exists(root_path)
    random_split = args.random_split
    print("Loading data from ",root_path,"\nWhether to split the data:",random_split)
    yolo2coco(root_path,random_split)