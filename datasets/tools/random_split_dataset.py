import os
import argparse

import numpy
import random
import shutil
from tqdm import tqdm, trange

def random_split(config, images_path, labels_path, train_rate):
    filenames = os.listdir(images_path)
    total_nums = len(filenames)

    random.shuffle(filenames)
    offset = int(total_nums * train_rate)

    for name in tqdm(filenames[:offset], desc='generate train dataset'):
        shutil.copy(images_path + name, config.save_train_img_path + name)
        shutil.copy(labels_path + name, config.save_train_lab_path + name)

    for name in tqdm(filenames[offset:], desc='generate test dataset'):
        shutil.copy(images_path + name, config.save_test_img_path + name)
        shutil.copy(labels_path + name, config.save_test_lab_path + name)
    
    print('split completed!')
    print('total nums:', total_nums, 'train nums:', offset, 'test nums:', total_nums - offset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', default="./LSUI/backup/input/",help='path of input images', type=str)
    parser.add_argument('--labels_path', default="./LSUI/backup/GT/",   help='path of label images', type=str)
    parser.add_argument('--train_rate',  default=0.8, help="the rate of train_dataset and the rate of test_dataset is 1 - train_rate")

    parser.add_argument('--save_train_img_path', default="./generate/train/images/", help="the path to save train_dataset")
    parser.add_argument('--save_train_lab_path', default="./generate/train/labels/", help="the path to save train_dataset")
    parser.add_argument('--save_test_img_path',  default="./generate/test/images/",  help="the path to save test_dataset")
    parser.add_argument('--save_test_lab_path',  default="./generate/test/labels/",  help="the path to save test_dataset")


    config = parser.parse_args()
    
    if not os.path.exists(config.save_train_img_path):
        os.makedirs(config.save_train_img_path)
    if not os.path.exists(config.save_train_lab_path):
        os.makedirs(config.save_train_lab_path)
    if not os.path.exists(config.save_test_img_path):
        os.makedirs(config.save_test_img_path)
    if not os.path.exists(config.save_test_lab_path):
        os.makedirs(config.save_test_lab_path)
    
    random_split(config, config.images_path, config.labels_path, config.train_rate)
