import os
import os.path as osp
import random
import shutil
import sys
import json
import glob
import xml.etree.ElementTree as ET
import argparse

# voc_dir = '/data/xiaowenjie/nas/datasets/urpc2021/new_urpc/before/'  #remember to modify the path
# voc_annotations = voc_dir + 'boxes/'
# txt_dir = voc_dir + 'txt/'
# coco_ann_dir = voc_dir + 'coco_ann/'

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_dir):
    """Generate category name to id mapping from a list of xml files.
    Arguments:
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    xml_files = os.listdir(xml_dir)
    xml_files.sort()
    for xml_file in xml_files:
        print('read kinds:', xml_file)
        xml_file = osp.join(xml_dir, xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def voc_to_coco(voc_ann, coco_ann_save_dir, txt_dir):
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(voc_ann)
    bnd_id = START_BOUNDING_BOX_ID
    txts = os.listdir(txt_dir)
    txts.sort()
    
    for txt in txts:
        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
        file = open(osp.join(txt_dir, txt))
        lines = file.readlines()
        lines.sort()
        for i, img_id in enumerate(lines):
            img_id = img_id.split('\n')[0]
            print('voc_to_coco', txt, img_id)
            tree = ET.parse(osp.join(voc_ann, img_id + '.xml'))
            root = tree.getroot()
            filename = img_id + '.jpg'
            image_id = i + 1
            size = get_and_check(root, "size", 1)
            width = int(get_and_check(size, "width", 1).text)
            height = int(get_and_check(size, "height", 1).text)
            image = {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id,
            }
            json_dict["images"].append(image)
            ## Currently we do not support segmentation.
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, "object"):
                category = get_and_check(obj, "name", 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, "bndbox", 1)
                xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
                ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
                xmax = int(get_and_check(bndbox, "xmax", 1).text)
                ymax = int(get_and_check(bndbox, "ymax", 1).text)
                assert xmax > xmin
                assert ymax > ymin
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

        for cate, cid in categories.items():
            cat = {"supercategory": "none", "id": cid, "name": cate}
            json_dict["categories"].append(cat)

        json_file = osp.join(coco_ann_save_dir, 'instances_' + txt[:-4] + '2017.json')
        os.makedirs(osp.dirname(json_file), exist_ok=True)
        json_fp = open(json_file, "w")
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()
        print(categories)
        print(len(categories.keys()))

if __name__ == '__main__':
    parer = argparse.ArgumentParser()
    parer.add_argument('voc_ann')
    parer.add_argument('txt_dir')
    parer.add_argument('coco_save')
    arg = parer.parse_args()
    voc_to_coco(arg.voc_ann, arg.coco_save, arg.txt_dir)
