#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import PIL.ImageDraw
from tqdm import tqdm
import cv2

IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.webp'}
SEMI_SUPERVISED_TRAIN_RATIO = 0.8


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def images_labelme(data, image_id, file_name=None):
    image = {}
    image['height'] = data['imageHeight']
    image['width'] = data['imageWidth']
    image['id'] = image_id
    if file_name is not None:
        image['file_name'] = file_name
    elif '\\' in data['imagePath']:
        image['file_name'] = data['imagePath'].split('\\')[-1]
    else:
        image['file_name'] = data['imagePath'].split('/')[-1]
    return image


def images_cityscape(data, image_id, img_file):
    image = {}
    image['height'] = data['imgHeight']
    image['width'] = data['imgWidth']
    image['id'] = image_id
    image['file_name'] = img_file
    return image


def categories(label, labels_list):
    category = {}
    category['supercategory'] = 'component'
    category['id'] = len(labels_list) + 1
    category['name'] = label
    return category


def annotations_rectangle(points, label, image_id, annotation_id, label_to_num):
    annotation = {}
    seg_points = np.asarray(points).copy()
    seg_points[1, :] = np.asarray(points)[2, :]
    seg_points[2, :] = np.asarray(points)[1, :]
    annotation['segmentation'] = [list(seg_points.flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_id
    annotation['bbox'] = list(
        map(float, [
            points[0][0], points[0][1], points[1][0] - points[0][0], points[1][
                1] - points[0][1]
        ]))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = annotation_id
    return annotation


def annotations_polygon(height, width, points, label, image_id, annotation_id,
                        label_to_num):
    annotation = {}
    annotation['segmentation'] = [list(np.asarray(points).flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_id
    annotation['bbox'] = list(map(float, get_bbox(height, width, points)))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = annotation_id
    return annotation


def get_bbox(height, width, points):
    polygons = points
    mask = np.zeros([height, width], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)
    return [
        left_top_c, left_top_r, right_bottom_c - left_top_c,
        right_bottom_r - left_top_r
    ]


def is_image_file(file_name):
    return osp.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS


def list_image_items(image_dir):
    items = []
    for file_name in sorted(os.listdir(image_dir), key=str.lower):
        if not is_image_file(file_name):
            continue
        items.append({
            'stem': osp.splitext(file_name)[0],
            'file_name': file_name,
            'image_path': osp.join(image_dir, file_name)
        })
    return items


def index_json_files(json_dir):
    json_files = {}
    for file_name in sorted(os.listdir(json_dir), key=str.lower):
        if not file_name.lower().endswith('.json'):
            continue
        json_files[osp.splitext(file_name)[0]] = osp.join(json_dir, file_name)
    return json_files


def collect_labelme_items(image_dir, json_dir, require_json):
    labeled_items = []
    unlabeled_items = []
    missing_labels = []
    json_files = index_json_files(json_dir)

    for image_item in list_image_items(image_dir):
        label_path = json_files.get(image_item['stem'])
        dataset_item = dict(image_item)
        dataset_item['json_path'] = label_path
        if label_path is None:
            if require_json:
                missing_labels.append(image_item['file_name'])
            else:
                unlabeled_items.append(dataset_item)
            continue
        labeled_items.append(dataset_item)

    if missing_labels:
        preview = ', '.join(missing_labels[:5])
        if len(missing_labels) > 5:
            preview += ', ...'
        raise FileNotFoundError(
            'Missing Labelme JSON files for source images: {}'.format(preview))

    return labeled_items, unlabeled_items


def split_train_val(items, train_ratio=SEMI_SUPERVISED_TRAIN_RATIO):
    split_index = int(len(items) * train_ratio)
    return items[:split_index], items[split_index:]


def load_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def build_categories_from_label_map(label_to_num):
    categories_list = []
    for label, label_id in sorted(label_to_num.items(), key=lambda item: item[1]):
        categories_list.append({
            'supercategory': 'component',
            'id': label_id,
            'name': label
        })
    return categories_list


def build_label_map_for_labelme(labeled_items):
    label_to_num = {}
    for item in labeled_items:
        data = load_json_file(item['json_path'])
        for shape in data.get('shapes', []):
            label = shape['label']
            if label not in label_to_num:
                label_to_num[label] = len(label_to_num) + 1
    return label_to_num, build_categories_from_label_map(label_to_num)


def get_image_size(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Failed to read image: {}'.format(image_path))
    return image.shape[0], image.shape[1]


def append_labelme_annotations(data, image_id, next_annotation_id,
                               label_to_num, annotations_list):
    for shape in data.get('shapes', []):
        label = shape['label']
        shape_type = shape.get('shape_type', 'polygon')
        if shape_type == 'polygon':
            annotations_list.append(
                annotations_polygon(data['imageHeight'], data['imageWidth'],
                                    shape['points'], label, image_id,
                                    next_annotation_id, label_to_num))
            next_annotation_id += 1
        elif shape_type == 'rectangle':
            (x1, y1), (x2, y2) = shape['points']
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            points = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]]
            annotations_list.append(
                annotations_rectangle(points, label, image_id,
                                      next_annotation_id, label_to_num))
            next_annotation_id += 1
    return next_annotation_id


def build_unlabeled_image_entry(item, image_id):
    height, width = get_image_size(item['image_path'])
    return {
        'height': height,
        'width': width,
        'id': image_id,
        'file_name': item['file_name']
    }


def build_labelme_coco(items, label_to_num, categories_list):
    data_coco = {
        'images': [],
        'categories': categories_list,
        'annotations': []
    }
    next_annotation_id = 1
    for image_id, item in enumerate(items, start=1):
        if item.get('json_path'):
            data = load_json_file(item['json_path'])
            data_coco['images'].append(
                images_labelme(data, image_id, item['file_name']))
            next_annotation_id = append_labelme_annotations(
                data, image_id, next_annotation_id, label_to_num,
                data_coco['annotations'])
        else:
            data_coco['images'].append(build_unlabeled_image_entry(item, image_id))
    return data_coco


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def write_coco_json(output_path, data_coco):
    with open(output_path, 'w', encoding='utf-8') as file_obj:
        json.dump(data_coco, file_obj, indent=4, cls=MyEncoder)


def copy_partition_images(items, output_dir):
    ensure_directory(output_dir)
    for item in items:
        shutil.copyfile(item['image_path'], osp.join(output_dir, item['file_name']))


def get_duplicate_file_names(items):
    duplicates = []
    seen = set()
    for item in items:
        file_name = item['file_name']
        if file_name in seen and file_name not in duplicates:
            duplicates.append(file_name)
            continue
        seen.add(file_name)
    return duplicates


def convert_labelme_semi_supervised(args, parser):
    required_args = {
        '--source_image_dir': args.source_image_dir,
        '--source_json_dir': args.source_json_dir,
        '--target_image_dir': args.target_image_dir,
        '--target_json_dir': args.target_json_dir,
    }
    missing_args = [name for name, value in required_args.items() if not value]
    if missing_args:
        parser.error(
            'Labelme semi-supervised conversion requires {}'.format(
                ', '.join(missing_args)))

    directory_args = {
        'source image directory': args.source_image_dir,
        'source json directory': args.source_json_dir,
        'target image directory': args.target_image_dir,
        'target json directory': args.target_json_dir,
    }
    for description, path in directory_args.items():
        if not osp.isdir(path):
            parser.error('The {} does not exist: {}'.format(description, path))

    try:
        source_items, _ = collect_labelme_items(
            args.source_image_dir, args.source_json_dir, require_json=True)
    except FileNotFoundError as error:
        parser.error(str(error))
    target_labeled_items, target_unlabeled_items = collect_labelme_items(
        args.target_image_dir, args.target_json_dir, require_json=False)

    source_train_items, source_val_items = split_train_val(source_items)
    target_train_items, target_val_items = split_train_val(target_labeled_items)
    train_st_items = source_train_items + target_train_items

    label_to_num, categories_list = build_label_map_for_labelme(
        source_items + target_labeled_items)

    annotations_dir = osp.join(args.output_dir, 'annotations')
    ensure_directory(annotations_dir)

    output_datasets = {
        'train_st.json': train_st_items,
        'source_val.json': source_val_items,
        'target_val.json': target_val_items,
        'target_unlabeled.json': target_unlabeled_items,
    }
    for output_name, items in output_datasets.items():
        write_coco_json(
            osp.join(annotations_dir, output_name),
            build_labelme_coco(items, label_to_num, categories_list))

    if args.copy_images:
        duplicate_train_names = get_duplicate_file_names(train_st_items)
        if duplicate_train_names:
            preview = ', '.join(duplicate_train_names[:5])
            if len(duplicate_train_names) > 5:
                preview += ', ...'
            parser.error(
                'Cannot copy merged train_st images because duplicate file names exist: {}'
                .format(preview))

        copy_partition_images(train_st_items,
                              osp.join(args.output_dir, 'train_st'))
        copy_partition_images(source_val_items,
                              osp.join(args.output_dir, 'source_val'))
        copy_partition_images(target_val_items,
                              osp.join(args.output_dir, 'target_val'))
        copy_partition_images(target_unlabeled_items,
                              osp.join(args.output_dir, 'target_unlabeled'))

    print('Source labeled: {} total, {} train, {} val'.format(
        len(source_items), len(source_train_items), len(source_val_items)))
    print('Target labeled: {} total, {} train, {} val'.format(
        len(target_labeled_items), len(target_train_items),
        len(target_val_items)))
    print('Merged train_st: {}'.format(len(train_st_items)))
    print('Target unlabeled: {}'.format(len(target_unlabeled_items)))


def voc_get_label_anno(ann_dir_path, ann_ids_path, labels_path):
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))

    with open(ann_ids_path, 'r') as f:
        ann_ids = [lin.strip().split(' ')[-1] for lin in f.readlines()]

    ann_paths = []
    for aid in ann_ids:
        if aid.endswith('xml'):
            ann_path = os.path.join(ann_dir_path, aid)
        else:
            ann_path = os.path.join(ann_dir_path, aid + '.xml')
        ann_paths.append(ann_path)

    return dict(zip(labels_str, labels_ids)), ann_paths


def voc_get_image_info(annotation_root, im_id):
    filename = annotation_root.findtext('filename')
    assert filename is not None
    img_name = os.path.basename(filename)

    size = annotation_root.find('size')
    width = float(size.findtext('width'))
    height = float(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': im_id
    }
    return image_info


def voc_get_coco_annotation(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, "label is not in label2id."
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.findtext('xmin'))
    ymin = float(bndbox.findtext('ymin'))
    xmax = float(bndbox.findtext('xmax'))
    ymax = float(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, "Box size error."
    o_width = xmax - xmin
    o_height = ymax - ymin
    anno = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
    }
    return anno


def voc_xmls_to_cocojson(annotation_paths, label2id, output_dir, output_file):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # bounding box start id
    im_id = 0
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = voc_get_image_info(ann_root, im_id)
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = voc_get_coco_annotation(obj=obj, label2id=label2id)
            ann.update({'image_id': im_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        im_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)
    output_file = os.path.join(output_dir, output_file)
    with open(output_file, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def widerface_to_cocojson(root_path):
    train_gt_txt = os.path.join(root_path, "wider_face_split", "wider_face_train_bbx_gt.txt")
    val_gt_txt = os.path.join(root_path, "wider_face_split", "wider_face_val_bbx_gt.txt")
    train_img_dir = os.path.join(root_path, "WIDER_train", "images")
    val_img_dir = os.path.join(root_path, "WIDER_val", "images")
    assert train_gt_txt
    assert val_gt_txt
    assert train_img_dir
    assert val_img_dir
    save_path = os.path.join(root_path, "widerface_train.json")
    widerface_convert(train_gt_txt, train_img_dir, save_path)
    print("Wider Face train dataset converts sucess, the json path: {}".format(save_path))
    save_path = os.path.join(root_path, "widerface_val.json")
    widerface_convert(val_gt_txt, val_img_dir, save_path)
    print("Wider Face val dataset converts sucess, the json path: {}".format(save_path))


def widerface_convert(gt_txt, img_dir, save_path):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [{'supercategory': 'none', 'id': 0, 'name': "human_face"}]
    }
    bnd_id = 1  # bounding box start id
    im_id = 0
    print('Start converting !')
    with open(gt_txt) as fd:
        lines = fd.readlines()

    i = 0
    while i < len(lines):
        image_name = lines[i].strip()
        bbox_num = int(lines[i + 1].strip())
        i += 2
        img_info = get_widerface_image_info(img_dir, image_name, im_id)
        if img_info:
            output_json_dict["images"].append(img_info)
            for j in range(i, i + bbox_num):
                anno = get_widerface_ann_info(lines[j])
                anno.update({'image_id': im_id, 'id': bnd_id})
                output_json_dict['annotations'].append(anno)
                bnd_id += 1
        else:
            print("The image dose not exist: {}".format(os.path.join(img_dir, image_name)))
        bbox_num = 1 if bbox_num == 0 else bbox_num
        i += bbox_num
        im_id += 1
    with open(save_path, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def get_widerface_image_info(img_root, img_relative_path, img_id):
    image_info = {}
    save_path = os.path.join(img_root, img_relative_path)
    if os.path.exists(save_path):
        img = cv2.imread(save_path)
        image_info["file_name"] = os.path.join(os.path.basename(
            os.path.dirname(img_root)), os.path.basename(img_root),
            img_relative_path)
        image_info["height"] = img.shape[0]
        image_info["width"] = img.shape[1]
        image_info["id"] = img_id
    return image_info


def get_widerface_ann_info(info):
    info = [int(x) for x in info.strip().split()]
    anno = {
        'area': info[2] * info[3],
        'iscrowd': 0,
        'bbox': [info[0], info[1], info[2], info[3]],
        'category_id': 0,
        'ignore': 0,
        'blur': info[4],
        'expression': info[5],
        'illumination': info[6],
        'invalid': info[7],
        'occlusion': info[8],
        'pose': info[9]
    }
    return anno


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_type',
        help='the type of dataset, can be `voc`, `widerface`, `labelme` or `cityscape`')
    parser.add_argument('--json_input_dir', help='legacy annotated directory')
    parser.add_argument('--image_input_dir', help='legacy image directory')
    parser.add_argument(
        '--source_image_dir',
        help='source domain image directory for labelme semi-supervised conversion')
    parser.add_argument(
        '--source_json_dir',
        help='source domain json directory for labelme semi-supervised conversion')
    parser.add_argument(
        '--target_image_dir',
        help='target domain image directory for labelme semi-supervised conversion')
    parser.add_argument(
        '--target_json_dir',
        help='target domain json directory for labelme semi-supervised conversion')
    parser.add_argument(
        '--copy_images',
        action='store_true',
        help='copy images into split directories under output_dir')
    parser.add_argument(
        '--output_dir', help='output dataset directory', default='./')
    parser.add_argument(
        '--train_proportion',
        help='the proportion of train dataset',
        type=float,
        default=0.8)
    parser.add_argument(
        '--val_proportion',
        help='the proportion of validation dataset',
        type=float,
        default=0.2)
    parser.add_argument(
        '--test_proportion',
        help='the proportion of test dataset',
        type=float,
        default=0.0)
    parser.add_argument(
        '--voc_anno_dir',
        help='In Voc format dataset, path to annotation files directory.',
        type=str,
        default=None)
    parser.add_argument(
        '--voc_anno_list',
        help='In Voc format dataset, path to annotation files ids list.',
        type=str,
        default=None)
    parser.add_argument(
        '--voc_label_list',
        help='In Voc format dataset, path to label list. The content of each line is a category.',
        type=str,
        default=None)
    parser.add_argument(
        '--voc_out_name',
        type=str,
        default='voc.json',
        help='In Voc format dataset, path to output json file')
    parser.add_argument(
        '--widerface_root_dir',
        help='The root_path for wider face dataset, which contains `wider_face_split`, `WIDER_train` and `WIDER_val`.And the json file will save in this path',
        type=str,
        default=None)
    args = parser.parse_args()
    try:
        assert args.dataset_type in ['voc', 'labelme', 'cityscape', 'widerface']
    except AssertionError as e:
        print(
            'Now only support the voc, cityscape dataset and labelme dataset!!')
        os._exit(0)

    if args.dataset_type == 'voc':
        assert args.voc_anno_dir and args.voc_anno_list and args.voc_label_list
        label2id, ann_paths = voc_get_label_anno(
            args.voc_anno_dir, args.voc_anno_list, args.voc_label_list)
        voc_xmls_to_cocojson(
            annotation_paths=ann_paths,
            label2id=label2id,
            output_dir=args.output_dir,
            output_file=args.voc_out_name)
    elif args.dataset_type == "widerface":
        assert args.widerface_root_dir
        widerface_to_cocojson(args.widerface_root_dir)
    elif args.dataset_type == 'cityscape':
        parser.error(
            'The source/target semi-supervised workflow is implemented for labelme only.')
    else:
        convert_labelme_semi_supervised(args, parser)


if __name__ == '__main__':
    main()
