#!/usr/bin/env python
# coding: utf-8


# TODO: make it pip installable


import os
import sys
import numpy as np
import click
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


def crop_image(image, box):
    ''' 
        crop image from given box anchors
        boxes are always square and centered around the object
        The box is symmetrically padded before and after to prevent 
        problems with objects close to boundaries
        
        box : [y1, x1, y2, x2]
    '''    
    y1, x1, y2, x2 = box
    w, h = image.shape[:2]
    boxw, boxh = abs(x1 - x2), abs(y1 - y2)
    delta_box = abs(boxw - boxh)
    _image = np.pad(image, ((delta_box, delta_box),
                            (delta_box, delta_box),
                            (0, 0)),
                   mode="symmetric")
    #     x1c = x1 + delta
    crop_w = delta_box
    if crop_w % 2 != 0:
        crop_w += 1
    crop_x1 = int(x1 - crop_w / 2 + delta_box)
    crop_x2 = int(x2 + crop_w / 2 + delta_box)
    crop_y1 = int(y1 - crop_w / 2 + delta_box)
    crop_y2 = int(y2 + crop_w / 2 + delta_box)
    im_crop = _image[crop_y1:crop_y2, crop_x1:crop_x2]
    return im_crop

def filter_roi(rois, _filter):
    ''' filter ROIs according to filter array (boolean list) '''
    assert rois.shape[0] == len(_filter)
    return rois[_filter, :]

def filter_scores(scores, _filter):
    ''' filter scores according to filter list '''
    assert len(scores) == len(_filter)
    return [scores[i] for i in range(len(scores)) if _filter[i]]

def filter_class_ids(class_ids, _filter):
    ''' filter scores according to filter list '''
    assert len(class_ids) == len(_filter)
    return class_ids[_filter]

def filter_masks(masks, _filter):
    ''' filter masks according to filter list '''
    assert masks.shape[-1] == len(_filter)
    return np.concatenate([masks[..., i] for i in range(len(_filter)) if _filter[i]])[..., np.newaxis]

def postprocess_mask(mask):
    ''' boolean array to float image '''
    return mask.squeeze() * 1.0


def infer_file(f_path, model, class_names):
    image_dir = os.path.dirname(f_path)
    ''' run mask rcnn from image provided path to image'''
    image = skimage.io.imread(f_path)

    in_name, in_ext = os.path.splitext(os.path.basename(f_path))

    results = model.detect([image], verbose=1)

    r = results[0]

    target_class = "cat"
    c_idx = class_names.index(target_class)
    _filter = [True if _idx == c_idx else False for _idx in r["class_ids"]]

    if True in _filter:
        rr = {
            "rois" : filter_roi(r["rois"], _filter),
            "scores" : filter_scores(r["scores"], _filter),
            "masks" : filter_masks(r["masks"], _filter),
            "class_ids" : filter_class_ids(r["class_ids"], _filter)
        }
        box = list(rr["rois"].squeeze())
        im_crop = crop_image(image, box)
        mask_crop = crop_image(rr["masks"], box)

        out_dir = os.path.join(image_dir, os.pardir, "cropped")
        os.makedirs(out_dir, exist_ok=True)
        out_name = ''.join(["cropped_", in_name, in_ext])
        out_name = os.path.join(out_dir, out_name)
        skimage.io.imsave(out_name, im_crop)

        out_dir = os.path.join(image_dir, os.pardir, "mask")
        os.makedirs(out_dir, exist_ok=True)
        out_name = ''.join(["mask_", in_name, in_ext])
        out_name = os.path.join(out_dir, out_name)
        skimage.io.imsave(out_name, postprocess_mask(mask_crop))
    else:
        print("no cat found")


def infer_folder(folder_path, model, class_names):
    files = os.listdir(folder_path)
    for file in files:
        infer_file(os.path.join(folder_path, file), model, class_names)

@click.command()
@click.argument("file_or_folder")
@click.argument("target_class")
def main(file_or_folder, target_class):
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']


    if target_class not in class_names:
        print("target class {} not in class names".format(target_class))
        print(class_names)

    if os.path.isdir(file_or_folder):
        infer_folder(file_or_folder, model, class_names)
    else:
        infer_file(file_or_folder, model, class_names)




if __name__ == '__main__':
    main()