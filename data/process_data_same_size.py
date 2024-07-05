import cv2
import logging
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

root = 'INSERT_DATA_ROOT'
out_root = 'INSERT_OUTPUT_DIR'
# cls_list = ['AMD', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Other']
cls_list = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
target_size = [405, 540] # (1024, 1024), can be customized.


def random_crop(image, crop_shape, padding=10):
    img_h = image.shape[0]
    img_w = image.shape[1]
    img_d = image.shape[2]

    oshape_h = img_h + 2 * padding
    oshape_w = img_w + 2 * padding
    img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
    img_pad[padding:padding+img_h, padding:padding+img_w, 0:img_d] = image

    # nh = random.randint(0, oshape_h - crop_shape[0])
    # nw = random.randint(0, oshape_w - crop_shape[1])
    nh = (oshape_h-crop_shape[0])//2
    nw = (oshape_w-crop_shape[1])//2
    image_crop = img_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]

    return image_crop

for cls in cls_list:
    if not os.path.exists(os.path.join(out_root, cls)):
        os.makedirs(os.path.join(out_root, cls))
    cls_root = os.path.join(root, cls)
    item_list = os.listdir(cls_root)
    for item in tqdm(item_list):
        image_path = os.path.join(cls_root, item)
        # print(image_path)
        image = cv2.imread(image_path)
        img_h = image.shape[0]
        img_w = image.shape[1]
        # img_d = image.shape[2]
        # w_sameratio = int(1024 / img_h * img_w)
        # image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        image = random_crop(image, [405, 540], padding=10)
        image_path = os.path.join(out_root, cls, item)
        cv2.imwrite(image_path, image)
        # print(image_path)
