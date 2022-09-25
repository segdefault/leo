from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import shutil
import cv2

import consts
import data


def preprocess_mask(masks, mask_name):
    print(f"Processing: {mask_name}")
    image_id = mask_name.split('.')[0]
    image_id = image_id.lstrip('0') or '0'
    image_path = f"{consts.original_images_path}/{image_id}.jpg"

    image = cv2.imread(image_path)
    image = cv2.resize(image, consts.image_size)

    rgb_mask = np.zeros_like(image)
    gray_mask = np.zeros_like(image)

    for mask_type in consts.palette:
        if mask_type in masks[mask_name]:
            sub_mask_path = f"{consts.original_masks_path}/{masks[mask_name][mask_type]}"
            sub_mask = cv2.imread(sub_mask_path)
            sub_mask = cv2.resize(sub_mask, consts.image_size)

            gray_mask = cv2.add(gray_mask, sub_mask)
            rgb_mask = cv2.bitwise_and(rgb_mask, cv2.bitwise_not(sub_mask))

            indices = np.where(sub_mask != 0)
            sub_mask[indices[0], indices[1], :] = data.get_mask_color(image_id, mask_type)
            rgb_mask = cv2.add(rgb_mask, sub_mask)

    image = cv2.bitwise_and(image, gray_mask)

    cv2.imwrite(f"{consts.images_data_path}/blank/{mask_name}", image)
    cv2.imwrite(f"{consts.masks_data_path}/blank/{mask_name}", rgb_mask)


def group_masks_in_dict(masks, filename):
    mask_name = f"{filename.split('_')[0]}.{filename.split('.')[1]}"
    mask_type = f"{'_'.join(filename.split('.')[0].split('_')[1:])}"

    if mask_name in masks:
        masks[mask_name][mask_type] = filename
    else:
        masks[mask_name] = {mask_type: filename}


def preprocess_masks():
    shutil.rmtree(f"{consts.masks_data_path}", ignore_errors=True)
    shutil.rmtree(f"{consts.images_data_path}", ignore_errors=True)
    os.makedirs(f"{consts.masks_data_path}/blank", exist_ok=True)
    os.makedirs(f"{consts.images_data_path}/blank", exist_ok=True)

    mask_files = os.listdir(f"{consts.original_masks_path}")
    masks = {}
    for filename in mask_files:
        group_masks_in_dict(masks, filename)

    with ThreadPoolExecutor(max_workers=consts.max_workers) as executor:
        list(executor.map(lambda mask_name: preprocess_mask(masks, mask_name), masks))


if __name__ == "__main__":
    preprocess_masks()
