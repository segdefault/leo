from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import pandas

import consts

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)

seed = 1234
x_gen = None
y_gen = None

def get_x_gen():
    global x_gen

    if x_gen is None:
        x_gen = datagen.flow_from_directory(
            f"{consts.masks_data_path}",
            target_size=consts.image_size,
            class_mode=None,
            seed=seed,
        )
    
    return x_gen

def get_y_gen():
    global y_gen

    if y_gen is None:
        y_gen = datagen.flow_from_directory(
            f"{consts.images_data_path}",
            target_size=consts.image_size,
            class_mode=None,
            seed=seed,
        )
    
    return y_gen

def real_batch(patch_shape):
    sketches = get_x_gen().next()
    images = get_y_gen().next()

    real_y = np.ones((sketches.shape[0], patch_shape, patch_shape, 1))

    return [sketches, images], real_y


def fake_batch(g_model, samples, patch_shape):
    fake_images = g_model.predict(samples)
    fake_y = np.zeros((len(fake_images), patch_shape, patch_shape, 1))

    return fake_images, fake_y


def get_attributes():
    with open(consts.celeb_a_attributes_path, "r") as f:
        dataCount = int(f.readline())
        columns = f.readline().split()
        rows = []

        for (i, line) in enumerate(f):
            rows.append(line.split())

    return pandas.DataFrame(rows, columns=columns)


attributes = get_attributes()

def get_hair_color(id):
    row = attributes.iloc[int(id)]

    if int(row['Blond_Hair']) == 1:
        return consts.palette['hair']['blond']
    elif int(row['Brown_Hair']) == 1:
        return consts.palette['hair']['brown']
    elif int(row['Gray_Hair']) == 1:
        return consts.palette['hair']['gray']
    else:
        return consts.palette['hair']['black']

def get_mask_color(image_id, mask_type):
    if mask_type == 'hair':
        return get_hair_color(image_id)
    else:
        return consts.palette[mask_type]
