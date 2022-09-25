import numpy as np
import keras
import cv2

import preprocess
import consts

def increase_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l2 = clahe.apply(l)

    lab = cv2.merge((l2,a,b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

generator = keras.models.load_model(consts.generator_path)
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = preprocess.reshape(frame)

    sketch = preprocess.to_sketch(frame).astype(np.uint8)
    sketch = increase_contrast(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR))
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    sketch = sketch.astype(np.float) / 255.

    image = sketch.reshape((1,) + consts.input_shape)
    image = generator(image)

    sketch = sketch.reshape(consts.input_shape)
    cv2.imshow('Mirror', sketch)
    sketch = (255. * sketch).astype(np.uint8)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB) / 255.
    image = image.numpy().reshape(consts.output_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    concat = np.concatenate((frame / 255., sketch, image), axis=1)

    cv2.imshow('Mirror', concat)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()