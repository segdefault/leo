from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import numpy as np
import keras
import sys
import cv2

import model
import data
import consts

continue_training = False

def save_preview(g_model, n_patch):
    [X_realA, X_realB], y_real = data.real_batch(n_patch)
    X_fakeB, _ = data.fake_batch(g_model, X_realA, n_patch)

    imgs = np.concatenate((X_realA, X_realB, X_fakeB), axis=1)

    for i in range(imgs.shape[0]):
        img = cv2.cvtColor(imgs[i] * 255., cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{consts.preview_dir}/{i}.jpg", img)


def train(continue_training=False):
    if continue_training:
        print("Loading old models...")
        discriminator = keras.models.load_model(consts.discriminator_path)
        generator = keras.models.load_model(consts.generator_path)
    else:
        print("Building new models...")
        discriminator = model.build_discriminator()
        generator = model.build_generator()

    gan = model.build_gan(generator, discriminator)
    patch_length = discriminator.output_shape[1]

    for i in range(sys.maxsize):
        [sketches, real_images], y_real = data.real_batch(patch_length)
        fake_images, y_fake = data.fake_batch(generator, sketches, patch_length)

        real_loss = discriminator.train_on_batch([sketches, real_images], y_real)
        fake_loss = discriminator.train_on_batch([sketches, fake_images], y_fake)

        generator_loss, _, _ = gan.train_on_batch(
            sketches, [y_real, real_images])

        print("STEP [%d]> losses: real[%.3f] fake[%.3f] g[%.3f]" %
              (i+1, real_loss, fake_loss, generator_loss))

        if (i+1) % (10) == 0:
            discriminator.save(f"{consts.discriminator_path}")
            generator.save(f"{consts.generator_path}")
            save_preview(generator, patch_length)


if __name__ == "__main__":
    train(continue_training=continue_training)
