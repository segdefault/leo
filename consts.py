max_workers = 12

preview_dir = "preview"
model_path = "model"
generator_path = 'generator.hdf5'
discriminator_path = 'discriminator.hdf5'

original_masks_path = "celeb_a/masks"
original_images_path = "celeb_a/images"
images_data_path = "/tmp/data/images"
masks_data_path = "/tmp/data/masks"
celeb_a_attributes_path = "CelebAMask-HQ-attribute-anno.txt"

canvas_size = (512, 512)
image_size = (256, 256)
input_shape = image_size + (3,)
output_shape = image_size + (3,)

palette = {
    'neck': (71, 16, 0),
    'skin': (71, 15, 47),
    'mouth': (0, 64, 100),
    'l_lip': (0, 49, 100),
    'u_lip': (39, 100, 97),
    'l_eye': (0, 95, 90),
    'r_eye': (1, 73, 75),
    'l_brow': (36, 38, 98),
    'r_brow': (4, 8, 89),
    'nose': (121, 79, 71),
    'eye_g': (43, 100, 39),
    'hair': {
        'black': (1, 4, 63),
        'blond': (0, 255, 242),
        'brown': (0, 91, 179),
        'gray': (167, 212, 255),
    },
    'l_ear': (1, 64, 0),
    'r_ear': (100, 86, 8),
    'ear_r': (89, 56, 0),
    'hat': (57, 16, 83),
}
