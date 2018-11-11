
import os

from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
slim.separable_conv2d
PATH_TO_TEST_IMAGES_DIR = 'D:\Images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    plt.figure()
    plt.imshow(image)
plt.show()
