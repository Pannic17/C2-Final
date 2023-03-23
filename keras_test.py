

import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

"""
## Introduction
Unlike most tutorials, where we first explain a topic then show how to implement it,
with text-to-image generation it is easier to show instead of tell.
Check out the power of `keras_cv.models.StableDiffusion()`.
First, we construct a model:
"""

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

"""
Next, we give it a prompt:
"""

images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)