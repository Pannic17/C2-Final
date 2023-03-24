import math

import cv2
import time
import keras_cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

HAAR_CASCADE_PATH = "haarcascade_frontalface_alt2.xml"


def detect(origin, cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    # origin = cv2.imread(filename)
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    result = cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=5)

    print(result)

    # create a new white image the same size as our original
    white = np.ones_like(origin) * 255

    for (x, y, w, h) in result:
        wd = math.ceil(w * 0.1)
        hd = math.ceil(h * 0.1)
        print("##################RESULT")
        print(w, h)
        print(x, y)
        print("END#####################")

        # white = cv2.rectangle(white, (x, y), (x + w, y + h), (0, 0, 0), -1)
        # draw a circle on the white image where the face is
        white = cv2.circle(white, (x + w // 2, y + h // 2), wd, (0, 0, 0), -1)

    white = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    return white


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


def generate_image(model, hash_string):
    # modern pop art background for poster with large geometric pattern,
    # a astronaut sit on the ring of a hot jupiter watching the permanent sunset of a wolf-rayet star
    # "modern pop art background for poster with large geometric pattern, concept art, tifo" +
    hash_text = "modern pop art background for poster with large geometric pattern, concept art, tifo, " + hash_string
    # + "pop art"
    images = model.text_to_image(hash_text, batch_size=1)
    # cv2.imshow("KERAS 01", images[0])
    # cv2.waitKey(0)
    return images[0]


def get_adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)
    # cv2.imshow("ADAPTIVE THRESHOLD", thresh)
    # cv2.waitKey(0)
    return thresh


def cast_image_with_mask(image, mask, generated):
    # generate image of 512x512
    new_image = cv2.resize(image, (512, 512))
    # iterate over all pixels
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # if mask is black, set pixel to generated
            if mask[i, j] == 0:
                new_image[i, j] = generated[i, j]
            elif mask[i, j] == 255:
                new_image[i, j] = image[i, j]
    cv2.imshow("RESULT", new_image)
    cv2.waitKey(0)
    return image


def get_hash_to_string(image):
    p_hash_func = cv2.img_hash.PHash_create()
    p_hash = p_hash_func.compute(image)[0]
    hash_string = ""
    for i in p_hash:
        hash_string += str(i) + ", "
    print(hash_string)
    return hash_string


def get_face_mean_mask(image, face):
    # get the mean adaptive threshold of the image
    new_image = image.copy()
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_face = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    thresh_body = get_adaptive_threshold(image)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            # if mask is black, set pixel to generated
            if face[i, j] == 0:
                new_image[i, j] = 255 - thresh_face[i, j]
            elif face[i, j] == 255:
                new_image[i, j] = thresh_body[i, j]
    return new_image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("START")
    # detect("Test8.png", HAAR_CASCADE_PATH)
    image = cv2.imread("Test8.png")
    # resize image to 512x512
    image = cv2.resize(image, (512, 512))
    cv2.imshow("ORIGIN", image)

    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

    hash_string = get_hash_to_string(image)

    # hash_text = "modern pop art background for poster with large geometric pattern, concept art, tifo， " + hash_string

    # keras_image = keras.preprocessing.image.load_img("Test4.png", target_size=(512, 512))
    # keras_image = keras.preprocessing.image.img_to_array(keras_image)
    # keras_mask = keras.preprocessing.image.load_img("Test4_mask.png", color_mode="grayscale", target_size=(512, 512))
    # keras_mask = keras.preprocessing.image.img_to_array(keras_mask)
    # keras_image = tf.shape(image)
    # keras_mask = tf.shape(mask)

    generated = generate_image(model, hash_string)
    cv2.imshow("GENERATED", generated)

    face = detect(image, HAAR_CASCADE_PATH)
    mask = get_face_mean_mask(image, face)
    cv2.imshow("MASK", mask)

    result = cast_image_with_mask(image, mask, generated)
    cv2.imshow("RESULT", result)

    cv2.waitKey(0)
