# -*- coding: utf-8 -*-

from os import path
import numpy as np
import cv2

def crop_image(image, crop_from, crop_to):
    ## crop the image to focus on the road situation.
    return image[crop_from:crop_to, :, :]


def resize_image(image, new_h, new_w):
    ## resize image from (160, 320) to (NEW_H, NEW_W)
    return cv2.resize(image, (new_w, new_h))


def normalize_256_base(value):
    ## normalize the value with the base of 256 to the value between -1 and 1.
    return value/127.5 - 1.


def normalize_input_base(value):
    ## normalize the value with the base of maximum and minimum input values to the value between 0 and 1. 
    max_val = np.max(value)
    min_val = np.min(value)
    return 0.1 + np.divide((value - min_val)*0.8, (max_val - min_val))


def to_gray_hsv(image):
    ## convert image into HSV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # split the image into H, S, V
    H, S, V = cv2.split(image)
    return normalize_256_base(S)


def to_gray_cv2(image):
    ## convert image into grayscale with cv2 library
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return normalize_256_base(image)

def to_gray_yuv(image):
    ## convert image into YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # split the image into Y, U, V
    Y, U, V = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    Y = clahe.apply(Y)
    return normalize_input_base(Y)

def adjust_brightness(image):
    ## convert image into HSV
    tmp_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ## get the ratio to adjust the brightness
    ratio = np.random.uniform(.25, 1.00)
    ## apply ratio to V channel which is the second channel of HSV format
    tmp_img[:, :, 2] = tmp_img[:, :, 2] * ratio
    ## return RGB color image
    return cv2.cvtColor(tmp_img, cv2.COLOR_HSV2RGB)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(12,6))

    img = plt.imread('../assets/img/center.jpg')

    fig.add_subplot(3,3,1)
    plt.axis('off')
    plt.title("origin", fontsize=12)
    plt.imshow(img)

    cropped_img = np.zeros([img.shape[0], img.shape[1], 3], dtype='uint8')
    cropped_img[53:133, :, :] = crop_image(img, 53, 133)
    fig.add_subplot(3,3,4)
    plt.axis('off')
    plt.title("cropped", fontsize=12)
    plt.imshow(cropped_img)

    resized_img = resize_image(img, 16, 32)
    fig.add_subplot(3,3,5)
    plt.axis('off')
    plt.title("resized 16x32", fontsize=12)
    plt.imshow(resized_img)

    to_gray_img = to_gray_hsv(img)
    fig.add_subplot(3,3,6)
    plt.axis('off')
    plt.title("to gray (S of HSV)", fontsize=12)
    plt.imshow(to_gray_img, cmap='gray')

    to_gray_img = to_gray_yuv(img)
    fig.add_subplot(3,3,7)
    plt.axis('off')
    plt.title("to gray (Y of YUV)", fontsize=12)
    plt.imshow(to_gray_img, cmap='gray')

    to_gray_img = to_gray_cv2(img)
    fig.add_subplot(3,3,8)
    plt.axis('off')
    plt.title("to gray (CV2 lib)", fontsize=12)
    plt.imshow(to_gray_img, cmap='gray')

    brightness_img = adjust_brightness(img)
    fig.add_subplot(3,3,9)
    plt.axis('off')
    plt.title("adjust brightness", fontsize=12)
    plt.imshow(brightness_img)
    
    plt.tight_layout()
    plt.savefig('../assets/img/utils.png')
    plt.show()
    
