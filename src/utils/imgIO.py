import numpy as np
from PIL import Image
__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
version: 4.0.0 beta
PhD student at PSI-ESAT, KU Leuven
Supervisor: Prof. Luc Van Gool
Research Domain: Computer Vision, Machine Learning

Address:
Kasteelpark Arenberg 10 - bus 2441
B-3001 Heverlee
Belgium

Group website: http://www.esat.kuleuven.be/psi/visics
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""


def read_image(fn, gray=False):
    """
    read image using PIL, and return as numpy array.
    :param fn: filename of the target image
    :param gray: whether convert into gray scale
    :return: img as numpy array
    """
    if gray:
        file = Image.open(fn).convert('L')  # if original image is gray scale, then directly read without conversion.
        img = np.asarray(file).astype(np.uint8)
    else:
        file = Image.open(fn)
        img = np.asarray(file)

    return img


def rgb2gray(img):
    """
    convert image from RGB(0=R,1=G,2=B) * in OpenCV it is BGR
    :param img: rgb image to be converted
    :return: gray_image as numpy array in Int
    """
    gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return gray_img


def write_img(fn, img):
    """
    output img(np array format) into a file.
    :param fn: filename of output image
    :param img: image in np.array format
    :return: a flag
    """
    img_PIL = Image.fromarray(img)
    img_PIL.save(fn)
