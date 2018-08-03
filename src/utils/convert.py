import numpy as np
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


def img_double2int(img, min_in=None, max_in=None, max_int=255):
    """
    it converts img from double to int, if no min_in and max_in specified, the min and max from img will be taken.
    :param img: input 2D array / matrix / image
    :param min_in: minimum boundary of image
    :param max_in: maximum boundary of image
    :param max_int: minimum boundary of output image is zero, yet maximum boundary could be specified, default 255.
    :return: converted image
    """

    if min_in is None:
        min_in = np.amin(img, axis=(0, 1), keepdims=False)
        max_in = np.amax(img, axis=(0, 1), keepdims=False)
    img -= min_in
    img = img.astype(dtype=np.float) * max_int / max_in
    img = np.round(img)

    img = np.array(img, dtype=np.int)

    return img