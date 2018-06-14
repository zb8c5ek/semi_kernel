import numpy as np


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