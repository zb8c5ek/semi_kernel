import numpy as np
import torch as pt
from src.utils.warping import mesh_xy_coordinates_of_given_2D_dimensions
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


def gaussian_filter(ori_img, theta, cuda=True, crop_size=None, return_kernel=False):
    """
    * this function is implemented in tensor stacking manner.
    Gaussian filtering, using 3*theta x 2 + 1 as the size, so that 99.7% would be covered in theory. crop_size is
    set as an option in case different size of gaussian kernel is in desire. one shall remember that kernel size changes
    automatically with theta, crop_size means only extract part of the kernel as the output, so as crop means.
    *return_kernel is provided as an option in case the kernel itself might be in use e.g. visualisation
    :param ori_img: np array. original image to be filtered;
    :param theta: the variance, here assume both x direction and y direction share the same variance across the kernel;
    :param cuda: whether using cuda or not, assume to be True;
    :param crop_size: the desired size to be cropped from original (3*theta*2+1)** size
    :param return_kernel: bool, whether return the kernel too ! in np.array.
    :return: filtered image: and (optional) applied gaussian kernel, both in np.array format.
    """
    # ----- determine kernel size -----
    kernel_size = np.array(2*np.ceil(3*[theta, theta]), dtype=np.int) + 1  # kernel_size in (x_size, y_size) format

    # ----- generate gaussian kernel -----
    center_shift = np.floor(kernel_size / 2)
    kernel_mesh = mesh_xy_coordinates_of_given_2D_dimensions(kernel_size)
    kernel_mesh[0] -= center_shift[0]
    kernel_mesh[1] -= center_shift[1]

    gaussian_kernel = np.exp(-np.power(kernel_mesh[0]/theta, 2)/2 - np.power(kernel_mesh[1]/theta, 2)/2)
    # TODO: check whether a regulation to 1sum is necessary

    # ----- generate shift list and stacking image shifts -----



def bilaterial_filtering():
    pass


def median_filtering():
    pass


def sobel_filtering():
    pass