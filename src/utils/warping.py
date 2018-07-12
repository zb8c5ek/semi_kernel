import torch as pt
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


def mesh_xy_coordinates_of_given_2D_dimensions(dimensions):
    """
    For a given dimension, say (X, Y), it returns a two 2D matrix. The first matrix contains the x coordinates of the
    matrix, while the second matrix contains the y coordinates of the matrix.
    dimensions = [y_dimension, x_dimension]
    * y(row)_dimension first so that one can simply provides the parameter by Matrix.shape to this function
    :param dimensions: [y_dimension, x_dimension]
                x_dimension: the range of x dimension extends to e.g. 1280
                y_dimension: the range of y dimension extends to e.g. 800
    :return: (X_grid, Y_grid), two matrix the size specified by x_dimension (column range), y_dimension (row range)
    """
    y_dimension, x_dimension = dimensions
    x = np.linspace(0, x_dimension - 1, x_dimension)
    y = np.linspace(0, y_dimension - 1, y_dimension)

    X, Y = np.meshgrid(x, y)  # rows determined by y and columns determined by x

    return X, Y


def warping_with_given_homography(ori_img, H, preserve, interpolation, cuda=True):
    """
    it warps the original image to a new image plane, regarding homography H. The warping (interpolation) is carried
    out in tensor manner to get along with the boosting. As most warping function aims to preserve the 'image size',
    this function can preserve the target area, meaning the target area of original image are fully preserved in the
    outcome warped image.

    * Besides, cuda switch should always be provided at least for low-level basic processing. So that the computation
    source is on command of the user. Leave this for now, if no cuda, what's the point of using pytorch?

    This function bases the Cartesian system that: each pixel H[x, y, 1]' = [tx', ty', t]', i.e. the first row of
    homography maps the x coordinates(column).

    #TODO: check the unresolved question mark '?'
    :param ori_img: the image to be warped. (in numpy format)
    :param H: homography in a 3x3 matrix (in numpy format)
    :param preserve: i) four coordinates, saying [y_start, y_end, x_start, x_end], ii) True: full frame iii) False: make
    the outcome image the same size as input image. Coordinates using the same Cartesian(?) system as input image.
    :param cuda: whether on cuda processing or not.
    :param interpolation: i) bilinear ii) cubic
    :return: warped_image, shifting parameter (?)
    """
    X, Y = mesh_xy_coordinates_of_given_2D_dimensions(ori_img.shape)
    T = np.ones(X.shape, dtype=np.float)
    cartesian_coordinates = np.stack([X, Y, T], axis=-1)  # -1 means the last axis, R x C x 3

    if cuda:
        coordinates_PT = pt.from_numpy(cartesian_coordinates).half().cuda()
        H_PT = pt.from_numpy(H).half().cuda()
    else:
        raise ValueError('CPU version is not implemented.')

    # ---------- Calculate for warpped coordinates ------------ #
    temp_coordinates_PT = coordinates_PT.view(coordinates_PT.size(0)*coordinates_PT.size(1), 3).unsqueeze(-1)
    coordinates_PT_warped = pt.bmm(H_PT.unsqueeze(0).expand(temp_coordinates_PT.size(0), *H_PT.size()),
                                   temp_coordinates_PT).view(*coordinates_PT.size())

    coordinates_PT_warped[:, :, 0] /= coordinates_PT_warped[:, :, -1]
    coordinates_PT_warped[:, :, 1] /= coordinates_PT_warped[:, :, -1]

    # ---------- Trace back into original image coordinates, with padding and shift ------------ #

