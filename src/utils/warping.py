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


class warpingImage(object):
    """
    this class aims to preserve / represent as much information on warping plane as possible.
    it saves image as i) image plane and ii) coordinates, to tackle the warping coordinates change problem.
    the coordinates are determined in the warping image plane from previous image. and the coordinates represents
    where the corresponding image is cropped.
    coordinates are in the format of [min_x, min_y, max_x, max_y]
    """
    def __init__(self, origin_size=None, full_img_coor=None):
        self.full_img = None     # preserve the full original image in warped image plane
        self.full_img_coor = None    # in int
        self.direct_img = None   # the image with the same center and same size as original image
        self.direct_img_coor = None  # in int
        self.roi_img = None      # the image with ROI region preserved
        self.roi_img_coor = None     # in int
        self.origin_size = None

        if full_img_coor is not None:
            self.full_img_coor = full_img_coor
        if origin_size is not None:
            self.origin_size = origin_size

    def set_direct_img_coors(self, direct_img_coor=None):
        """
        it calculates the size of direct img, to have the exact size of self.origin_size and share the same center with
        self.full_img
        pre-requirement: self.full_img_coor and self.origin_size must be assigned already.
        :return:
        """
        if direct_img_coor is not None:
            self.direct_img_coor = direct_img_coor
        else:

            center_x = np.int(np.floor(0.5 * (self.full_img_coor[0] + self.full_img_coor[2])))
            center_y = np.int(np.floor(0.5 * (self.full_img_coor[1] + self.full_img_coor[3])))

            if self.origin_size[0] % 2 == 1:
                self.direct_img_coor = np.array([center_x - np.int(np.floor(self.origin_size[0] / 2)),
                                                 center_y - np.int(np.floor(self.origin_size[1] / 2)),
                                                 center_x + np.int(np.floor(self.origin_size[0] / 2)),
                                                 center_y + np.int(np.floor(self.origin_size[1] / 2))])
            else:
                self.direct_img_coor = np.array([center_x - np.int(np.floor(self.origin_size[0] / 2)),
                                                 center_y - np.int(np.floor(self.origin_size[1] / 2)),
                                                 center_x + np.int(np.floor(self.origin_size[0] / 2)) - 1,
                                                 center_y + np.int(np.floor(self.origin_size[1] / 2)) - 1])



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


def bilinear_warping_given_ori_img_coor_inv(ori_img, coor_inv, nan_val=255, cuda=True):
    """
    it finishes the warping by interpolates values using coordinates contained in coor_inv from the original image
    ori_img. interpolation is implemented on the formula of bilinear.
    :param ori_img: original image, in gray scale assumed
    :param coor_inv: inverse coordinates from target image, in format [h, w, 2] where [:,:,0] is x and [:,:,1] is y
    :param nan_val: the value for unvalid coordinates
    :param cuda: whether perform on cuda
    :return: interpolated / warped image with the same size as coor_inv
    """
    valid_x_min = 0  # the values must larger or equal to this one to have valid interpolation
    valid_y_min = 0
    valid_x_max = ori_img.shape[1]-1    # the value must be smaller (not even equal) to have valid interpolation
    valid_y_max = ori_img.shape[0]-1    # the edge does not matter anyway (should be -2, yet range function decrease 1)

    if valid_y_min < int(pt.min(coor_inv[:, :, 1])):
        valid_y_min = int(pt.min(coor_inv[:, :, 1]))
    if valid_y_max > int(pt.max(coor_inv[:, :, 1]))+1:  # include the row after the floor operation
        valid_y_max = int(pt.max(coor_inv[:, :, 1]))+1

    if cuda:
        temp_img_inv = pt.HalfTensor(coor_inv.shape[0], coor_inv.shape[1], 4).cuda().fill_(float('nan'))
    else:
        temp_img_inv = pt.HalfTensor(coor_inv.shape[0], coor_inv.shape[1], 4).fill_(float('nan'))

    temp_zero_index = 0 * coor_inv[:, :, 0].unsqueeze(-1)
    previous = pass

    for y_now in np.arange(valid_y_min, valid_y_max):
        selected_y = (coor_inv[:, :, 1] >= y_now) * (coor_inv[:, :, 1] < y_now+1)


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

    :param ori_img: the image to be warped. (in numpy format)
    :param H: homography in a 3x3 matrix (in numpy format)
    :param preserve: i) four coordinates, saying [y_start, y_end, x_start, x_end], ii) True: full frame iii) False: make
    the outcome image the same size as input image. Coordinates using the same Cartesian system as input image.
    :param cuda: whether on cuda processing or not.
    :param interpolation: i) bilinear ii) cubic
    :return: warped_image, shifting parameter

    """
    X, Y = mesh_xy_coordinates_of_given_2D_dimensions(ori_img.shape)
    T = np.ones(X.shape, dtype=np.float)
    cartesian_coordinates = np.stack([X, Y, T], axis=-1)  # -1 means the last axis, R x C x 3
    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[-1, -1]
    if cuda:
        coordinates_PT = pt.from_numpy(cartesian_coordinates).half().cuda()
        H_PT = pt.from_numpy(H).half().cuda()
    else:
        coordinates_PT = pt.from_numpy(cartesian_coordinates).half()
        H_PT = pt.from_numpy(H).half()

    # ---------- Calculate for warpped coordinates ------------ #
    temp_coordinates_PT = coordinates_PT.view(coordinates_PT.size(0)*coordinates_PT.size(1), 3).unsqueeze(-1)
    coordinates_PT_warped = pt.bmm(H_PT.unsqueeze(0).expand(temp_coordinates_PT.size(0), *H_PT.size()),
                                   temp_coordinates_PT).view(*coordinates_PT.size())

    coordinates_PT_warped[:, :, 0] /= coordinates_PT_warped[:, :, -1]
    coordinates_PT_warped[:, :, 1] /= coordinates_PT_warped[:, :, -1]

    # ---------- Determine coordinates range in the target image ------------ #
    if cuda:
        min_x = np.floor(np.min(coordinates_PT_warped[:, :, 0].cpu().numpy()))
        min_y = np.floor(np.min(coordinates_PT_warped[:, :, 1].cpu().numpy()))
        max_x = np.ceil(np.max(coordinates_PT_warped[:, :, 0].cpu().numpy()))
        max_y = np.ceil(np.max(coordinates_PT_warped[:, :, 1].cpu().numpy()))
    else:
        min_x = np.min(coordinates_PT_warped[:, :, 0].numpy())
        min_y = np.min(coordinates_PT_warped[:, :, 1].numpy())
        max_x = np.max(coordinates_PT_warped[:, :, 0].numpy())
        max_y = np.max(coordinates_PT_warped[:, :, 1].numpy())

    # so far, the target region is decided i.e. where coordinates would be in the target frame. Now shall trace back.

    warped_img = warpingImage(origin_size=np.array([ori_img.shape[1], ori_img.shape[0]]),
                              full_img_coor=np.array([min_x, min_y, max_x, max_y]))
    warped_img.set_direct_img_coors(direct_img_coor=None)
    # warped_img.full_img_coor = np.array([min_x, min_y, max_x, max_y])
    # warped_img.origin_size = np.array([ori_img.shape[1], ori_img.shape[0]])

    # --------- mesh grids for tracing back the coordinates ------------ #
    X_inv, Y_inv = mesh_xy_coordinates_of_given_2D_dimensions([max_y - min_y + 1, max_x - min_x + 1])
    X_inv += min_x
    Y_inv += min_y

    T_inv = np.ones(X_inv.shape, dtype=np.float)
    cartesian_coordinates_inv = np.stack([X_inv, Y_inv, T_inv], axis=-1)  # -1 means the last axis, R x C x 3
    # --------- cast into tensor ---------- #
    if cuda:
        coordinates_inv_PT = pt.from_numpy(cartesian_coordinates_inv).half().cuda()
        H_inv_PT = pt.from_numpy(H_PT).half().cuda()
    else:
        coordinates_inv_PT = pt.from_numpy(cartesian_coordinates_inv).half()
        H_inv_PT = pt.from_numpy(H_PT).half()

    # ---------- Calculate for warpped coordinates ------------ #
    temp_coordinates_inv_PT = coordinates_inv_PT.view(
        coordinates_inv_PT.size(0)*coordinates_inv_PT.size(1), 3).unsqueeze(-1)
    coordinates_inv_PT_warped = pt.bmm(H_inv_PT.unsqueeze(0).expand(temp_coordinates_inv_PT.size(0), *H_inv_PT.size()),
                                   temp_coordinates_inv_PT).view(*coordinates_inv_PT.size())

    coordinates_inv_PT_warped[:, :, 0] /= coordinates_inv_PT_warped[:, :, -1]
    coordinates_inv_PT_warped[:, :, 1] /= coordinates_inv_PT_warped[:, :, -1]


