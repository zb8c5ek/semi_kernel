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


def generate_shifted_image_stack(ori_img, kernel_mesh=None, shift_list=False, cuda=True):
    """
    it generates the shifted image stack, to describe neighbors at a vast parallel manner i.e. tensor. original
    image is shifted and stacked according to the kernel_mesh.
    :param ori_img: original image as the source of the stack.
    :param kernel_mesh: (mesh_x, mesh_y), which has identical size of the spatial kernel to be applied. gaussian kernel
    for instance, or theta_d kernel for bilateral filter.
    :param shift_list: whether to return the list of shifts. if True, y(row)_shift_list and x(column)_shift_list is
    returned for further processing e.g. weight calculation for bilateral kernel.
    :param cuda: True. non-cuda version is not implemented.
    :return:
    """
    if cuda is not True:
        raise ValueError("CUDA is required !")
    kernel_size = kernel_mesh[0].shape
    if (kernel_size[0] % 2 == 0) or (kernel_size[1] % 2 == 0):
        raise ValueError("Kernel size must be ODD in all dimensions, double check !kernel_mesh!.")
    # ----- prepare parameters -----
    num_kernel_instance = kernel_size[0]*kernel_size[1]
    center_shift = int(kernel_size / 2)
    hp = wp = center_shift

    hs_matrix = kernel_mesh[1]
    ws_matrix = kernel_mesh[0]

    ys_flatten = hs_matrix.flatten(order='C')
    xs_flatten = ws_matrix.flatten(order='C')
    # ----- move original image to cuda and get padded -----
    padded_ori_img_cuda = pt.from_numpy(np.pad(ori_img,
                                               pad_width=((hp, hp), (wp, wp)),
                                               mode='constant',
                                               constant_values=((0, 0), (0, 0)))).half().cuda()
    # ----- initialize storage of shift_image_stack -----
    shift_img_stack = pt.cuda.HalfTensor(ori_img.shape[0],
                                         ori_img.shape[1],
                                         num_kernel_instance).fill_(0)

    # ----- generate the shift_img_stack -----
    for i in np.arange(num_kernel_instance):
        shift_img_stack[:, :, i] = padded_ori_img_cuda[hp+ys_flatten[i]:-hp+ys_flatten[i],
                                   wp+xs_flatten[i]:-wp+xs_flatten[i]]
    if shift_list:
        return shift_img_stack, ys_flatten, xs_flatten
    else:
        return shift_img_stack


def coop_square_kernel_with_shifted_image_stacks(ori_img, kernel, kernel_mesh=None,
                                                 kernel_norm=False,
                                                 cuda=True):
    """
    it is a sub-function for function filtering.gaussian_filter. this function applied the kernel weights, and using
    i) bmm to multiply the weights for shifted image stacks (instead of searching for neighbor or convolution).
    :param ori_img: the image to be filtered
    :param kernel: kernel to apply on the image. kernel size must be odd in both dimensions.
    :param kernel_mesh: the mesh coordinates in (x,y) of the kernel, if not given then generate here.
    :param kernel_norm: if True, the kernel is normalised so that the weight is exactly sum to 1, apply to location
    invariant kernels e.g. gaussian kernel.
    # :param kernel_space_norm: if True, the kernel at each entry is normalized so that the sum would be 1. this
    # parameter is different from 'kernel_norm', as for kernels which are location dependent e.g. bilateral filter,
    # the normalisation shall be performed differently at each last dimension instance.
    :param cuda: must be True
    :return: filtered image in its original format
    """
    if cuda is not True:
        raise ValueError("CUDA is required !")
    if (kernel.shape[0] % 2 == 0) or (kernel.shape[1] % 2 == 0):
        raise ValueError("Kernel size must be ODD in all dimensions.")
    # ------ process shifted image s stack ------
    # space_shift_list_primary =np.zeros([num_kernel_instance, 2])
    if kernel_mesh is None:
        kernel_mesh = generate_kernel_mesh(kernel.shape)
    shifted_img_stack = generate_shifted_image_stack(ori_img=ori_img,
                                                     kernel_mesh=kernel_mesh,
                                                     shift_list=False, cuda=cuda)
    # for (hs, ws) in zip(ys_flatten, xs_flatten):  --> the grammar to iterate two variables simultaneously.
    # ----- adjust size and apply matrix multiplication to get the product -----
    #   WARNING: the size of tensors shall be handled with extreme caution !!!
    kernel_flatten = kernel.flatten(order='C')
    if kernel_norm:
        kernel_flatten = kernel_flatten / np.sum(kernel_flatten)
    kernel_flatten_cuda = pt.from_numpy(kernel_flatten).half().cuda()
    filtered_img_stack = pt.matmul(shifted_img_stack, kernel_flatten_cuda)
    filtered_img_cuda = pt.sum(filtered_img_stack, 2)
    raw_filtered_img = filtered_img_cuda.cpu().float().numpy()

    # ----- return as the same data type as the input image e.g. float or int -----
    filtered_img = raw_filtered_img.astype(ori_img.dtype)

    return filtered_img


def generate_kernel_mesh(kernel_size):
    """
    it generates the kernel mesh of given kernel size. the kernel_size must be ODD number in all dimensions.
    :param kernel_size: np.array, a kernel_size describe a kernel in [#rows, #columns] manner
    :return: kernel mesh in format [mesh_x, mesh_y]
    """
    if (kernel_size[0] % 2 == 0) or (kernel_size[1] % 2 == 0):
        raise ValueError("Kernel size must be ODD in all dimensions.")
    center_shift = np.floor(kernel_size / 2)
    kernel_mesh = mesh_xy_coordinates_of_given_2D_dimensions(kernel_size)
    kernel_mesh[0] -= center_shift[0]
    kernel_mesh[1] -= center_shift[1]
    return kernel_mesh


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
    kernel_size = np.array(2*np.ceil(3*[theta, theta]), dtype=np.int) + 1
    # kernel_size in (y_size, x_size) format
    # ----- generate gaussian kernel -----
    kernel_mesh = generate_kernel_mesh(kernel_size)

    gaussian_kernel = np.exp(-np.power(kernel_mesh[0]/theta, 2)/2 - np.power(kernel_mesh[1]/theta, 2)/2)
    # ----- generate corresponding [hs, vs] list -----
    if cuda:
        filtered_img = coop_square_kernel_with_shifted_image_stacks(ori_img=ori_img,
                                                                    kernel=gaussian_kernel,
                                                                    kernel_mesh=kernel_mesh,
                                                                    cuda=True)
    else:
        raise ValueError("CUDA-free version is not supported.")

    if return_kernel:
        return filtered_img, gaussian_kernel
    else:
        return filtered_img


def bilateral_filtering_using_shifted_image_stacks(ori_img, theta_d, theta_r, cuda=True):
    """
    bilateral filtering is implemented in shifted image stacks manner (shifted tensor). different from filtering using
    a fixed square kernels, bilateral filter has different weights at each pixel. theta_d, which is a spatial gaussian
    kernel, is used to determine the size of the bilateral kernel, while theta_r describes the weights distribution
    within intensity channel.
    :param ori_img: the image to be filtered, in np.array format, both int and float are supported.
    :param theta_d: variance of spatial gaussian kernel c.f. journal for concrete definition
    :param theta_r: variance of intensity gaussian kernel
    :param cuda: True, none-cuda version is not implemented.
    :return: filtered image, in np.array format and the same data type (dtype)
    """
    # ----- determine kernel size -----
    kernel_size = np.array(2*np.ceil(3*[theta_d, theta_d]), dtype=np.int) + 1
    # ----- generate shifted image stack -----
    kernel_mesh = generate_kernel_mesh(kernel_size)
    shifted_img_stack, ys_list, xs_list = generate_shifted_image_stack(ori_img=ori_img,
                                                                       kernel_mesh=kernel_mesh,
                                                                       shift_list=True,
                                                                       cuda=cuda)
    # ----- calculate kernel tensor -----
    !!! use the double loop, i and end - i, and the middle one is the 0 one
    for i in np.arange(len(ys_list)):
        exp_term_1 = ys_list[i] * ys_list[i] + xs_list[i] * xs_list[i] / (2 * theta_d *theta_d)
        exp_term_2 =


def median_filtering():
    pass


def sobel_filtering():
    pass