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


def coop_square_kernel_with_shifted_image_stacks(ori_img, kernel, kernel_mesh=None, cuda=True):
    """
    it is a sub-function for function filtering.gaussian_filter. this function applied the kernel weights, and using
    i) bmm to multiply the weights for shifted image stacks (instead of searching for neighbor or convolution).
    :param ori_img: the image to be filtered
    :param kernel: kernel to apply on the image. kernel size must be odd in both dimensions.
    :param kernel_mesh: the mesh coordinates in (x,y) of the kernel, if not given then generate here.
    :param cuda: must be True
    :return: filtered image in its original format
    """
    if cuda is not True:
        raise ValueError("CUDA is required !")
    if (kernel.shape[0]%2==0) or (kernel.shape[1]%2==0):
        raise ValueError("Kernel size must be ODD in all dimensions.")
    kernel_size = kernel.shape
    num_kernel_instance = kernel_size[0]*kernel_size[1]
    center_shift = int(kernel_size / 2)
    hp = wp = center_shift
    # ------ process shifted image s stack ------
    # space_shift_list_primary =np.zeros([num_kernel_instance, 2])
    if kernel_mesh is None:
        kernel_mesh = generate_kernel_mesh(kernel_size)
    hs_matrix = kernel_mesh[1]
    ws_matrix = kernel_mesh[0]

    kernel_flatten = kernel.flatten(order='C')
    ys_flatten = hs_matrix.flatten(order='C')
    xs_flatten = ws_matrix.flatten(order='C')

    padded_ori_img_cuda = pt.from_numpy(np.pad(ori_img,
                                               pad_width=((hp, hp), (wp, wp)),
                                               mode='constant',
                                               constant_values=((0,0),(0,0)))).half().cuda()

    shift_img_stack = pt.cuda.HalfTensor(1,
                                         kernel_size[0]*kernel_size[1],
                                         ori_img.shape[0]+2*hp,
                                         ori_img.shape[1]+2*wp).fill_(0)

    for i in np.arange(num_kernel_instance):
        shift_img_stack[0, i, hp:-hp, wp:-wp] = padded_ori_img_cuda[hp+ys_flatten[i]:-hp+ys_flatten[i],
                                                wp+xs_flatten[i]:-wp+xs_flatten[i]]
    # for (hs, ws) in zip(ys_flatten, xs_flatten):  --> the grammar to iterate two variables simultaneously.
    # ----- adjust size and apply matrix multiplication to get the product -----
    #   WARNING: the size of tensors shall be handled with extreme caution !!!
    kernel_flatten_cuda = pt.from_numpy(kernel_flatten).half().cuda()
    filtered_img_stack = pt.matmul(shift_img_stack, kernel_flatten_cuda)
    filtered_img_cuda = pt.sum(filtered_img_stack[0, :, :, :], 0)
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
    if (kernel_size[0]%2==0) or (kernel_size[1]%2==0):
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
        # kernel_size in (x_size, y_size) format
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


def bilaterial_filtering():
    pass


def median_filtering():
    pass


def sobel_filtering():
    pass