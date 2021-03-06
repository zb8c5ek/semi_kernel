import numpy as np
import torch as pt
import time
import cv2  # TODO: opencv is usable yet far from flexible and well-maintained. try to self-produce corresponding functions.
from src.utils.filtering import generate_kernel_mesh
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
"""
author notes

about evaluation:
as stereo and structured light are sensitive to camera set-up, a full pipeline shall share the same ground. also, to
establish a systematic analysis in the perspective of practice and convinience, all data are performed on a data set
with our own capture, released along with the code. * some more inspirations shall come in here ...

engineering novelty has been long under-estimated since the traditional hardware structure did not revolt for long. yet 
we are at a tide of era in which tensor computation is becoming very affordable and accessible, thanks to the pursue 
made by tensor community.
"""


class SemiKernelMatching(object):
    def __init__(self):
        """
        Semi-kernel is in its version 0.0, rectification related problem is not considered. It aims to use semi-kernel
        techniques to achieve smoother disparity map and/or surface, with less GPU/TPU memory consumption, at the mean
        while speed up as number of paths reduced.
        """
        # ---- basic parameters ----
        self.raw_img0 = None
        self.raw_img1 = None
        self.img0 = None
        self.img1 = None
        # ---- image dimensions ----
        self.raw_img_dimension = None
        self.img_dimension = None

        self.d_min = 0
        self.d_max = 63
        self.scaling = 1.0
        self.disp_num = self.d_max - self.d_min + 1
        self.census_transform_window_size = (7, 9)  # height, width

        self.census_img0 = None
        self.census_img1 = None

        self.path_num = 1  # Implementing only one path for the moment (horizontal one, later the vertical one)
        self.shift = np.array([[-1, 0], [0, -1]])   # WARNING: only path 0 and 1 are initialised.

        # self.P_init = 0
        # self.P_gap = 128  #P[0] = P_init, P[-1] = P_gap, as in journal
        self.P = np.array([0, 16, 64])
        self.number_of_neighbours_depth_aggregation = 2 + 2 * len(self.P)
        self.W = np.array([])   # TODO: set the weighting array, and determine the weighting number
        self.number_of_neighbours_space_weighting = 8
        self.costVolume_L = None
        self.costVolume_R = None

        self.temp_depth_L = None    # perform the depth aggregation of L based matching
        self.temp_space_L = None    # perform the spacial weighting parameter of L based matching
        self.temp_depth_R = None
        self.temp_space_R = None

        self.unary_cost_L = None
        self.unary_cost_R = None

        # ===== space kernel parameters =====

        self.theta_d = None
        self.theta_r = None

        # ===== padding parameters =====
        # require reset with change of theta_d and theta_r
        self.disp_pad = 3
        self.height_pad = 2
        self.width_pad = 2

    def initialize_images(self, img0, img1, scaling=None):
        """
        It initialises the input images, and assign them to class attributes accordingly, with transformation e.g. scale

        :param img0: base image
        :param img1: matching image
        :param scaling: if not None, a float between [0, 1] is expected.
        :return: None
        """
        if scaling is None:
            self.scaling = 1.0
        else:
            self.scaling = scaling

        self.raw_img0 = img0
        self.raw_img1 = img1
        self.raw_img_dimension = img0.shape

        if self.scaling != 1:
            # TODO: replace this cv2 resize function
            self.img0 = cv2.resize(img0, (0, 0), fx=self.scaling, fy=self.scaling)
            self.img1 = cv2.resize(img1, (0, 0), fx=self.scaling, fy=self.scaling)
        else:
            self.img0 = self.raw_img0.copy()
            self.img1 = self.raw_img1.copy()

        self.img_dimension = self.img0.shape

    def initialise_space_kernel_weights_using_bilateral_kernel(self, theta_d, theta_r):
        """
        it initialises the bilateral weight, according to given images. it also changes the padding parameter, which
        is needed to initialize GPU storage later. the result bilateral weight would be in the size as h x w x n
        (where n is the number of spatial neighbors in account). theta_d and theta_r are used to determine
        the kernel size, also the kernel shape. optimum parameter shall coop with application itself.

        this determines the padding parameter.
        :param theta_d: spatial based variance term, scalar (float or integer)
        :param theta_r: intensity based variance term, scalar (float or integer)
        :return: update to self.space_weight_kernel *h x w x n, update to self.ts_vs_list, update to self.ts_vs_distance
        """
        # ===== generate semi-gaussian kernel =====
        # --- determine kernel size ---
        kernel_size = np.array(2 * np.ceil(3 * [theta_d, theta_d]), dtype=np.int) + 1
        # --- generate gaussian kernel ---
        kernel_mesh = generate_kernel_mesh(kernel_size)
        gaussian_kernel = np.exp(-np.power(kernel_mesh[0] / theta_d, 2) / 2 - np.power(kernel_mesh[1] / theta_d, 2) / 2)
        # --- crop into semi-kernel and normalise ---
        ... the kernel seems to be related to the path ...

    def initialize_GPU_storage(self):
        """
        This function initialize GPU storage according to the parameter set to the class instance i.e. the parameters
        in 'self'
        :return: None
        """
        # -------- initialise cost volume: costVolume and related ----------
        self.costVolume_L = pt.cuda.HalfTensor(self.path_num, self.disp_num + 2*self.disp_pad,
                                               self.img_dimension[0] + 2*self.height_pad,
                                               self.img_dimension[1] + 2*self.width_pad).zero_()
        self.costVolume_R = pt.cuda.HalfTensor(self.path_num, self.disp_num + 2*self.disp_pad,
                                               self.img_dimension[0] + 2*self.height_pad,
                                               self.img_dimension[1] + 2*self.width_pad).zero_()

        self.temp_depth_L_horizontal = pt.cuda.HalfTensor(self.path_num,
                                                          self.number_of_neighbours_depth_aggregation,
                                                          self.disp_num + 2*self.disp_pad,
                                                          self.img_dimension[0] + 2*self.height_pad)

        self.temp_depth_R_horizontal = pt.cuda.HalfTensor(self.path_num,
                                                          self.number_of_neighbours_depth_aggregation,
                                                          self.disp_num + 2*self.disp_pad,
                                                          self.img_dimension[0] + 2*self.height_pad)

        self.temp_depth_L_vertical = pt.cuda.HalfTensor(self.path_num,
                                                        self.number_of_neighbours_depth_aggregation,
                                                        self.disp_num + 2*self.disp_pad,
                                                        self.img_dimension[1] + 2*self.width_pad)

        self.temp_depth_R_vertical = pt.cuda.HalfTensor(self.path_num,
                                                        self.number_of_neighbours_depth_aggregation,
                                                        self.disp_num + 2*self.disp_pad,
                                                        self.img_dimension[1] + 2*self.width_pad)

        self.temp_space_L_horizontal = pt.cuda.HalfTensor(self.path_num,
                                                          self.number_of_neighbours_space_weighting,
                                                          self.disp_num + 2 * self.disp_pad,
                                                          self.img_dimension[0] + 2 * self.height_pad)

        self.temp_space_R_horizontal = pt.cuda.HalfTensor(self.path_num,
                                                          self.number_of_neighbours_space_weighting,
                                                          self.disp_num + 2 * self.disp_pad,
                                                          self.img_dimension[0] + 2 * self.height_pad)

        self.temp_space_L_vertical = pt.cuda.HalfTensor(self.path_num,
                                                        self.number_of_neighbours_space_weighting,
                                                        self.disp_num + 2 * self.disp_pad,
                                                        self.img_dimension[1] + 2 * self.width_pad)

        self.temp_space_R_vertical = pt.cuda.HalfTensor(self.path_num,
                                                        self.number_of_neighbours_space_weighting,
                                                        self.disp_num + 2 * self.disp_pad,
                                                        self.img_dimension[1] + 2 * self.width_pad)

        self.unary_cost_L = pt.cuda.HalfTensor(self.disp_num + 2 * self.disp_pad,
                                               self.img_dimension[0] + 2 * self.height_pad,
                                               self.img_dimension[1] + 2 * self.width_pad).zero_() + 1023

        self.unary_cost_R = pt.cuda.HalfTensor(self.disp_num + 2 * self.disp_pad,
                                               self.img_dimension[0] + 2 * self.height_pad,
                                               self.img_dimension[1] + 2 * self.width_pad).zero_() + 1023

    def set_disp_range(self, d_min, d_max):
        """
        disparity range shall be set through this function, and this function ALONE, as d_min, d_max and d_num shall
        always update simultaneously.
        :param d_min: minimum disparity
        :param d_max: maximum disparity
        :return: None
        """
        self.d_min = d_min
        self.d_max = d_max
        self.disp_num = self.d_max - self.d_min + 1
        print("Disparity parameters are re-set:  d_min: %i, d_max: %i, disp_num: %i" % (self.d_min, self.d_max, self.disp_num))
        # TODO: reset the d related tensors

    def set_P_sequence(self, P=None):
        """
        it sets values of P. Once the values are reset, the temp function shall be re-initialized. Also, the
        attribute describing the number of neighbours in depth aggregation require update.
        # :param P_init: the penalty parameter when disparity shift is zero
        # :param P_gap: the penalty parameter when disparity is assumed to be dis-continued.
        :param P: a vector of p values regarding different level of dis-continuity. P[0] = P_init, P[-1] = P_gap, hence
        the P_init and P_gap is canceled.
        :return: None
        """
        # if P_init is not None:
        #     self.P_init = P_init
        # if P_gap is not None:
        #     self.P_gap = P_gap
        if P is not None:
            self.P = P
            self.number_of_neighbours_depth_aggregation = len(P) * 2 - 2
        print("P is updated: P: %s, #P: %i" % (str(self.P), self.number_of_neighbours_depth_aggregation))
        # TODO: reset the related tensors if needed.

    def compute_unary_cost(self, census_window_size=(7, 9), r=None, R=None):
        """
        unary cost; census transformed image
        it computes unary cost (based on census transform) on both images. Aside the unary cost, it also provides the
        census transformed images.
        :param census_window_size: the window of census transformation, if not given. Taking original values in
        self.census_transform_window_size
        :param r: the lower bound of disparity, if None then set to self.d_min
        :param R: the higher bound of disparity, if None then set to self.d_max
        :return: None
        """
        pass

    def horizontal_path_swiping(self):
        pass
        # TODO: it aims to perform the global single path search along horizontal direction

    def calculate_pairwise_cost_LR(self, census_h, census_w, r=None, R=None):
        # TODO imported, need to be adapted
        """
        # Modification Done, testing remains.
        As padding parameter is added. The true disparity should be calculated regarding both the padding and the shift
        from r to R.
        :return: self.unit_cost would be updated accordingly. Together with self.unit_costLR, which is the M->B mirror.
        """
        last_time = time.time()
        census_img0 = self.census_transform(census_h, census_w, self.matching_img0)
        census_img1 = self.census_transform(census_h, census_w, self.matching_img1)

        if r is None:
            r = self.d_min
        if R is None:
            R = self.d_max

        for d in np.arange(r, R + 1):

            i = d - r + self.disp_pad

            if d < 0:
                # print(self.census_img0[:, :, -d:].size())
                # print(self.census_img1[:, :, :d].size())
                self.unit_cost_L[i, self.height_pad:-self.height_pad,
                -d + self.width_pad:-self.width_pad] = torch.sum(
                    census_img0[:, :, -d:] != census_img1[:, :, :d], 0
                )
            elif d > 0:
                # print(self.census_img0[:, :, :-d].size())
                # print(self.census_img1[:, :, d:].size())
                self.unit_cost_L[i, self.height_pad:-self.height_pad,
                self.width_pad:-(d + self.width_pad)] = torch.sum(
                    census_img0[:, :, :-d] != census_img1[:, :, d:], 0
                )
            else:
                self.unit_cost_L[i, self.height_pad:-self.height_pad,
                self.width_pad:-self.width_pad] = torch.sum(
                    census_img0[:, :, :] != census_img1[:, :, :], 0
                )

            d_LR = -d
            # -R : r
            i_LR = d_LR + R + self.disp_pad

            if d_LR < 0:
                self.unit_cost_R[i_LR, self.height_pad:-self.height_pad,
                -d_LR + self.width_pad:-self.width_pad] = torch.sum(
                    census_img1[:, :, -d_LR:] != census_img0[:, :, :d_LR], 0
                )
            elif d_LR > 0:
                self.unit_cost_R[i_LR, self.height_pad:-self.height_pad,
                self.width_pad:-(d_LR + self.width_pad)] = torch.sum(
                    census_img1[:, :, :-d_LR] != census_img0[:, :, d_LR:], 0
                )
            else:
                self.unit_cost_R[i_LR, self.height_pad:-self.height_pad,
                self.width_pad:-self.width_pad] = torch.sum(
                    census_img1[:, :, :] != census_img0[:, :, :], 0
                )

        print(
                'Census transformation and Unary-Cost Volume Preparation accomplished in: %.4f' % (
                time.time() - last_time))
        self.census_img0 = census_img0
        self.census_img1 = census_img1

    def census_transform(self, window_h, window_w, img, cuda=True):
        # TODO: imported, need to be adapted
        """
        This function perform census transform. As padding parameter is determined by window_h and window_w, henceforth
        the function is self-contained / standable.
        :param window_w:
        :param window_h:
        :param img0:
        :param img1:
        :return:
        """
        # Images are transformed into tensor, so that the manipulation is in PT way.

        w_pad_0 = np.int(np.floor(window_w / 2))
        w_pad_1 = np.int(np.floor(window_w / 2 - (1 - (window_w % 2))))

        h_pad_0 = np.int(np.floor(window_h / 2))
        h_pad_1 = np.int(np.floor(window_h / 2 - (1 - (window_h % 2))))

        pad_img = np.array(np.pad(img, ((h_pad_0, h_pad_1), (w_pad_0, w_pad_1)), 'constant', constant_values=(0, 0)),
                           dtype=np.float32)

        if cuda:
            pad_img_pt = pt.from_numpy(pad_img).cuda()
            census_img = pt.cuda.ByteTensor(window_w * window_h, img.shape[0], img.shape[1]).zero_()
        else:
            pad_img_pt = pt.from_numpy(pad_img)
            census_img = pt.ByteTensor(window_w * window_h, img.shape[0], img.shape[1]).zero_()
        if cuda:
            census_img = census_img.cuda()

        counter_channel = 0

        for w in np.arange(-w_pad_0, w_pad_1 + 1, dtype=np.int32):
            for h in np.arange(-h_pad_0, h_pad_1 + 1, dtype=np.int32):
                census_img[counter_channel, :, :] = (pad_img_pt[h_pad_0:-h_pad_1, w_pad_0:-w_pad_1] -
                                                     pad_img_pt[h_pad_0 + h:h_pad_0 + h + img.shape[0],
                                                     w_pad_0 + w:w_pad_0 + w + img.shape[1]]) >= 0
                counter_channel += 1

        return census_img

    def get_simplified_pad_parameter(self):
        """
        return dp, hp, wp from self.disp_pad / height_pad / width_pad
        :return: dp, hp, wp
        """
        dp = self.disp_pad
        hp = self.height_pad
        wp = self.width_pad
        return dp, hp, wp

    def L_tensor_extractor_into_corresponding_temp_tensor(self, ij, r, h_shift=0, w_shift=0):
        """
        this function extracts the corresponding tensor from self.L, given ij and r(path_num), x_shift indicates the
        tensor shift in these three dimensions. the one indicated with r direction (encoded as in journal), says the
        selection of a single row / column indexed by ij, the rest two dimensions that the shift is volume shift (v2v).
        here r and corresponding _shift indicates overlap information, therefore a double-check could be proposed to
        avoid possible mistake, i.e. the index shifted corresponding to r shall be consistent with _shift.

        one shall bear in mind, tensor manipulation extracts tensor address, therefore manipulation on extracted tensor
        affects original tensor from whom it is extracted.
        :param ij: index of target column / row
        :param r: path number
        :param h_shift: height shift (first entry)  t': hp:-hp = t: hp+hs:-hp+hs
        :param w_shift: width shift (first entry)   t': wp:-wp = t: wp+ws:-wp+ws
        :return out_tensor: extracted tensor as self.temp_*_tensor size [1, 1, dp:-dp, wp/hp:-wp/hp]
        """
        dp, hp, wp = self.get_simplified_pad_parameter()
        self.temp_depth_L_vertical[r, :, :, :].fill_(255)   #TODO: double check whehter this value is big enough
        # ------ double check based on r and _shift ------
        if r == 0:
            if h_shift != -1:
                raise ValueError("Path number r and height shift did not match !r: %i; h_shift: % i" % (r, h_shift))

            # ------ process path #0 ------
            # === Case 1 ===
            self.temp_depth_L_vertical[r, 0, dp:-dp, wp:-wp] = \
                self.costVolume_L[r, dp:-dp, ij+h_shift, wp+w_shift:-wp+w_shift] + self.P[0]
            # === Case 2 ===
            for p in np.arange(1, len(self.P)-1):
                # WARNING: the disparity might always increase instead of decrease ... double check with SGM paper
                self.temp_depth_L_vertical[r, 2*p-1, dp:-dp, wp:-wp] = \
                    self.costVolume_L[r, dp+p:-dp+p, ij+h_shift, wp+w_shift:-wp+w_shift] + self.P[p]
                self.temp_depth_L_vertical[r, 2*p, dp:-dp, wp:-wp] = \
                    self.costVolume_L[r, dp-p, -dp-p, ij+h_shift, wp+w_shift:-wp+w_shift] + self.P[p]
            # === Case 3 ===
            self.temp_depth_L_vertical[r, -1, dp:-dp, wp:-wp] = \
                pt.min(self.costVolume_L[r, dp:-dp,
                       ij+h_shift,
                       wp+w_shift:-wp+w_shift],
                       dim=0).repeat(1, self.disp_num+2*dp, 1, 1) + self.P[-1]

            out_tensor = pt.min(self.temp_depth_L_vertical[r, :, dp:-dp, wp:-wp], dim=0)
            return out_tensor
        else:
            raise ValueError("Path num: %i is NOT supported yet !!!" % r)

    def depth_slope_cost_calculation(self, ij, r, h_shift, w_shift):
        """
        calculate the depth related cost term in semi-kernel SGM formula. besides the input above, self.L, self.P are
        inherited through class.
        :param ij: ij parameter, indicating which row / column is being processed.
        :param r: path number: 0 for vertical 6 etc. (cf. journal)
        :param h_shift: height shift, depends on r, to simplify computation, shall be passed from higher function
        :param w_shift: width shift, depends on r, to simplify computation, shall be passed from higher function
        :return: update_tensor for depth term, in order to update in global_dynamic_programming
        """
        update_tensor = self.L_tensor_extractor_into_corresponding_temp_tensor(ij=ij, r=r,
                                                                               h_shift=h_shift,
                                                                               w_shift=w_shift)
        return update_tensor

    def space_semi_kernel_cost_calculation(self, ij, r):
        pass

    def global_dynamic_programming(self):
        """
        it performs dynamic programming, based on all the paths. the function loops over the image to acquire values in
        the full cost volume. as paths shall be adjustable, each path shall be a call-able block and could be integrated
        with ease e.g. the parameter passing shall be self-contained.
        :return: None, afterwards the full cost volume shall be ready, and minimum disparity be able to extract.
        """
        # ----- vertical path processing -----
        for ij in np.arange(self.height_pad, self.height_pad + self.img_dimension[0]):
            # --- path 0 ---

            depth_tensor_update_now = self.depth_slope_cost_calculation(ij=ij, r=0,
                                                                        h_shift=self.shift[0, 0],
                                                                        w_shift=self.shift[0, 1]
                                                                        )
