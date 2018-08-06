import numpy as np
import torch as pt
import time
import cv2  # TODO: opencv is usable yet far from flexible and well-maintained. try to self-produce corresponding functions.
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

engineering novelty has been long under-estimated since the traditional hardware did not revolt for long. yet we are
at a tide of era in which tensor computation is becoming very affordable and accessible, thanks to the pursue made by
deep learning community.
"""


class SemiKernelSGM(object):
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
        self.P_init = 0
        self.P_gap = 128
        self.P = np.array([4, 8, 16, 32, 64])
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

            self.img0 = cv2.resize(img0, (0, 0), fx=self.scaling, fy=self.scaling)
            self.img1 = cv2.resize(img1, (0, 0), fx=self.scaling, fy=self.scaling)
        else:
            self.img0 = self.raw_img0.copy()
            self.img1 = self.raw_img1.copy()

        self.img_dimension = self.img0.shape

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

    def set_P_sequence(self, P_init=None, P_gap=None, P=None):
        """
        it sets values of P. Once the values are reset, the temp function shall be re-initialized. Also, the
        attribute describing the number of neighbours in depth aggregation require update.
        :param P_init: the penalty parameter when disparity shift is zero
        :param P_gap: the penalty parameter when disparity is assumed to be dis-continued.
        :param P: a vector of p values regarding different level of dis-continuity.
        :return: None
        """
        if P_init is not None:
            self.P_init = P_init
        if P_gap is not None:
            self.P_gap = P_gap
        if P is not None:
            self.P = P
            self.number_of_neighbours_depth_aggregation = 2 + len(P) * 2
        print("P is updated: P_init: %.4f, P_gap: %.4f, P: %s, #P: %i" % (self.P_init, self.P_gap, str(self.P),
                                                                          self.number_of_neighbours_depth_aggregation))
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

