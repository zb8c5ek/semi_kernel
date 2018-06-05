import numpy as np
import torch as pt
import cv2  # TODO: opencv is usable yet far from flexible and well-maintained. try to self-produce corresponding functions.


class SemiKernelSGM(object):
    def __init__(self):
        """
        Semikernel is in its version 0.0, rectification related problem is not considered. It aims to use semi-kernel
        techniques to achieve smoother disparity map and/or surface, with less GPU/TPU memory consumption, at the mean
        while rapid.
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
        self.W = np.array([])   #TODO: set the weighting array, and determine the weighting number
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

