import numpy as np
import torch as pt
import cPickle as pickle
from src.utils import visual, convert
import time
import cv2 as cv
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
fn_data_raw_res = '../../data/raw_res'

start_time = time.time()
with open(fn_data_raw_res, 'rb') as fn:
    data_dict = pickle.load(fn)
print("Loading data cost %.4f s." % (time.time() - start_time))

[res_0_pattern, res_0_capture, res_10, res_1_pattern, res_1_capture, res_01] = data_dict

res_0_marker = res_0_capture[0, 0, :, :]
res_0_ref = res_0_capture[0, -1, :, :]

res_1_marker = res_1_capture[0, 1, :, :]
res_1_ref = res_1_capture[0, -1, :, :]

# --------  Otsu --------
visual.img_show(res_0_marker)
visual.img_show(res_0_ref)

blur_res_0_marker = cv.GaussianBlur(convert.img_double2int(res_0_marker),(7,7),0)
ret,th = cv.threshold(blur_res_0_marker,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

