import torch as pt
import numpy as np

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


def otsu_thresholding(img, bins=20):
    """"""

    img_min = pt.min(img)   # without dimension, 'min' function searches for the global minimum. maximum the same
    img_max = pt.max(img)

    b = pt.histc(input=img, bins=bins, min=img_min, max=img_max, out=None)
    b_value = pt.linspace(img_min, img_max, step=bins)
    p = b / pt.sum(b)

    omega_0 = pt.FloatTensor(bins).fill_(0)
    omega_1 = pt.FloatTensor(bins).fill_(0)

    miu_0 = pt.FloatTensor(bins).fill_(0)
    miu_1 = pt.FloatTensor(bins).fill_(0)

    # -------------------- #

    omega_0[0] = p[0]
    omega_1[0] = pt.sum(p) - p[0]

    miu_0[0] = b_value[0] * p[0]    # miu value is without divide over omega
    miu_1[0] = pt.sum(b_value[1:]*p[1:])

    for t in np.arange(1, bins):

        omega_0[t] = omega_0[t-1] + p[t]
        omega_1[t] = omega_1[t-1] - p[t]

        miu_0[t] = miu_0[t-1] + b_value[t] * p[t]
        miu_1[t] = miu_1[t-1] - b_value[t] * p[t]

    miu_0 = miu_0 / omega_0
    miu_1 = miu_1 / omega_1

    theta_b = (miu_0 - miu_1).pow(2) * omega_0 * omega_1

    thres = b_value[pt.argmax(theta_b)]

    return thres




