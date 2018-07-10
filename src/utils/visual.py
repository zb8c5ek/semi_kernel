from matplotlib import pyplot as plt
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


def img_show(img, colorbar=True, gray=False):
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    if colorbar:
        plt.colorbar()
    plt.show()

