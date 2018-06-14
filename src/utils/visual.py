from matplotlib import pyplot as plt

def img_show(img,colorbar=True, gray=False):
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    if colorbar:
        plt.colorbar()
    plt.show()