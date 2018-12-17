import numpy as np
from matplotlib import pyplot as plt
import cv2


def side_by_side_plot(image1, image2, title1='Image 1', title2='Image 2'):
    figure, axarr = plt.subplots(1,2)
    axarr[0].imshow(image1)
    axarr[1].imshow(image2)
    plt.show()


def showImage(image, title='Photo'):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def showGrayImage(image, title='Photo'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def showChannels(image):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for c, ax in zip(range(3), axs):
        tmp_img = np.zeros(image.shape, dtype="uint8")
        tmp_img[:, :, c] = image[:, :, c]
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)
        ax.imshow(tmp_img, cmap='gray')
        ax.set_axis_off()
    plt.show()
