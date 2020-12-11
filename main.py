
import numpy as np
import cv2 as cv
import sys
import zipfile

from matplotlib import pyplot as plt


def get_frequencies(image, channel_value, bins=256, ):
    color_values = np.zeros(bins)

    rows = image.shape[0]
    cols = image.shape[1]
    # for arr_part in enumerate(color_values):
    for x in range(0, rows):
        for y in range(0, cols):
            # if image[x, y, channel_value] == color_values[arr_part[0]]:
            color_values[image[x, y, channel_value]] += 1

    return color_values


# this is custom implementation
def get_color_hist_rgb(image):
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    res_all = np.zeros((3, 256))
    for channel_id, c in zip(channel_ids, colors):
        # get 1 channel
        res = get_frequencies(image, channel_id)
        res_all[channel_id] = res
        plt.plot(res, color=c)

    return res_all


# this is to test
def get_color_histogram_numpy(image):
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)

    # create the histogram plot, with three lines, one for
    # each color
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(image[:, :, channel_id], bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color=c)

# 2. Write a program to measure the L 2 distance between color histograms of two images.
def distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

# 1. Write a program to extract the color histogram of each of the 2,000 images. Choose the
# parameters required with justifications. Implement your own histogram code and compare its
# results with open-source API like OpenCV and numpy.

def read_and_plot(img_name, plot=False):
    im = cv.imread(img_name)

    if (plot):
        plt.figure(1)
        plt.title('custom plot')

    histogram_vector = get_color_hist_rgb(im)

    if (plot):
        plt.figure(2)
        plt.title('numpy plot')

        # test
        get_color_histogram_numpy(im)
        plt.show()
    return histogram_vector


v1 = read_and_plot('ukbench00000.jpg', True)
v1_2 = read_and_plot('ukbench00001.jpg', True)
v2 = read_and_plot('ukbench00009.jpg')

print(distance(v1, v2))
print(distance(v1, v1_2))

archive = zipfile.ZipFile('Homework1.zip', 'r')
imgdata = archive.read('img_01.png')