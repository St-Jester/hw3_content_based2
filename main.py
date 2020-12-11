import os

import numpy as np
import cv2 as cv
import time
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
def get_color_hist_rgb(image, plot=False):
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    res_all = np.zeros((3, 256))
    for channel_id, c in zip(channel_ids, colors):
        # get 1 channel
        res = get_frequencies(image, channel_id)
        res_all[channel_id] = res
        if plot:
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

def read_and_plot(img_name, plot=False, test=False):
    im = cv.imread(img_name)

    if plot:
        plt.figure(1)
        plt.title('custom plot')

    histogram_vector = get_color_hist_rgb(im)

    if test:
        plt.figure(2)
        plt.title('numpy plot')

        # test
        get_color_histogram_numpy(im)
        plt.show()
    return histogram_vector


# 3.
# Use 5 images shown above (ukbench00004.jpg; ukbench00040.jpg; ukbench00060.jpg;
# ukbench00588.jpg; ukbench01562.jpg) as queries. For each query image, find 10 best matches
# from the 2,000 images based on the color histogram similarity.
# Plot the query image and the 10 returned matches (use icons of reduced resolution to save
# space).
# == make matrix of distances

# print(distance(v1, v2))
# print(distance(v1, v1_2))
# read_and_plot('../Homework1/ukbench00004.jpg')
image_quieries = ['ukbench00004.jpg', 'ukbench00040.jpg', 'ukbench00060.jpg', 'ukbench00588.jpg', 'ukbench01562.jpg']


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def find_similar(query, images):
    print('find_similar')
    histogram_vector_query = get_color_hist_rgb(query)

    all_distances = np.zeros(len(images))
    for (i, image) in enumerate(images):
        histogram_vector = get_color_hist_rgb(image)
        if i % 100 == 0:
            print(f"done histogram_vector{i}")
        all_distances[i] = distance(histogram_vector_query, histogram_vector)
        if i % 100 == 0:
            print(f"done all_distances{i}")

    return all_distances


def find_n_smallest(distances, k=11):
    print(f"find_n_smallest")
    return np.argpartition(distances, k)


for im_query in image_quieries:
    start = time.time()

    im = cv.imread('../Homework1/' + im_query)
    images = load_images_from_folder('../Homework1/')
    found_distances = find_similar(im, images)

    smallest_indices = find_n_smallest(found_distances)
    print(smallest_indices[1:11])
    small_images = []
    for img_index in smallest_indices[1:11]:
        small_images.append(cv.resize(images[img_index], (100, 100)))

    numpy_vertical = np.vstack(small_images)

    end = time.time()
    print(f"time elapsed: {end - start}")
    cv.imwrite(im_query+'_results.jpg', numpy_vertical)

# # [ 5  9  7 61 60  0  1  3 11 10]
# window_name = 'image'
#
# numpy_vertical = np.vstack((images[5], images[9]))
# cv.imshow('Numpy Vertical', numpy_vertical)
# #
# # cv.imshow(window_name, images[5])
# # cv.imshow(window_name, images[9])
cv.waitKey()

