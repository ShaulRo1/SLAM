import sol4_utils as s4_utils
from skimage import feature
# import sol5 as s5
import sol3 as s3
import numpy as np
import random
import Cube as Cube
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
from itertools import combinations


def pixel_to_real_world_point(row, col, depth):
    # WIDTH = 640
    # HEIGHT = 480
    # DEPTH_TO_M = 0.001
    # SCL = 1.0 / 520.0

    depthFocalLength = 525.0
    centerX = 319.5
    centerY = 239.5
    x = (col - centerX) * depth / depthFocalLength
    y = (row - centerY) * depth/ depthFocalLength
    z = depth

    # z = depth * DEPTH_TO_M
    # x = (col - WIDTH / 2) * z * SCL
    # y = (row - HEIGHT / 2) * z * SCL
    return np.array([x, y, z])


def pixels_to_world_points(pixels):
    points = []
    for pixel in pixels:
        r, c, d = pixel
        point = pixel_to_real_world_point(r, c, d)
        points += [point]
    return np.array(points)

def get_rand_pixel(im):
    h, w = im.shape
    row = random.randint(0, h - 1)
    col = random.randint(0, w - 1)
    return row, col, im[row, col]


def get_n_rand_pixels(num_pixels, im):
    pixels = []
    while num_pixels > 0:
        pixel = get_rand_pixel(im)
        if pixel not in pixels:
            pixels += [pixel]
            num_pixels -= 1
    return np.array(pixels)


def get_plane_from_points(points):
    p1, p2, p3 = points
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d


def apply_plane_on_point(point, plane):
    x, y, z = point
    a, b, c, d = plane
    return a * x + b * y + c * z + d


def if_colinear(points):
    p1, p2, p3 = points
    arr = np.array([p1, p2, p3]).reshape(3, 3)
    return np.linalg.det(arr) == 0


def ransac_plane(im, num_of_iterations, threshold):
    best_plane_agreement = 0
    best_plane = (0, 0, 0, 0)
    all_ims = []
    best_im = []
    while num_of_iterations > 0:
        print('--------------'+str(num_of_iterations))
        cur_agreement = 0
        points = pixels_to_world_points(get_n_rand_pixels(3, im))
        while if_colinear(points):
            points = pixels_to_world_points(get_n_rand_pixels(3, im))
        plane = get_plane_from_points(points)
        row = 0
        col = 0
        cur_im = np.zeros(im.shape)
        while row < 480:
            while col < 640:
                point = pixel_to_real_world_point(row, col, im[row, col])
                diff = apply_plane_on_point(point, plane)
                if abs(diff) < threshold:
                    cur_agreement += 1
                    cur_im[row, col] = 1
                col += 1
            col = 0
            row += 1
        all_ims.append(cur_im)
        if cur_agreement > best_plane_agreement:
            best_plane = plane
            best_plane_agreement = cur_agreement
            best_im = cur_im
        num_of_iterations -= 1
        print(cur_agreement)
    return best_plane, all_ims, best_im




def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


im = s4_utils.read_image("C:/Users/Shaul Ro/PycharmProjects/Tutorialed_Work/assets/table_small_1_100_depth.png", s4_utils.RGB_REP)
original_im = s4_utils.read_image("C:/Users/Shaul Ro/PycharmProjects/Tutorialed_Work/assets/table_small_1_100.png", s4_utils.RGB_REP)
edge = feature.canny(im, sigma=5)

plane, all_ims, best_im = ransac_plane(im, 9, 4)
show_images(all_ims, 3)