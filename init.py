import sol4_utils as s4_utils
import cv2
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


def find_plane(points):
    p1, p2, p3 = points[0], points[1], points[2]
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d


def apply_plane_on_pixel(x, y, z, plane):
    return float(plane[0]*x + plane[1]*y + plane[2]*im[y, x] - plane[3])


def get_rand_points(im, num_of_points):
    im_shape = im.shape
    if num_of_points > im_shape[0] * im_shape[1]:
        return
    points = []
    while num_of_points > 0:
        y = random.randint(0, im_shape[0] - 1)
        x = random.randint(0, im_shape[1] - 1)
        if (x, y) not in points:
            points += [np.array([x, y, im[y, x]])]
            num_of_points -= 1
    return points


def get_sub_image(im, row, col):
    im_height, im_width = im.shape
    start_row = max(0, row - 10)
    start_col = max(0, col - 10)
    end_row = min(im_height, row + 10)
    end_col = min(im_width, col + 10)
    return im[start_row:end_row].T[start_col:end_col].T, (start_row, start_col)



def get_rand_edge_pixel(edges_im, start_pixel=(0,0)):
    edges_rows, edge_cols = np.where(edges_im == 0)
    rand_pixel = random.sample(range(0, edges_rows.shape[0]), 1)[0]
    row = edges_rows[rand_pixel] + start_pixel[0]
    col = edge_cols[rand_pixel] + start_pixel[1]
    return row, col


def pixel_to_point(im, pixel):
    WIDTH = 640
    HEIGHT = 480
    DEPTH_TO_M = 0.001
    SCL = 1.0 / 520.0
    r, c, d = pixel
    z = d * DEPTH_TO_M
    x = (c - WIDTH / 2) * z * SCL
    y = (r - HEIGHT / 2) * z * SCL
    return np.array([x, y, z])


def get_random_edge_points(im, edges_im):
    p1 = get_rand_edge_pixel(edges_im)
    pixel_neighborhood, start_pixel = get_sub_image(edges_im, p1[0], p1[1])
    point1 = pixel_to_point(im, p1)
    point2 = pixel_to_point(im, get_rand_edge_pixel(pixel_neighborhood, start_pixel))
    point3 = pixel_to_point(im, get_rand_edge_pixel(pixel_neighborhood, start_pixel))


    return np.array([point1, point2, point3])


def ransac_plane(im, num_of_iterations, threshold):
    best_plane_agreement = 0
    best_plane = (0, 0, 0, 0)
    all_ims = []
    best_im = None
    edges = feature.canny(im, sigma=3)
    while num_of_iterations > 0:
        # print(num_of_iterations)
        cur_agreement = 0
        cur_im = np.zeros(im.shape)
        points = get_random_edge_points(im, edges)
        plane = find_plane(points)
        i = 0
        j = 0
        while j < 480:
            while i < 640:
                diff = apply_plane_on_pixel(i, j, plane)
                if abs(diff) < threshold:
                    cur_agreement += 1
                    cur_im[j, i] = 1
                i += 1
            i = 0
            j += 1
        all_ims.append(cur_im)


        if cur_agreement > best_plane_agreement:
            best_plane = plane
            best_plane_agreement = cur_agreement
            best_im = cur_im
        num_of_iterations -=1
        print(cur_agreement)

    return best_plane, all_ims, best_im, edges


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


im = s4_utils.read_image("/Users/shaulr/PycharmProjects/miniSLAM/table_small/table_small_1/table_small_1_100_depth.png", s4_utils.RGB_REP)
original_im = s4_utils.read_image("/Users/shaulr/PycharmProjects/miniSLAM/table_small/table_small_1/table_small_1_100.png", s4_utils.RGB_REP)
edge = feature.canny(im, sigma=5)


thresh = 10
ims = [original_im, im, edge]
titles = ['original', 'depth', 'canny_edge_sigma_3']
# show_images(ims, 1, titles)


for _ in range(0, 8):
    print('****'+ str(_) +'******')
    plane, all_ims, b_im, edges = ransac_plane(im, 1, thresh)
    ims.append(b_im)
    # ims.append(edges)
    titles.append(str(thresh))
    thresh = thresh * 1.1