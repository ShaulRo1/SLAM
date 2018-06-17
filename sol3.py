import numpy as np
import scipy.signal as signal
import scipy.ndimage
from scipy.misc import imread as imread
import skimage.color as skimage
import matplotlib.pyplot as plt
import os
import random
# --------------------------------------------Constants---------------------------------------------
# ORIGINAL_IMAGE_COEFF = 1
# # IM_1_EXAMPLE_1 = 'externals/mouth.jpg'
# IM_2_EXAMPLE_1 = 'externals/ostrich.jpg'
# MASK_EXAMPLE_1 = 'externals/mask_mouth.jpg'
# IM_1_EXAMPLE_2 = 'externals/supernova.jpg'
# IM_2_EXAMPLE_2 = 'externals/brain_scan.jpg'
# MASK_EXAMPLE_2 = 'externals/mask_brain.jpg'
# MASK_THRESHOLD = 0.0005
# TITLES = ['Image_1', 'Image_2', 'Mask', 'Blended_image']
# ----------------------------------------Lambda Functions------------------------------------------
get_bound_on_image_size = lambda shape: np.ceil(np.log2(min(shape))) - 4

get_num_cols_of_image = lambda im: im.shape[1]

# ----------------------------------------Helper Functions------------------------------------------
"""
Returns True iff the image is colored
"""
is_image_colored = lambda image: len(image.shape) > 2

CONVERSION_MTRX = np.matrix([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
YIQ_REP = 1
RGB_REP = 2
MAX_INTENSITY = 255
NUM_OF_BINS = 256


def read_image(filename, representation):
    """
	This function turns an image to a different representation.
	image (1) or an RGB image (2).
	:param filename: string containing the image filename to read.
	:param representation: representation code, either 1 or 2 defining whether the output should be a grayscale.
	:return: a different representation of the image.
	"""
    im = imread(filename)
    is_colored = is_image_colored(im)
    if is_colored:
        if representation == YIQ_REP:
            im = np.divide(skimage.rgb2gray(im), float(MAX_INTENSITY)).astype(np.float64)
            return im
        else:
            im = imread(filename, mode='RGB')
    im = np.divide(im, float(MAX_INTENSITY)).astype(np.float64)
    return im


def get_1D_gaussian_kernel(kernel_size):
    """
	Calculate a 1d gaussian blur matrix.
	:param kernel_size: The size of the matrix to return.
	:return: A gaussian blur matrix of size kernel_size^2
	"""
    if kernel_size == 1:
        return np.array([[float(1)]])
    gaussian_blur = np.array([[1, 1]])
    for i in range(0, kernel_size - 2):
        gaussian_blur = signal.convolve2d(gaussian_blur, np.array([[1, 1]])).astype(np.float64)
    return np.divide(gaussian_blur, np.sum(gaussian_blur))


def reduce(layer, kernel):
    """
	This function reduces an image by padding zeros in it's even indices and convolving with a given
	kernel.
	:param layer: The image to reduce.
	:param kernel: The kernel to convolve with.
	:return: The reduced image.
	"""
    im = scipy.ndimage.filters.convolve(layer, kernel)
    return im[::2, ::2]


def expand(layer, expanded_shape, blur_kernel):
    """
	This function expands an image by padding zeros in it's odd indices and convolving with a given
	kernel.
	:param layer: The image to expand.
	:param expanded_shape: The shape to extend to.
	:param blur_kernel: The kernel to convolve with.
	:return: The expanded image.
	"""
    blur_kernel = blur_kernel * 4
    if layer.shape == expanded_shape:
        return layer
    expanded = np.zeros(expanded_shape)
    expanded[::2, ::2] = layer
    return scipy.ndimage.filters.convolve(expanded, blur_kernel)


def stretch_matrix(matrix, new_max, new_min):
    """
	This functions stretches the values of a matrix to a given
	"""
    cur_min = np.amin(matrix)
    cur_max = np.amax(matrix)
    return (((matrix - cur_min) / (cur_max - cur_min)) * (new_max - new_min)) + new_min


def relpath(filename):
    """
	Get path to a file according to it's relative path.
	:param filename: the file
	"""
    return os.path.join(os.path.dirname(__file__), filename)


def get_blending_example(im1_path, im2_path, mask_path, im_kernel_size, mask_kernel_size):
    """
	Helper function to display an example
	:param im1_path: first image
	:param im2_path: second image
	:param mask_path: the mask
	:param im_kernel_size: images kernel size
	:param mask_kernel_size: mask kernel size
	"""
    im1 = read_image(relpath(im1_path), 2)
    im2 = read_image(relpath(im2_path), 2)
    mask = read_image(relpath(mask_path), 1)
    mask[mask > MASK_THRESHOLD] = True
    mask[mask <= MASK_THRESHOLD] = False
    mask = mask.astype(np.bool)
    blended_im = pyramid_blending(im1, im2, mask, im_kernel_size, im_kernel_size, mask_kernel_size)
    display_images([im1, im2, mask, blended_im])

    return (im1, im2, mask, blended_im)


# ------------------------------------------Ex3 Functions-------------------------------------------
def build_gaussian_pyramid(im, max_levels, filter_size):
    """
	This function constructs a gaussian pyramid of a given image.
	:param im: A grayscale image with double values in [0,1].
	:param max_levels: Max number of levels in the pyramid.
	:param filter_size: The size of the gaussian filter to be used when constructing the pyramid.
	:return: A resulting array with max_level elements where each element is grayscale image of
	level of the pyramid.
	"""
    max_level_bound = min(get_bound_on_image_size(im.shape), max_levels)
    kernel_1D = get_1D_gaussian_kernel(filter_size)
    kernel_2D = signal.convolve2d(kernel_1D, np.transpose(kernel_1D)).astype(np.float64)
    pyr = []
    reduced = im
    for i in range(0, int(max_level_bound)):
        pyr.append(reduced)
        reduced = reduce(im, kernel_2D)
        im = reduced
    return (pyr, kernel_1D)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
	This function constructs a laplacian pyramid of a given image.
	:param im: A grayscale image with double values in [0,1].
	:param max_levels: Max number of levels in the pyramid.
	:param filter_size: The size of the gaussian filter to be used when constructing the pyramid.
	:return: A resulting array with max_level elements where each element is grayscale image of
	level of the pyramid.
	"""
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    num_of_layers = len(gaussian_pyr) - 1
    kernel_2D = signal.convolve2d(filter_vec, np.transpose(filter_vec), ).astype(np.float64)
    for index, layer in zip(range(num_of_layers, -1, -1), gaussian_pyr[::-1]):
        if (index == num_of_layers):
            pyr.append(layer)
            continue
        expanded = expand(gaussian_pyr[index + 1], layer.shape, kernel_2D)
        pyr.append(np.subtract(layer, expanded))
    pyr.reverse()
    return (pyr, filter_vec)


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
	This function reconstructs an image from its laplacian pyramid
	:param lpyr: The laplacian pyramid
	:param filter_vec: The filter vector used to construct the pyramid
	:param coeff: Array of coefficients with length of number of layers
	:return: The reconstructed original image
	"""
    lpyr = np.multiply(lpyr, coeff)
    kernel_2D = signal.convolve2d(filter_vec, np.transpose(filter_vec)).astype(np.float64)
    levels = len(lpyr)
    orig_im = lpyr[levels - 1]
    for layer in range(levels - 2, -1, -1):
        orig_im = expand(orig_im, lpyr[layer].shape, kernel_2D)
        orig_im += lpyr[layer]
    return orig_im


def render_pyramid(pyr, levels):
    """
	This function renders an image that contains all the pyramid layers.
	:param pyr: The pyramid to display.
	:param levels: Number of levels to display.
	:return: The result described above.
	"""
    num_rendered_rows = pyr[0].shape[0]
    vfunc = np.vectorize(get_num_cols_of_image)
    num_rendered_cols = np.sum(vfunc(pyr[:levels]))
    pyr_im = np.zeros((num_rendered_rows, num_rendered_cols))
    cur_col = 0
    for i in range(0, levels):
        layer = stretch_matrix(pyr[i], 1, 0)
        rows, cols = layer.shape
        pyr_im[: rows, cur_col: cur_col + cols] += layer
        cur_col += cols
    return pyr_im


def display_pyramid(pyr, levels):
    """
	This function displays the pyramid in a single image.
	:param pyr: The pyramid to display.
	:param levels: The number of levels to display.
	:return: The image that contains all pyramid levels
	"""
    if levels == 0:
        return
    pyr_im = render_pyramid(pyr, levels)
    plt.imshow(pyr_im, cmap=plt.cm.gray)
    plt.show(block=True)


def RGB_pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
	Blend colored images by blending each channel separately and joining them to a single image.
	"""
    red_blend = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im,
                                 filter_size_mask)
    green_blend = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im,
                                   filter_size_mask)
    blue_blend = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im,
                                  filter_size_mask)
    blended_im = np.zeros(im1.shape)
    blended_im[:, :, 0] = red_blend
    blended_im[:, :, 1] = green_blend
    blended_im[:, :, 2] = blue_blend
    return blended_im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
	This function blends two images using laplacian pyramids.
	:param im1:First image
	:param im2:Second image
	:param mask:the mask
	:param max_levels:number of levels in the pyramids
	:param filter_size_im:image filter size
	:param filter_size_mask:mask filter size
	:return:a blended image
	"""
    if is_image_colored(im1) and is_image_colored(im2):
        return RGB_pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    lpyr_im1, filter_vec_im1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lpyr_im2, filter_vec_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gpyr_mask, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels,
                                                        filter_size_mask)
    lpyr_blended = []
    levels = len(lpyr_im1)
    for i in range(levels):
        layer = np.multiply(gpyr_mask[i], lpyr_im1[i]) + np.multiply(1 - gpyr_mask[i], lpyr_im2[i])
        lpyr_blended.append(layer)
    coeff_arr = [ORIGINAL_IMAGE_COEFF] * levels
    return laplacian_to_image(lpyr_blended, filter_vec_im1, coeff_arr).clip(0, 1)


def display_images(images):
    """
	This function displays the array of images given to it.
	:param images: An array of images.
	"""
    num_of_images = len(images)
    TITLES = []
    i=0
    for im in images:
        TITLES += [str(i)]
    fig = plt.figure(figsize=(3,3))

    for i, (image, title) in enumerate(zip(images, TITLES)):
        cur_sub_plt = fig.add_subplot(1, num_of_images, i + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        cur_sub_plt.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * num_of_images)
    plt.show()


def blending_example1():
    """
	The first example of blending done by me.
	:return:
	"""
    return get_blending_example(IM_1_EXAMPLE_1, IM_2_EXAMPLE_1, MASK_EXAMPLE_1, 15, 15)


def blending_example2():
    """
	The second example of blending done by me.
	:return:
	"""
    return get_blending_example(IM_1_EXAMPLE_2, IM_2_EXAMPLE_2, MASK_EXAMPLE_2, 15, 15)


# im = read_image('/Users/shaulr/IdeaProjects/HoleFilling/src/test.png', 2)
# display_images([im])
