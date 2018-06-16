import numpy as np
import scipy.signal as signal
import scipy.ndimage
from scipy.misc import imread as imread
# import skimage.color as skimage
import matplotlib.pyplot as plt
import os

# --------------------------------------------Constants---------------------------------------------
CONVERSION_MTRX = np.matrix([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
YIQ_REP = 1
RGB_REP = 2
MAX_INTENSITY = 255
NUM_OF_BINS = 256
ORIGINAL_IMAGE_COEFF = 1

# ----------------------------------------Lambda Functions------------------------------------------
"""
Returns True iff the image is colored
"""
is_image_colored = lambda image: len(image.shape) > 2

get_bound_on_image_size = lambda shape: np.ceil(np.log2(min(shape))) - 4

get_num_cols_of_image = lambda im: im.shape[1]


# ----------------------------------------Helper Functions------------------------------------------

def get_gaussian_blur_kernel(kernel_size):
    """
	Calculate a 2d gaussian blur matrix.
	:param kernel_size: The size of the matrix to return.
	:return: A gaussian blur matrix of size kernel_size^2
	"""
    if kernel_size == 1:
        return np.array([float(1)]).reshape(1, 1)
    blur_kernel = np.array([[1, 1]])
    for i in range(0, kernel_size - 2):
        blur_kernel = scipy.signal.convolve2d(blur_kernel, np.array([[1, 1]])).astype(np.float64)
    gaussian_blur = scipy.signal.convolve2d(blur_kernel, np.transpose(blur_kernel)).astype(np.float64)
    return np.divide(gaussian_blur, np.sum(gaussian_blur))


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


# ---------------------------------------sol4_utils Functions---------------------------------------
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


def blur_spatial(im, kernel_size):
    """
	This function performs a blurring on an image using a 2d convolution with a gaussian blur.
	:param im: The image to blur.
	:param kernel_size: The size of the gaussian blur matrix.
	:return: A blurred image of the same size.
	"""
    gaussian_kernel = get_gaussian_blur_kernel(kernel_size)
    return scipy.signal.convolve2d(im, gaussian_kernel, mode='same', boundary='symm')


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
