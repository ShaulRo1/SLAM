import numpy as np
import random
# import sol5_utils
import skimage.color
import matplotlib.pyplot as plt
import copy

from pathlib import Path
from scipy import misc
from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
from scipy.ndimage.filters import convolve
# --------------------------------------------Constants---------------------------------------------
IMAGE_COLOR_REP = 1
GRAY_IM = 1
RGB_IM = 2
MAX_INTENSITY = 255
TRAIN_TO_VALID_RATIO = 0.8
DEFAULT_FILTER_SIZE = 3
IMAGE_VALUE_REFACTOR_SCALE = 0.5
NUM_RES_BLOCK_RANGE = [1, 2, 3, 4, 5]

MIN_SIGMA = 0
MAX_SIGMA = 0.2
DENOISING_PATCH = (24, 24)
DENOISING_NUM_OF_CHANNELS = 48

DEBLURRING_PATCH = (16, 16)
DEBLURRING_NUM_OF_CHANNELS = 32
DEFAULT_KERNEL_LIST = [7]

BONUS_IMAGE_SHAPE = (64, 64)

# ----------------------------------------Helper Functions------------------------------------------
def read_image(file_name, representation):
    """reads image and returns array of pixels in gray or RGB scale according to
     the given representation"""
    if representation == GRAY_IM:
        image = misc.imread(file_name)
        if len(image.shape) >= 3:
            image = skimage.color.rgb2gray(image)
        else:
            image = image / MAX_INTENSITY
        return image.astype(np.float64)
    else:
        image = misc.imread(file_name, mode='RGB')
        return image.astype(np.float64) / MAX_INTENSITY


def get_random_crop(im, corrupted_im, crop_height, crop_width):
    """
    This function returns a random cropped patch of an image and corresponding patch in the same
    blurred image.
    :param im: The original image.
    :param corrupted_im: The corrupted image.
    :return: A tupple with the patch of the original image and the blurred image.
    """
    patch_indices = (random.randint(0, im.shape[0] - crop_height),
                     random.randint(0, im.shape[1] - crop_width))
    patch = im[patch_indices[0]: patch_indices[0] + crop_height,
            patch_indices[1]: patch_indices[1] + crop_width].reshape(1, crop_height, crop_width)
    corrupted_patch = corrupted_im[patch_indices[0]: patch_indices[0] + crop_height,
                      patch_indices[1]: patch_indices[1] + crop_width].reshape(1, crop_height, crop_width)
    return (patch, corrupted_patch)

# ----------------------------------------sol5 Functions--------------------------------------------
def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    This function outputs data_generator, a Python’s generator object which outputs random tuples of
    the form (source_batch, target_batch), where each output variable is an array of shape
    (batch_size, 1,height, width), target_batch is made of clean images, and source_batch is their
    respective randomly corrupted version according to corruption_func(im).
    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient
    				   Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a
    						single argument,and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract
    :return: A python generator
    """
    img_dict = {}
    height = crop_size[0]
    width = crop_size[1]
    while True:
        rand_images_indices =np.random.choice(np.arange(len(filenames)), batch_size)
        rand_img_names = np.array(filenames)[rand_images_indices]
        source_batch = []
        target_batch = []
        for rand_file_name in rand_img_names:
            if rand_file_name in img_dict:
                im = img_dict[rand_file_name]
            else:
                im = read_image(rand_file_name, IMAGE_COLOR_REP)
                img_dict[rand_file_name] = im
            corrupted_im = corruption_func(im)
            patch, corrupted_patch = get_random_crop(im, corrupted_im, height, width)
            source_batch.append(corrupted_patch - IMAGE_VALUE_REFACTOR_SCALE)
            target_batch.append(patch - IMAGE_VALUE_REFACTOR_SCALE)
        yield (np.array(source_batch), np.array(target_batch))


def resblock(input_tensor, num_channels):
    """
    The above function takes as input a symbolic input tensor and the number of channels for each
     of its convolutional layers, and returns the symbolic output tensor of the layer configuration.
    :return: The output tensor of a single residual block.
    """
    convolved = Convolution2D(num_channels, DEFAULT_FILTER_SIZE, DEFAULT_FILTER_SIZE,
                      border_mode='same')(input_tensor)
    activated = Activation('relu')(convolved)
    result = Convolution2D(num_channels, DEFAULT_FILTER_SIZE, DEFAULT_FILTER_SIZE,
                      border_mode='same')(activated)
    return merge([input_tensor, result], mode='sum')


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    The above function should return an untrained Keras model.
    :param height: The height of the layers.
    :param width: The width of the layers.
    :param num_channels: The number of channels of the convolution.
    :param num_res_blocks: The number of residual blocks of the neural network.
    :return: An untrained ResNet with num_res_blocks residual blocks.
    """
    input = Input(shape=(IMAGE_COLOR_REP, height, width))
    convolved = Convolution2D(num_channels, DEFAULT_FILTER_SIZE, DEFAULT_FILTER_SIZE,
                              border_mode='same')(input)
    pre_resblocks = Activation('relu')(convolved)
    post_resblock = pre_resblocks
    for _ in range(num_res_blocks):
        post_resblock = resblock(post_resblock, num_channels)
    summed_layer = merge([pre_resblocks, post_resblock], mode='sum')
    out = Convolution2D(1, DEFAULT_FILTER_SIZE, DEFAULT_FILTER_SIZE, border_mode='same')(summed_layer)
    return Model(input=input, output=out)


def train_model(model, images, corruption_func, batch_size, samples_per_epoch,
                num_epochs, num_valid_samples):
    """
    The above function divides the images into a training set and validation set, using an 80-20
    split, and generate from each set a dataset with the given batch size and corruption
    function. Then, it calls the compile() method of the model using the “mean squared
    error” loss and ADAM optimizer. Instead of the default values for ADAM,it uses
    Adam(beta_2=0.9). Finally, it calls fit_generator to actually train the model.
    :param model: A general neural network model for image restoration.
    :param images: A list of file paths pointing to image files. You should assume these paths are
     complete, and should append anything to them.
    :param corruption_func: A corruption function
    :param batch_size: The size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch (actual samples, not batches!).
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every
    epoch
    :return: A trained model
    """
    train_im = images[: int(len(images) * TRAIN_TO_VALID_RATIO)]
    valid_im = images[int(len(images) * TRAIN_TO_VALID_RATIO) :]
    crop_size = model.input_shape[2:]
    train_set = load_dataset(train_im, batch_size, corruption_func, crop_size)
    valid_set = load_dataset(valid_im, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_set,
                        samples_per_epoch=samples_per_epoch,
                        nb_epoch=num_epochs,
                        validation_data=valid_set,
                        nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    This function uses a neural network to restore a corrupted image.
    :param corrupted_image: A grayscale image of shape (height, width) and with values in the [0, 1]
     range of type float64
    :param base_model: A neural network trained to restore small patches
    :return: The restored image
    """
    input_shape = (IMAGE_COLOR_REP, ) + corrupted_image.shape
    a = Input(shape=input_shape)
    b = base_model(a)
    new_model = Model(input=a, output=b)
    new_cor_im = (corrupted_image - IMAGE_VALUE_REFACTOR_SCALE).reshape((1, 1,) + corrupted_image.shape)
    res_im = new_model.predict(new_cor_im)[0, 0, :, :].astype(np.float64)
    res_im = res_im + IMAGE_VALUE_REFACTOR_SCALE
    return np.clip(res_im, 0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    This function randomly samples a value of sigma, uniformly distributed between min_sigma and
    max_sigma, followed by adding to every pixel of the input image a zero-mean gaussian random variable
    with standard deviation equal to sigma.
    :param image: The original image.
    :param min_sigma: Min sigma value
    :param max_sigma: Max sigma value
    :return: A blurred image with values of type i/256
    """
    sigma = random.random()*(max_sigma - min_sigma) + min_sigma
    mean = 0
    gaussian_matrix = np.random.normal(mean, sigma, image.shape)
    corrupted_im = np.add(gaussian_matrix, image)
    res = (np.around(corrupted_im * 256)) / 256
    return  np.clip(res, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    The above method should train a network which expect patches of size 24×24, using 48 channels
    for all but the last layer. The corruption you will use is a gaussian noise with sigma 6
    in the range [0, 0.2]
    :param num_res_blocks: Number of residual blocks in the NN
    :param quick_mode: If true the train model arguments are changed according to the ex5
    description.
    :return: A trained model.
    """
    filenames = sol5_utils.images_for_denoising()
    corruption_func = lambda im: add_gaussian_noise(im, MIN_SIGMA, MAX_SIGMA)
    batch_size, samples_per_epoch, num_epochs, num_valid_samples = \
        (10, 30, 2, 30) if quick_mode else (100, 10 ** 4, 5, 10 ** 3)
    model = build_nn_model(*DENOISING_PATCH, DENOISING_NUM_OF_CHANNELS, num_res_blocks)
    train_model(model, filenames, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples)
    return model


def random_motion_blur(image, list_of_kernel_sizes):
    """
    The function samples an angle at uniform from the range [0, π), and choses a kernel size at
    uniform from the list list_of_kernel_sizes, followed by applying the previous function with
    the given image and the randomly sampled parameters.
    :param image: The original image
    :param list_of_kernel_sizes: A list of optional kernel sizes
    :return: A blurred image.
    """
    kernel_size = random.choice(list_of_kernel_sizes)
    angle = np.random.uniform(0, np.pi)
    return add_motion_blur(image, kernel_size, angle)


def add_motion_blur(image, kernel_size, angle):
    """
    This function should simulate motion blur on the given image using a square kernel
    of size kernel_size where the line (as described above) has the given angle in radians, measured
    relative to the positive horizontal axis,
    :param image: The image
    :param kernel_size: The size of the kernel
    :param angle: The angle in which we blur the image.
    :return: A blurred image.
    """
    motion_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, motion_kernel)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    The above method should train a network which expect patches of size 16×16,
    and have 32 channels in all layers except the last. The dataset should use a random motion blur
    of kernel size equal to 7
    :param num_res_blocks: Number of residual blocks in the NN
    :param quick_mode: If true the train model arguments are changed according to the ex5
    description.
    :return: A trained model.
    """
    filenames = sol5_utils.images_for_deblurring()
    corruption_func = lambda im: random_motion_blur(im, DEFAULT_KERNEL_LIST)
    batch_size, samples_per_epoch, num_epochs, num_valid_samples = \
        (10, 30, 2, 30) if quick_mode else (100, 10 ** 4, 10, 10 ** 3)
    model = build_nn_model(*DEBLURRING_PATCH, DEBLURRING_NUM_OF_CHANNELS, num_res_blocks)
    train_model(model, filenames, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples)
    return model

# ----------------------------------------Bonus-----------------------------------------------------
def bonus_load_dataset(corrupted_im, random_im, max_sigma, min_sigma):
    """
    This function adds random normal noise to a given random image.
    :param corrupted_im: The corrupted image (target)
    :param random_im: A uniformly distributed noise in the shape of the corrupted image (source)
    :param max_sigma: The maximum sigma to add normal noise to the uniform image
    :param min_sigma: The minimum sigma to add normal noise to the uniform image
    :return:
    """
    mean = 0
    target = corrupted_im.reshape(1, 1, *BONUS_IMAGE_SHAPE)
    while True:
        gaussian_matrix = np.random.normal(mean, 0.3, corrupted_im.shape)
        source = (random_im + gaussian_matrix).reshape(1, 1, *BONUS_IMAGE_SHAPE)
        yield (np.array(source), np.array(target))


def test_deep_restor(corrupted_image,
                     num_epochs_range=(900,700,900,1000),
                     num_channels_range=(12,24,32,48),
                     num_resblocks=(5,10,15,20,25)):
    """
    Using this function I tried to find the correct arguments to build the network, all results
    were bad, I assigned the best arguments to the deep_prior_restore image below.
    """
    pathlist = Path('/cs/usr/shaul_ro/safe/Image_Processing/ex5-shaul_ro/bonus_tests/images').glob(
        '*.png')
    paths = []
    for path in pathlist:
        paths.append(str(path))
    for num_ep in num_epochs_range:
        for num_chan in num_channels_range:
            for num_res in num_resblocks:
                name = '/cs/usr/shaul_ro/safe/Image_Processing/ex5-shaul_ro/bonus_tests/images/' \
                       + str(num_chan) + '_channels___' + str(num_ep) + '_epochs__' +str(
                    num_res)+ '_resblocks__.png'
                if name in paths:
                    # it took a few days to run, if the image was already generated I dont want to
                    #create it again.
                    continue
                random_im = np.random.uniform(0, 1, corrupted_image.shape)
                original_random_im = copy.deepcopy(random_im.reshape(1, 64, 64))
                bonus_generator = bonus_load_dataset(corrupted_image, random_im, 0, 0.2)
                model = build_nn_model(64, 64, num_chan, num_res)
                model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
                model.fit_generator(bonus_generator,
                                    samples_per_epoch=1,
                                    nb_epoch=num_ep)
                restored_image = model.predict(original_random_im[np.newaxis, ...])[0, 0].astype(
                    np.float64)
                restored_image = np.clip(restored_image, 0, 1)
                plt.gray()
                plt.imsave(name, restored_image)



def deep_prior_restore(corrupted_image):
    """
    uses "Deep Image Prior" principles to restore the given corrupted image, using only the corrupted
    image itself as a dataset.
    :param corrupted_image: the corrupted image
    :return: the restored image
    """
    random_im = np.random.uniform(0, 1, corrupted_image.shape)
    original_random_im = copy.deepcopy(random_im.reshape(1, 64, 64))
    bonus_generator = bonus_load_dataset(corrupted_image, random_im, 0, 0.2)
    model = build_nn_model(64, 64, 32, 32)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(bonus_generator,
                        samples_per_epoch=1,
                        nb_epoch=800)
    restored_image = model.predict(original_random_im[np.newaxis, ...])[0, 0].astype(np.float64)
    restored_image = np.clip(restored_image, 0, 1)
    return restored_image


