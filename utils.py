import numpy as np
from PIL import Image
import glob
import os
from numpy.random import randint, uniform, poisson


def crop(img_array, box_size):
    img_height, img_width, img_channels = img_array.shape
    if img_channels == 4:
        img_array = img_array[:, :, 0:3]

    if img_height < box_size or img_width < box_size:
        raise Exception('Unable to crop.')

    up = randint(0, img_height - box_size)
    left = randint(0, img_width - box_size)

    return img_array[up:up + box_size, left:left + box_size]


def imread(filename):
    img = Image.open(filename)
    width, height = img.size
    img_array = np.array(img, dtype=np.uint8).reshape([height, width, -1])
    return img_array


def imsave(filename, array):
    img = Image.fromarray(array)
    img.save(filename)


def add_poisson_noise(img, peak=1.0):
    """
    Add poisson noise to image.
    """
    max_val = np.max(img).astype(np.float)
    ratio = peak / max_val
    scaled_img = img * ratio
    noisy = poisson(lam=scaled_img) / ratio
    return np.clip(noisy, 0.0, max_val)


def crop_images(jpg_dir, box_size=128, samples=None):
    res = []
    paths = glob.glob(os.path.join(jpg_dir, '*.jpg'))

    if samples:
        paths = paths[:samples]

    for path in paths:
        image = imread(path)

        # we only handle colorful images
        if image.shape[-1] != 3:
            continue

        try:
            cropped_image = crop(image, box_size)
        except:
            print('{} is unable to crop.'.format(path.split('/')[-1]))
            continue

        max_val = np.max(cropped_image)
        if max_val == 0:
            print('{} has zero max value.'.format(path.split('/')[-1]))
            continue

        res.append(cropped_image)

    return np.stack(res)


def add_poisson_noise_to_images(images):
    res = []
    for image in images:
        noisy_image = add_poisson_noise(image, peak=uniform(1.0, 30.0))
        res.append(noisy_image)
    return np.stack(res)


def RGB2YCbCr(r, g, b):
    """
    https://en.wikipedia.org/wiki/YCbCr
    """
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
    cr = 128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
    return y, cb, cr


def YCbCr2RGB(y, cb, cr):
    """
    https://en.wikipedia.org/wiki/YCbCr
    """
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return r, g, b


def psnr(x, y):
    """
    x: ground turth
    y: noisy image
    """
    mse = np.array((x - y)**2).mean()
    max_x = np.max(x)
    psnr = 10 * np.log10(max_x**2 / mse)
    # print('psnr: {}, mse: {}, max_x: {}'.format(psnr, mse, max_x))
    return psnr


def avg_psnr(x, y):
    return np.mean([psnr(xx, yy) for xx, yy in zip(x, y)])
