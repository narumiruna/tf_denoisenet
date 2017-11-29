import numpy as np
from PIL import Image


def crop(img_array, box_height=128, box_width=128):
    img_height, img_width, img_channels = img_array.shape

    if img_height < box_height or img_width < box_width:
        raise Exception('Unable to crop.')

    up = random.randint(0, img_height - box_height)
    left = random.randint(0, img_width - box_width)

    return img_array[up:up + box_height, left:left + box_width]


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
    noisy = np.random.poisson(lam=scaled_img) / ratio
    return np.clip(noisy, 0.0, max_val)


def crop_images(jpg_dir):
    black_img_count = 0
    small_img_count = 0

    res = []
    paths = glob.glob(os.path.join(jpg_dir, '*.jpg'))
    for path in paths:
        image = imread(path)

        try:
            cropped_image = crop(image)
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

        noisy_image = add_poisson_noise(image, peak=random.uniform(1.0, 30.0))
#         noisy_image = add_poisson_noise(image, peak=1.0)
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


# def RGB2YCbCr(img_array, dtype=None):
#     """
#     https://en.wikipedia.org/wiki/YCbCr
#     """
#     if np.issubdtype(img_array.dtype, np.float):
#         img_array *= 255.0

#     if len(img_array.shape) == 3:
#         r = img_array[:, :, 0]
#         g = img_array[:, :, 1]
#         b = img_array[:, :, 2]
#     elif len(img_array.shape) == 4:
#         r = img_array[:, :, :, 0]
#         g = img_array[:, :, :, 1]
#         b = img_array[:, :, :, 2]

#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     cb = 128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
#     cr = 128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)

#     ycbcr = np.stack([y, cb, cr], axis=-1)

#     if dtype:
#         return ycbcr.astype(dtype)
#     else:
#         return ycbcr


# def YCbCr2RGB(img_array, dtype=None):
#     """
#     https://en.wikipedia.org/wiki/YCbCr
#     """
#     if len(img_array.shape) == 3:
#         y = img_array[:, :, 0]
#         cb = img_array[:, :, 1]
#         cr = img_array[:, :, 2]
#     elif len(img_array.shape) == 4:
#         y = img_array[:, :, :, 0]
#         cb = img_array[:, :, :, 1]
#         cr = img_array[:, :, :, 2]

#     r = y + 1.402 * (cr - 128.0)
#     g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
#     b = y + 1.772 * (cb - 128.0)

#     rgb = np.stack([r, g, b], axis=-1)

#     if dtype:
#         return rgb.astype(dtype)
#     else:
#         return rgb
