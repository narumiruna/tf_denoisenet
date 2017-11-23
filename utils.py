import numpy as np
import skimage


def rescale(img_array):
    """
    Rescale image from [0, 255] to [0, 1] and shift [0, 1] to [-0.5, 0.5].
    """
    return img_array.astype(np.float32) / 255.0 - 0.5


def add_noise(img_array, peak=1.0):
    """
    Add poisson noise to image.
    """
    peak = float(peak)
    img_peak = (img_array + 0.5) * peak
    noisy = np.random.poisson(lam=img_peak)
    noisy = noisy / peak - 0.5
    return noisy


def RGB2YCbCr(img_array, dtype=None):
    """
    https://en.wikipedia.org/wiki/YCbCr
    """
    if np.issubdtype(img_array.dtype, np.float):
        img_array *= 255.0

    if len(img_array.shape) == 3:
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
    elif len(img_array.shape) == 4:
        r = img_array[:, :, :, 0]
        g = img_array[:, :, :, 1]
        b = img_array[:, :, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
    cr = 128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)

    ycbcr = np.stack([y, cb, cr], axis=-1)
    
    if dtype:
        return ycbcr.astype(dtype)
    else:
        return ycbcr


def YCbCr2RGB(img_array, dtype=None):
    """
    https://en.wikipedia.org/wiki/YCbCr
    """
    if len(img_array.shape) == 3:
        y = img_array[:, :, 0]
        cb = img_array[:, :, 1]
        cr = img_array[:, :, 2]
    elif len(img_array.shape) == 4:
        y = img_array[:, :, :, 0]
        cb = img_array[:, :, :, 1]
        cr = img_array[:, :, :, 2]

    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)

    rgb = np.stack([r, g, b], axis=-1)

    if dtype:
        return rgb.astype(dtype)
    else:
        return rgb


def main():
    import matplotlib.pyplot as plt
    img = plt.imread('sakura.jpg')

    ycbcr_img = RGB2YCbCr(img)

    recover_img = YCbCr2RGB(ycbcr_img)

    loss = np.linalg.norm(img - recover_img) / np.linalg.norm(img)
    print(loss)

    _, axarr = plt.subplots(1, 4)
    axarr[0].imshow(ycbcr_img.astype(np.uint8))
    axarr[1].imshow(recover_img.astype(np.uint8))
    img3 = add_noise(recover_img, peak=1.0) - recover_img
    axarr[2].imshow(img3.astype(np.uint8))
    axarr[3].imshow((skimage.util.random_noise(recover_img/255.0, mode='poisson')*255.0).astype(np.uint8))
    plt.show()


if __name__ == '__main__':
    main()
