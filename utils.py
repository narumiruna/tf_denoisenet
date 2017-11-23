import numpy as np


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
    noisy = np.random.poisson(lam=img_peak).astype(np.float32)
    noisy = noisy / peak - 0.5
    return noisy


def RGB2YCbCr(img_array, dtype=None):
    """
    https://en.wikipedia.org/wiki/YCbCr
    """
    if np.issubdtype(img_array.dtype, np.float):
        img_array *= 255.0

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
    cr = 128.0 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)

    ycbcr = np.stack([y, cb, cr], axis=2)
    if dtype:
        return ycbcr.astype(dtype)
    else:
        return ycbcr


def YCbCr2RGB(img_array, dtype=None):
    """
    https://en.wikipedia.org/wiki/YCbCr
    """
    y = img_array[:, :, 0]
    cb = img_array[:, :, 1]
    cr = img_array[:, :, 2]
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)

    rgb = np.stack([r, g, b], axis=2)

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

    plt.imshow(ycbcr_img[:, :, 0], cmap='gray')
    plt.imshow(ycbcr_img.astype(np.uint8))
    plt.show()

    # recover_img = add_noise(recover_img)
    plt.imshow(recover_img.astype(np.uint8))
    plt.show()


if __name__ == '__main__':
    main()
