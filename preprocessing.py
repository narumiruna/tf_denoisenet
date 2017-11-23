import glob
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import os
from datetime import datetime


def crop_voc_images(img_dir='data'):
    # create image dir if not exists
    os.makedirs(img_dir, exist_ok=True)

    # prefix of output filename
    filename_prefix = datetime.now().strftime('%Y%m%d%H%M%S')

    # get all voc jpge iamge file paths
    filepaths = glob.glob('VOC2012/JPEGImages/*')

    for i, f in enumerate(filepaths):
        img = plt.imread(f)

        try:
            crop_img = crop(img)
        except:
            continue

        plt.imsave('{}/{}_{}.jpg'.format(img_dir,
                                         filename_prefix, str(i).zfill(6)), crop_img)

        if i % 1000 == 0:
            print('完程度: {}'.format(i / len(filepaths)))


def crop(img_array,  box_height=128, box_width=128):
    img_height, img_width, _ = img_array.shape
    if img_height < box_height or img_width < box_width:
        raise Exception('Crop box ({}, {}) is larger than image ({}, {}).'.format(
            box_width, box_height, img_height, img_width))
    x = randint(0, img_height - box_height)
    y = randint(0, img_width - box_width)
    return img_array[x:x + box_height, y:y + box_width]


def check():
    filepaths = glob.glob('data/*.jpg')
    for i, f in enumerate(filepaths):
        img = plt.imread(f)
        h, w, _ = img.shape
        if h != 128 or w != 128:
            print(f)


def main():
    crop_voc_images()


if __name__ == '__main__':
    main()
