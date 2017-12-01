import tensorflow as tf 
from train import train
import argparse
import utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--cropping_size', '-cs', type=int, default=64)
parser.add_argument('--channels', '-ch', type=int, default=3)
parser.add_argument('--dir', '-d', type=str, default='images')
parser.add_argument('--layers', '-l', type=int, default=6)
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--filters', '-f', type=int, default=64)
parser.add_argument('--save_path', '-sp', type=str, default=None)
args = parser.parse_args()


def compare_bn():
    # load data and crop
    gt = utils.crop_images(args.dir, args.cropping_size)
    noise = utils.add_poisson_noise_to_images(gt)

    # scale to [-0.5, 0.5]
    gt_scaled = gt / 255.0 - 0.5
    noise_scaled = noise / 255.0 - 0.5

    del gt
    del noise

    losses, avg_psnr_list = train(gt_scaled,
                                  noise_scaled,
                                  args.batch_size,
                                  args.learning_rate,
                                  args.layers,
                                  args.epochs,
                                  args.filters,
                                  args.save_path)

    bn_losses, bn_avg_psnr_list = train(gt_scaled,
                                        noise_scaled,
                                        args.batch_size,
                                        args.learning_rate,
                                        args.layers,
                                        args.epochs,
                                        args.filters,
                                        args.save_path,
                                        batch_norm=True)

    # plot
    _, ax = plt.subplots(ncols=2)
    ax[0].plot(losses, label='dn')
    ax[0].plot(bn_losses, label='dn batch norm')
    ax[1].plot(avg_psnr_list, label='dn')
    ax[1].plot(bn_avg_psnr_list, label='dn batch norm')
    plt.legend()
    plt.show()


def main():

    # load data and crop
    gt = utils.crop_images(args.dir, args.cropping_size)
    noise = utils.add_poisson_noise_to_images(gt)

    # scale to [-0.5, 0.5]
    gt_scaled = gt / 255.0 - 0.5
    noise_scaled = noise / 255.0 - 0.5

    del gt
    del noise

    losses, avg_psnr_list = train(gt_scaled,
                                  noise_scaled,
                                  args.batch_size,
                                  args.learning_rate,
                                  args.layers,
                                  args.epochs,
                                  args.filters,
                                  args.save_path)

    # plot
    _, ax = plt.subplots(ncols=2)
    ax[0].plot(losses)
    ax[1].plot(avg_psnr_list)
    plt.show()


if __name__ == '__main__':
    # main()
    compare_bn()
