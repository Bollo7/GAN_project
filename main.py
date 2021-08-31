from train_wgan import WGAN
from train_gan import GAN
import argparse, os, torch


desc = "GAN for deployment on the server"
parser = argparse.ArgumentParser(description=desc, add_help=True)

parser.add_argument('--gan_type', type=str, default='GAN',
                            choices=['GAN', 'WGAN'],
                            help='The type of GAN')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='The name of dataset')

parser.add_argument('--INCEPTION_BATCH_SIZE', type=int, default=50, help='The size of the batch that is used to pass samples of images to the InceptionNet')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='The size of batch')
parser.add_argument('--IMAGE_SIZE', type=int, default=64, help='The size of input image')
parser.add_argument('--INPUT_NOISE', type=int, default=100, help='Input noise for generator')
parser.add_argument('--GENERATOR_FILTERS', type=int, default=64, help='The size of convolution filters of G')
parser.add_argument('--DISCRIMINATOR_FILTERS', type=int, default=64, help='The size of convolution filters of D')
parser.add_argument('--KERNEL_SIZE', type=int, default=4, help='The size of kernel for convolution layers')
parser.add_argument('--NUMBER_CHANNELS', type=int, default=3, help='The number of input channels')
parser.add_argument('--N_EPOCHS', type=int, default=1, help='The number of epochs to run')
parser.add_argument('--LR_G', type=float, default=0.0004, help="Generator's learning rate")
parser.add_argument('--LR_D', type=float, default=0.0002, help="Discriminator's learning rate")
parser.add_argument('--B1', type=float, default=0.5, help='Beta 1')
parser.add_argument('--B2', type=float, default=0.999, help='Beta 2')
parser.add_argument('--VECTOR_LEN', type=int, default=1000, help='The number of epochs to run')
parser.add_argument('--C', type=float, default=0.01, help='Clipping value')
parser.add_argument('--SEED', type=int, default=4242, help='Seed of a current run')



parser.add_argument('--benchmark_mode', type=bool, default=False)
parser.add_argument('--save_dir', type=str, default='saves', help='Folder to save pictures to')
parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
parser.add_argument('--log_dir', type=str, default='runs', help='Directory name to save training logs')


def check_args(args):

    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.EPOCHS >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.BATCH_SIZE >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

def main():
    # parse arguments
    args = check_args(parser.parse_args())
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    if args.gan_type == 'GAN':
        gan = GAN(0, args)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    gan.train()
    print(" [*] Training finished!")
    gan.save()

    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()    #main()