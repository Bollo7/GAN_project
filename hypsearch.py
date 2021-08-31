import argparse
import os
from train_wgan import WGAN
from train_gan import GAN
import timeit

desc = 'Hyperparameter search for GANs'
parser = argparse.ArgumentParser(description=desc)

groups = ['hyp_search', 'gan_hyps']

hyp_search = parser.add_argument_group('hyp_search')
hyp_search.add_argument('--LR_Ds', type=float, nargs='*', help='LR_D list 1, usage example "--LR_Ds 0.0001 0.0002 0.0004"')
hyp_search.add_argument('--LR_Gs', type=float, nargs='*', help='LR_G list 2, usage example "--LR_Gs 0.0001 0.0002 0.0004"')
hyp_search.add_argument('--SEEDS', type=int, nargs='*', help='SEED list, usage example "--SEEDS 4242 4343 4444"')
hyp_search.add_argument('--BATCH_SIZES', type=int, nargs='*', help='BATCH_SIZES list, usage example "--BATCH_SIZES 64 128"')


gan_hyps = parser.add_argument_group('gan_hyps')

gan_hyps.add_argument('--gan_type', type=str, default='WGAN',
                            choices=['GAN', 'WGAN'],
                            help='The type of GAN')

gan_hyps.add_argument('--INCEPTION_BATCH_SIZE', type=int, default=50, help='The size of the batch that is used to pass samples of images to the InceptionNet')
gan_hyps.add_argument('--IMAGE_SIZE', type=int, default=64, help='The size of input image')
gan_hyps.add_argument('--INPUT_NOISE', type=int, default=100, help='Input noise for generator')
gan_hyps.add_argument('--GENERATOR_FILTERS', type=int, default=64, help='The size of convolution filters of G')
gan_hyps.add_argument('--DISCRIMINATOR_FILTERS', type=int, default=64, help='The size of convolution filters of D')
gan_hyps.add_argument('--KERNEL_SIZE', type=int, default=4, help='The size of kernel for convolution layers')
gan_hyps.add_argument('--NUMBER_CHANNELS', type=int, default=3, help='The number of input channels')
gan_hyps.add_argument('--N_EPOCHS', type=int, default=25, help='The number of epochs to run')
gan_hyps.add_argument('--B1', type=float, default=0.5, help='Beta 1')
gan_hyps.add_argument('--B2', type=float, default=0.999, help='Beta 2')
gan_hyps.add_argument('--VECTOR_LEN', type=int, default=100, help='The number of epochs to run')
gan_hyps.add_argument('--C', type=float, default=0.01, help='Clipping value')
gan_hyps.add_argument('--DEVICE', type=int, default=0, help='CUDA device number')

gan_hyps.add_argument('--hyper_dir', type=str, default='hyper_saves', help='Main directory to store all the results from hyperparameter search')
gan_hyps.add_argument('--save_dir', type=str, default='search_saves', help='Directory name to save pictures to')
gan_hyps.add_argument('--result_dir', type=str, default='search_results', help='Directory name to save generated images to')
gan_hyps.add_argument('--log_dir', type=str, default='search_runs', help='Directory name to save training logs to')
gan_hyps.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')

class Hyperparams():
	def __init__(self, args):

		self.INCEPTION_BATCH_SIZE = args['INCEPTION_BATCH_SIZE']
		self.BATCH_SIZE = args['BATCH_SIZE']
		self.IMAGE_SIZE = args['IMAGE_SIZE']
		self.INPUT_NOISE = args['INPUT_NOISE']
		self.GENERATOR_FILTERS = args['GENERATOR_FILTERS']
		self.DISCRIMINATOR_FILTERS = args['DISCRIMINATOR_FILTERS']
		self.KERNEL_SIZE = args['KERNEL_SIZE']
		self.NUMBER_CHANNELS = args['NUMBER_CHANNELS']
		self.N_EPOCHS = args['N_EPOCHS']
		self.LR_G = args['LR_G']
		self.LR_D = args['LR_D']
		self.B1, self.B2 = args['B1'], args['B2']
		self.VECTOR_LEN = args['VECTOR_LEN']
		self.C = args['C']
		self.SEED = args['SEED']
		self.DEVICE = args['DEVICE']

		self.hyper_dir = args['hyper_dir']
		self.save_dir = args['save_dir']
		self.result_dir = args['result_dir']
		self.log_dir = args['log_dir']
		self.dataset = args['dataset']

def main():
	args = parser.parse_args()

	allparam_dic = {}
	for group in parser._action_groups:
		if group.title in groups:
			allparam_dic[group.title] = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}

	if args is None:
		exit()

	print(f'# # # Hyperparameter search lists: '
	      f'LR_Ds: {args.LR_Ds}, LR_Gs: {args.LR_Gs} # # #')

	print(f'Selected seeds: {args.SEEDS}')
	print(f'Selected batch sizes: {args.BATCH_SIZES}')

	try:
		os.makedirs(f'{args.hyper_dir}')
	except FileExistsError:
		pass

	for b in args.BATCH_SIZES:
		for k in args.SEEDS:
			for i in args.LR_Ds:
				for j in args.LR_Gs:

					allparam_dic['gan_hyps']['LR_D'] = i
					allparam_dic['gan_hyps']['LR_G'] = j
					allparam_dic['gan_hyps']['SEED'] = k
					allparam_dic['gan_hyps']['BATCH_SIZE'] = b

					gan_type = allparam_dic['gan_hyps']['gan_type']
					#seed = allparam_dic['gan_hyps']['SEED']
					batch_size = allparam_dic['gan_hyps']['BATCH_SIZE']
					save_dir, result_dir, log_dir = f'saves_LR_D{str(i)}_LR_G{str(j)}_{gan_type}_{k}_{batch_size}', f'results_LR_D{str(i)}_LR_G{str(j)}_{gan_type}_{k}_{batch_size}', f'logs_LR_D{str(i)}_LR_G{str(j)}_{gan_type}_{k}_{batch_size}'

					try:
						os.makedirs(f'{args.hyper_dir}/{save_dir}')
						os.makedirs(f'{args.hyper_dir}/{result_dir}')
						os.makedirs(f'{args.hyper_dir}/{log_dir}')
					except FileExistsError:
						pass

					save_dir, result_dir, log_dir = f'{args.hyper_dir}/{save_dir}', f'{args.hyper_dir}/{result_dir}', f'{args.hyper_dir}/{log_dir}'
					allparam_dic['gan_hyps']['save_dir'], allparam_dic['gan_hyps']['result_dir'], allparam_dic['gan_hyps']['log_dir'] = save_dir, result_dir, log_dir

					hyperparams = Hyperparams(allparam_dic['gan_hyps'])

					if args.gan_type == 'GAN':
						gan = GAN(hyperparams)
					elif args.gan_type == 'WGAN':
						gan = WGAN(hyperparams)

					print(f'# # # Training with LR_G = {str(j)}, LR_D = {str(i)}, BATCH_SIZE = {str(b)} # # #')
					starting_time = timeit.default_timer()

					gan.train()

					stop_time = timeit.default_timer()
					print('Time to train: ', stop_time - starting_time)

					gan.save()

	print(f'# # # Finished hypersearch! # # #')

if __name__ == '__main__':
	main()
