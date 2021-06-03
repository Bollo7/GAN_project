import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import os
#from torch.utils.tensorboard import SummaryWriter
from architecture import *
import datetime
import json
import pickle
import gc
from inception import *
import matplotlib.pyplot as plt
import time
import scipy
import numpy.linalg as la


def calc_seed(date=str(datetime.datetime.now())):
	year, month, day = int(date[:4]), int(date[5:7]), int(date[8:10])
	hours, minutes = int(date[11:13]), int(date[14:16])
	seed = year + month + day * (hours + minutes)
	return round(seed)

def w2_mod(v1, v2):
	'''
	FID calculation. Upgraded to handle complex values (if the vector is not of length 2048). Based on original FID implementation, reimplemented in PyTorch
	(source: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py).

	:param v1: Real vector
	:param v2: Fake vector
	:return: FID value
	'''
	eps = 1e-8
	m1, m2 = np.mean(v1, 0), np.mean(v2, 0)
	c1, c2 = np.cov(v1.T), np.cov(v2.T)
	norm = la.norm(m1 - m2)
	#sigmas = np.dot(c1, c2)
	offset = np.eye(c1.shape[0]) * eps
	sigmas = scipy.linalg.sqrtm((c1 + offset).dot(c2 + offset))

	if np.iscomplexobj(sigmas):
		if not np.allclose(np.diagonal(sigmas).imag, 0, atol=1e-3):
			m = np.max(np.abs(sigmas.imag))
			raise ValueError("Imaginary component {}".format(m))
		sigmas = sigmas.real

	tr_sigmas = np.trace(sigmas)
	res = norm**2 + np.trace(c1) + np.trace(c2) - 2 * (tr_sigmas)
	return res

class WGAN(object):
	def __init__(self, args):

		self.DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
		self.BATCH_SIZE = args.BATCH_SIZE
		self.IMAGE_SIZE = args.IMAGE_SIZE
		self.INPUT_NOISE = args.INPUT_NOISE
		self.GENERATOR_FILTERS = args.GENERATOR_FILTERS
		self.DISCRIMINATOR_FILTERS = args.DISCRIMINATOR_FILTERS
		self.KERNEL_SIZE = args.KERNEL_SIZE
		self.NUMBER_CHANNELS = args.NUMBER_CHANNELS
		self.N_EPOCHS = args.N_EPOCHS
		self.LR_G = args.LR_G
		self.LR_D = args.LR_D
		self.B1, self.B2 = args.B1, args.B2
		self.VECTOR_LEN = args.VECTOR_LEN
		self.C = args.C
		self.SEED = args.SEED

		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.log_dir = args.log_dir
		self.dataset = args.dataset
		self.model_name = 'WGAN'

		self.params = {
		        'BATCH_SIZE' : self.BATCH_SIZE,
		        'IMAGE_SIZE' : self.IMAGE_SIZE,
		        'NUMBER_CHANNELS': self.NUMBER_CHANNELS,
		        'INPUT_NOISE': self.INPUT_NOISE,
		        'GENERATOR_FILTERS': self.GENERATOR_FILTERS,
		        'DISCRIMINATOR_FILTERS': self.DISCRIMINATOR_FILTERS,
		        'LR_G': self.LR_G,
			    'LR_D':self.LR_D,
		        'N_EPOCHS': self.N_EPOCHS,
		        'KERNEL_SIZE': self.KERNEL_SIZE,
		        'B1': self.B1,
				'B2':self.B2,
		        'VECTOR_LEN': self.VECTOR_LEN,
				'dataset':self.dataset,
				'SEED':self.SEED
			}

		#torch.backends.cudnn.benchmark = True

		### Calculate seed for current

		# self.SEED = calc_seed()
		torch.manual_seed(self.SEED)
		# self.params['SEED'] = self.SEED
		print(f"Current run's seed: {self.SEED}, GAN type: {self.model_name}")

		### Create folders, save parameters and define writer ###

		try:
			os.makedirs(f'{self.log_dir}/WGAN_CIFAR10/test_real{self.SEED}')
			os.makedirs(f'{self.log_dir}/WGAN_CIFAR10/test_fake{self.SEED}')
		except FileExistsError:
			pass

		try:
			os.makedirs(f'{self.log_dir}/configs')

		except FileExistsError:
			pass

		with open(f'{self.log_dir}/configs/settings_{self.SEED}.json', 'w+') as f:
			json.dump(self.params, f)

		#self.writer = SummaryWriter(f'runs/WGAN_CIFAR10/{self.SEED}')

		### Transform data and initialize data loaders ###

		self.transform = transforms.Compose([
			transforms.Resize(self.IMAGE_SIZE),
			transforms.ToTensor(),
			# transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))
			transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

		self.train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=self.transform, download=True)

		# loader for training and for vector of 2000 images for Inception score
		self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True,                               pin_memory=True)

		self.real_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.VECTOR_LEN, shuffle=True)

		### initialize optimizers and networks' weights ###

		self.G = W_Generator(self.INPUT_NOISE, self.GENERATOR_FILTERS, self.NUMBER_CHANNELS, self.KERNEL_SIZE).to(self.DEVICE)
		self.D = W_Discriminator(self.DISCRIMINATOR_FILTERS, self.NUMBER_CHANNELS, self.KERNEL_SIZE).to(self.DEVICE)

		self.G.apply(weights_init)
		self.D.apply(weights_init)

		self.fixed_noise = torch.randn(64, self.INPUT_NOISE, 1, 1).to(self.DEVICE)
		self.big_noise = torch.randn(self.VECTOR_LEN, self.INPUT_NOISE, 1, 1).to(self.DEVICE)


		# criterion = nn.BCELoss() # TODO: WGAN loss

		### RMSprop for WGAN ###
		self.one = torch.FloatTensor([1])
		self.mone = self.one * -1
		self.one, self.mone = self.one.to(self.DEVICE), self.mone.to(self.DEVICE)
		self.G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.LR_G)
		self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.LR_D)

		self.len_data = len(self.train_dataset.data)
		self.log_when = self.len_data//self.BATCH_SIZE


	def train(self):
		img_list = []
		self.train_hist = {}
		self.train_hist['LR_D'], self.train_hist['LR_G'] = self.LR_D, self.LR_G
		self.train_hist['D_loss'] = []
		self.train_hist['G_loss'] = []
		self.train_hist['per_epoch_time'] = []
		self.train_hist['total_time'] = []
		self.train_hist['FIDs'] = []
		self.train_hist['FID_time'] = []

		self.G.train(), self.D.train()

		print('Training started!')
		start_time = time.time()
		for epoch in range(1, self.N_EPOCHS + 1):

			try:
				os.makedirs(f'{self.result_dir}/{str(datetime.datetime.now())[:10]}_fake_{epoch}_seed{torch.initial_seed()}')
				os.makedirs(f'{self.result_dir}/{str(datetime.datetime.now())[:10]}_real_{epoch}_seed{torch.initial_seed()}')
			except FileExistsError:
				pass

			gc.collect()

			epoch_start_time = time.time()
			for batch_idx, data in enumerate(self.train_loader):

				### Clip the D's gradient ###
				for p in self.D.parameters():
					p.data.clamp_(-self.C, self.C)

				### Update D ###
				self.D.zero_grad()

				x_real = data[0]
				x_real = x_real.to(self.DEVICE)

				D_real = self.D(x_real)
				D_real.backward(self.one)

				noise_z = torch.randn(self.BATCH_SIZE, self.INPUT_NOISE, 1, 1).to(self.DEVICE)
				x_fake = self.G(noise_z)

				D_fake = self.D(x_fake.detach())
				D_fake.backward(self.mone)
				D_loss = D_real - D_fake

				self.D_optimizer.step()

				### Update G ###
				self.G.zero_grad()

				noise_z = torch.randn(self.BATCH_SIZE, self.INPUT_NOISE, 1, 1).to(self.DEVICE)
				x_fake = self.G(noise_z)

				G_D_fake = self.D(x_fake)
				G_D_fake.backward(self.one)

				self.G_optimizer.step()

				self.train_hist['G_loss'].append(D_fake.item())
				self.train_hist['D_loss'].append(D_loss.item())

				if batch_idx != 0 and ((batch_idx + 1) % self.log_when//2) == 0:

					print(f'Epoch {epoch}/{self.N_EPOCHS} | Batch {batch_idx}/{len(self.train_loader)} '
					      f'Loss D: {D_loss.item():.4f} | Loss G: {D_fake.item():.4f}')

					with torch.no_grad():
						fake = self.G(self.fixed_noise).detach().cpu()

						img_grid_real = torchvision.utils.make_grid(x_real, normalize=True)
						img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

						# self.writer_real.add_image('Mnist Real Images', img_grid_real)
						torchvision.utils.save_image(img_grid_real,
						                             f'{self.result_dir}/{str(datetime.datetime.now())[:10]}_real_{epoch}_seed{self.SEED}/real_{epoch}_{batch_idx}.png')
						# self.writer_fake.add_image('Mnist Fake Images', img_grid_fake)
						torchvision.utils.save_image(img_grid_fake,
						                             f'{self.result_dir}/{str(datetime.datetime.now())[:10]}_fake_{epoch}_seed{self.SEED}/fake_{epoch}_{batch_idx}.png')
						plt.imshow(img_grid_fake.permute(1, 2, 0))
						img_list.append(fake)

				self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
				### Calculate FID ###
				if batch_idx != 0 and ((batch_idx + 1) % self.log_when == 0):

					### Disable Gradients ###
					with torch.no_grad():

						reals = []
						fakes = []

						big_fake = self.G(self.big_noise).detach().cpu()
						origs, _ = next(iter(self.real_loader))

						for fake, real in zip(big_fake, origs):
							data = fake.permute(1, 2, 0), real.permute(1, 2, 0)
							data = preprocess_image(data[0]), preprocess_image(data[1])
							fakes.append(data[0].unsqueeze(0)), reals.append(data[1].unsqueeze(0))

						real_vec, fake_vec = torch.cat(reals, dim=0), torch.cat(fakes, dim=0)

						rr = get_activations(real_vec, self.VECTOR_LEN, use_cuda=False)
						ff = get_activations(fake_vec, self.VECTOR_LEN, use_cuda=False)

						stat = w2_mod(rr, ff)

					# self.writer.add_scalar(f'norm of mean difference; seed {see} vector_len {VECTOR_LEN}', stat, global_step=epoch)
					print(f"FID at the end of epoch: {stat}")
					self.train_hist['FIDs'].append(stat)
		self.train_hist['total_time'].append(time.time() - start_time)
		print('Finished training!')

	def save(self):
		'''
		Function to save both model's learned parameters and training results

		'''
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + f'_G_{self.N_EPOCHS}_LR{self.LR_G}.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + f'_D_{self.N_EPOCHS}_LR{self.LR_D}.pkl'))

		with open(os.path.join(save_dir, self.model_name + f'_history_{self.N_EPOCHS}.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)