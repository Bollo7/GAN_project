import torch
from torch import nn
from torchvision.models import inception_v3
import numpy as np
from PIL import Image
import torch.nn.functional as F

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# Slightly modified implementation of PyTorch version of code based on original paper's implementation (source: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)
# All functions are kept the same, except of minor modifications in get_activations and preprocess_image functions.

def to_cuda(elements):
	"""
	Transfers elements to cuda if GPU is available
	Args:
		elements: torch.tensor or torch.nn.module
		--
	Returns:
		elements: same as input on GPU memory, if available
	"""
	if torch.cuda.is_available():
		return elements.to(device)
	return elements


class PartialInceptionNetwork(nn.Module):

	def __init__(self, transform_input=True):
		super().__init__()
		self.inception_network = inception_v3(pretrained=True)
		self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
		self.transform_input = transform_input

	def output_hook(self, module, input, output):
		# N x 2048 x 8 x 8
		self.mixed_7c_output = output

	def forward(self, x):
		"""
		Args:
			x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
		Returns:
			inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
		"""
		assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" + \
		                                     ", but got {}".format(x.shape)
		x = x * 2 - 1  # Normalize to [-1, 1]

		# Trigger output hook
		self.inception_network(x)

		# Output: N x 2048 x 1 x 1
		activations = self.mixed_7c_output
		activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
		activations = activations.view(x.shape[0], 2048)
		return activations


def get_activations(images, batch_size, use_cuda=True):
	"""
	Calculates activations for last pool layer for all CIFAR-10 images
	--
		Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
		batch size: batch size used for inception network
	--
	Returns: np array shape: (N, 2048), dtype: np.float32
	"""
	assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" + \
	                                          ", but got {}".format(images.shape)

	num_images = images.shape[0]
	inception_network = PartialInceptionNetwork()

	if use_cuda == True:
		inception_network = to_cuda(inception_network)
        
	inception_network.eval()
	n_batches = int(np.ceil(num_images / batch_size))
	inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
	for batch_idx in range(n_batches):
		start_idx = batch_size * batch_idx
		end_idx = batch_size * (batch_idx + 1)

		ims = images[start_idx:end_idx]

		if use_cuda == True:
			ims = to_cuda(ims)

		activations = inception_network(ims)
		activations = activations.detach().cpu().numpy()
		assert activations.shape == (ims.shape[0], 2048), "Expected output shape to be: {}, but was: {}".format(
			(ims.shape[0], 2048), activations.shape)
		inception_activations[start_idx:end_idx, :] = activations
	return inception_activations

def preprocess_image(im):
	"""Resizes and shifts the dynamic range of image to 0-1
	Args:
		im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
	Return:
		im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
	"""
	assert im.shape[2] == 3
	assert len(im.shape) == 3
	if im.dtype == np.uint8:
		im = im.astype(np.float32) / 255
	im = np.asarray(im, dtype=np.float32)
	im = (im - np.min(im)) / np.ptp(im)
	im = torch.from_numpy(im)
	im = torch.autograd.Variable(im, requires_grad=False)
	im = im.permute(2, 0, 1)
	im = im.unsqueeze(0)
	im = F.interpolate(im, 299, mode='bilinear')
	im = im.squeeze(0)
	assert im.max() <= 1.0
	assert im.min() >= 0.0
	assert im.dtype == torch.float32
	assert im.shape == (3, 299, 299)

	return im


def preprocess_images(images):

	"""Resizes and shifts the dynamic range of image to 0-1
	Args:
		images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
	Return:
		final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
	"""

	final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
	assert final_images.shape == (images.shape[0], 3, 299, 299)
	assert final_images.max() <= 1.0
	assert final_images.min() >= 0.0
	assert final_images.dtype == torch.float32
	return final_images
