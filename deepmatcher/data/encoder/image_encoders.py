#%%
import torch
import numpy as np
from PIL import Image

from pmcore.core.feature_extraction import DINOExtractor
# %%
class DINOImageEncoder():
	def __init__(self, device=None):
		self.model = DINOExtractor(patch_size=8, cuda=device == 'cuda')
		self.embedding_dim = 768
	
	def encode(self, images):
		if isinstance(images[0], str):
			out = torch.Tensor(np.vstack([self.model.from_file(img)['feature_vector'] for img in images]))
		elif isinstance(images[0], np.ndarray):
			out = torch.Tensor(np.vstack([self.model.from_binary(Image.fromarray(img))['feature_vector'] for img in images]))
		return out.to(torch.device(self.model.device or 'cpu'))
	
	def __call__(self, images):
		return self.encode(images)