# %%
import cv2
import numpy as np
import random
import os
import time

import pandas as pd
from collections import defaultdict

import torch

from torch.utils.data import Dataset, DataLoader
from typing import Callable

from deepmatcher.batch import AttrTensor
# %%
class DeepMatcherDataset(Dataset):
	def __init__(
			self,
			data_df: pd.DataFrame,
			label_col: str,
			text_cols: list = None,
			image_col: str = None,
			images_dir: str = '',
			tokenizer: Callable = None,
			image_size=(256, 256),
			prefixes=('left_', 'right_')
		):
		super().__init__()

		self.data_df = data_df
		print(f"received data_df : {self.data_df.shape}")
		self.data_df.info()
		print(f"supplied text columns : {', '.join(text_cols)}")
		print(f"supplied image column : {image_col}")
		print(f"supplied label column : {label_col}")

		self.label_col = label_col
		self.text_cols = text_cols
		self.image_col = image_col
		self.images_dir = images_dir
		self.image_size = image_size
		self.tokenizer = tokenizer
		self.prefixes = prefixes

		self.canonical_text_fields = self.text_cols
		self.all_text_fields = [prefix + col for col in self.text_cols for prefix in self.prefixes]

		self.use_text = isinstance(self.text_cols, list) and len(self.text_cols) > 0 and isinstance(self.tokenizer, Callable)
		self.use_image = isinstance(self.image_col, str) and self.image_col != ''

	def _assert_data(self):
		self._assert_cols()

	def _assert_cols(self):
		if self.use_text:
			for col in self.text_cols:
				left_col, right_col = f"left_{col}", f"right_{col}"
				assert left_col in self.data_df, f"{left_col} was not found in the dataframe"
				assert right_col in self.data_df, f"{right_col} was not found in the dataframe"
		if self.use_image:
			for col in (f"left_{self.image_col}", f"right_{self.image_col}"):
				assert col in self.data_df, f"{col} was not found in the dataframe"

	def __len__(self):
		return len(self.data_df)

	def __getitem__(self, idx):
		attrs_data = defaultdict(dict)
		if self.use_text:
			texts = [(col, prefix, self.data_df.at[idx, prefix + col]) for col in self.text_cols for prefix in self.prefixes]
			tokenized = self.tokenizer([t[-1] for t in texts])
			for i, (col, prefix, _) in enumerate(texts):
				attrs_data[col][prefix] = {k: tokenized[k][i] for k in tokenized}

		if self.use_image:
			for i, prefix in enumerate(self.prefixes):
				fimg = os.path.join(self.images_dir, self.data_df.at[idx, self.prefixes[i]+self.image_col])
				img = cv2.cvtColor(cv2.imread(fimg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, self.image_size, interpolate=cv2.INTER_LINEAR)
				attrs_data[self.image_col][self.prefixes[i]] = img
		return {
			'attrs': attrs_data,
			'labels': int(self.data_df.at[idx, self.label_col])
		}
	
	def wrap_tensors_into_attr_tensor(self, pt_tensor):
		return AttrTensor(pt_tensor, pt_tensor.lengths, None, None)
# %%

