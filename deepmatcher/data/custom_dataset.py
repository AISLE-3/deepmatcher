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
		print(f"supplied text columns : {', '.join(text_cols)}")
		print(f"supplied image column : {image_col}")
		print(f"supplied label column : {label_col}")

		self.label_col = label_col
		self.text_cols = text_cols
		self.image_col = image_col
		self.images_dir = images_dir
		self.image_size = image_size
		self.prefixes = prefixes

		self.tokenizer = tokenizer 

		self.use_text = isinstance(self.text_cols, list) and len(self.text_cols) > 0 and isinstance(tokenizer, Callable)
		self.use_image = isinstance(self.image_col, str) and self.image_col != ''

	def read_image(self, idx, col):
		fimg = os.path.join(self.images_dir, self.data_df.at[idx, col])
		img = cv2.cvtColor(cv2.imread(fimg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, self.image_size, interpolate=cv2.INTER_LINEAR)
		return img

	def __len__(self):
		return len(self.data_df)

	def __getitem__(self, idx):
		attrs_data = defaultdict(dict)
		if self.use_text:
			for col in self.text_cols:
				for prefix in self.prefixes:
					attrs_data[col][prefix] = self.data_df.at[idx, prefix + col]
		if self.use_image:
			for prefix in enumerate(self.prefixes):
				attrs_data[self.image_col][self.prefixes[i]] = self.read_image(idx, self.prefixes[i] + self.image_col)
		return {
			'attrs': attrs_data,
			'labels': int(self.data_df.at[idx, self.label_col])
		}

	def collate_fn(self, batch):
		batch_attrs = defaultdict(dict)
		batch_labels = torch.Tensor([sample['labels'] for sample in batch]).type(torch.long)

		if self.use_image:
			for prefix in self.prefixes:
				batch_attrs[self.image_col][prefix] = [sample['attrs'][self.image_col][prefix] for sample in batch]

		if self.use_text:
			for attr in self.text_cols:
				texts = [(batch_id, prefix, sample['attrs'][attr][prefix]) for batch_id, sample in enumerate(batch) for prefix in self.prefixes]
				tokenized = self.tokenizer([text[-1] for text in texts])
				slices = defaultdict(list)
				for i, (_, prefix, _) in enumerate(texts):
					slices[prefix].append(i)
				for prefix in self.prefixes:
					batch_attrs[attr][prefix] = {k : tokenized[k][slices[prefix]] for k in tokenized}
		return {
			"attrs" : dict(batch_attrs),
			"labels" : batch_labels
		}
