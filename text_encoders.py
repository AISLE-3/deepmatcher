#%%
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
#%%
class HFTextEncoder(nn.Module):
	def __init__(self, identifier : str, trainable : bool = False, seqtovec : str = 'cls', max_length=64, padding='max_length', truncation=True):
		super().__init__()
		self.model_identifier = identifier
		self.seqtovec = seqtovec
		self.cls_token_idx = 0
		self._tokenizer = AutoTokenizer.from_pretrained(self.model_identifier)
		self.model = AutoModel.from_pretrained(self.model_identifier)
		self.embedding_dim = 768 #TODO: get from AutoModel
		self.max_length = max_length
		self.padding = padding
		self.truncation = truncation

	def tokenizer(self, texts):
		texts = [str(text) for text in texts]
		return self._tokenizer(texts, return_tensors='pt', truncation=self.truncation, padding=self.padding, max_length=self.max_length)
	
	def forward(self, inputs):
		outputs = self.model(**inputs).last_hidden_state
		if self.seqtovec == 'cls':
			return outputs[:, self.cls_token_idx, :]
		else:
			return outputs

# from sentence_transformers import SentenceTransformer
# class STTextEncoder(nn.Module):
# 	def __init__(self, identifier : str):
# 		super().__init__()
# 		self.model_identifier = identifier
# 		self.model = SentenceTransformer(self.model_identifier)
# 		self.embedding_dim = 768 # TODO: get from SentenceTransformer
	
# 	def forward(self, texts):
# 		return self.model.forward(texts)