
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

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
		if not trainable:
			for param in self.model.parameters():
				param.requires_grad = False

	def mean_pooling(self, token_embeddings, attention_mask):
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


	def tokenizer(self, texts):
		texts = [str(text) for text in texts]
		return self._tokenizer(texts, return_tensors='pt', truncation=self.truncation, padding=self.padding, max_length=self.max_length)
	
	def forward(self, inputs):
		outputs = self.model(**inputs).last_hidden_state
		outputs = self.mean_pooling(outputs, inputs['attention_mask'])
		return outputs
