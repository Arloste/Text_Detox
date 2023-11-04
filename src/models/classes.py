# all imports
import pandas as pd
from nltk.tokenize import WordPunctTokenizer as WPT
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# reproducibility
torch.cuda.manual_seed(1)


class DetoxDataset(torch.utils.data.Dataset):
	"""
	This class creates a Pytorch Dataset from a given Pandas Dataframe
	
	Parameters:
	dataframe (pandas.DataFrame): the initial dataset with information
	max_size (int): maximum length of a sentence in the dataset; each sentence will be padded until this length
	pad_idx (int): index of padding; it should be taken from vocabulary
	
	Returns:
	torch.utils.data.Dataset: the dataset of input sentences, padded to max_size
	"""
	def __init__(self, dataframe, max_size, pad_idx):
		self.dataframe = dataframe
		self.max_size = max_size
		self.pad_idx = pad_idx
		self._preprocess()

	def _preprocess(self):
        # transform words into integers using vocab here
		self.references = list()
		self.translations = list()

		pbar = tqdm(total=self.dataframe.shape[0])
		for idx, row in self.dataframe.iterrows():
			# splitting the sentences into list of words and padding them to max length
			x_row = [int(x) for x in row.reference.split()]
			x_row += [self.pad_idx] * (self.max_size - len(x_row))
			y_row = [int(x) for x in row.translation.split()]
			y_row += [self.pad_idx] * (self.max_size - len(y_row))

			self.references.append(x_row)
			self.translations.append(y_row)
			pbar.update(1)

	def _get_sentence(self, index):
		return self.references[index]

	def _get_labels(self, index):
		return self.translations[index]

	def __getitem__(self, index):
		return self._get_sentence(index), self._get_labels(index)

	def __len__(self):
		return len(self.references)


def collate_batch(batch):
	"""
	This is a helper function for torch Dataloader
	"""
	sentences_batch = list()
	translations_batch = list()
	for x, y in batch:
		sentences_batch.append(x)
		translations_batch.append(y)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	# casting the lists to tensors and shaping them so they have dimensions (max_size, batch_size)
	sentences_batch = torch.as_tensor(sentences_batch)
	sentences_batch = torch.transpose(sentences_batch, -2, 1).to(device)
	translations_batch = torch.as_tensor(translations_batch)
	translations_batch = torch.transpose(translations_batch, -2, 1).to(device)
	return sentences_batch, translations_batch


class Detoxificator(nn.Module):
	"""
	This class defines the neural network architecture.
	It consists of one LSTM module and a fully connected layer
	"""
	def __init__(self, inp_dim, embed_dim, out_dim, pad_idx):
		super().__init__()

		self.embedding = nn.Embedding(inp_dim, embed_dim, padding_idx=pad_idx)
		self.lstm = nn.LSTM(embed_dim, 512, num_layers=1)
		self.out = nn.Linear(512, out_dim)

	def forward(self, x):
		x = self.embedding(x)
		x, (hidden, cell) = self.lstm(x)
		x = self.out(x)
		return x


def train(model, loader, optimizer, pad_idx, loss_fn1, loss_fn2, epochs):
	"""
	This is a training function for NN.
	
	Parameters:
	model (torch nn.Module): an instance of the Detoxification NN
	loader (torch Dataloader): torch dataset
	optimizer (torch.optim): torch optimizer
	pad_idx (idx): index of padding
	loss_fn1 (torch.nn): torch loss function that does not ignore padding values
	loss_fn2 (torch.nn): torch loss function that ignores padding values
	epochs (int): number of epochs
	"""
	model.train()
	for epoch in range(1, epochs+1):
		pbar = tqdm(loader)
		pbar.set_description_str(f"[{epoch}/{epochs}]")
		for batch in pbar:
			texts, labels = batch
			# zeroing the gradient
			optimizer.zero_grad()
			
			# making predictions
			outputs = model(texts)
			
			# reshaping the labels and outputs so i can calculate the loss
			outputs = outputs.view(-1, outputs.shape[-1])
			labels = labels.reshape(-1)
			
			# if the padding is correct, the loss is calculated only on the meaningful part of the sentence
			# with such approach we use computational resources not for learning the padding, but actual sentences
			idx = [labels==pad_idx]
			pad_accuracy = sum([x==y for x, y in zip(labels[idx], outputs.argmax(dim=1)[idx])]).item()*100/len(labels[idx])
			loss = loss_fn1(outputs, labels) if pad_accuracy<0.999 else loss_fn2(outputs, labels)
			
			loss.backward()
			optimizer.step()
			
			# calculates the accuracy of predictions on the meaningful part of the sentence
			# and shows this info in the tqdm progress bar
			idx = [labels!=pad_idx]
			accuracy = sum([x==y for x, y in zip(labels[idx], outputs.argmax(dim=1)[idx])]).item()*100/len(labels[idx])
			pbar.set_postfix({"loss": loss.item(), "accuracy":f"{round(accuracy, 3)}%"})


def predict_for_one(model, cat2idx, idx2cat, sentence, seq_length):
	"""
	This function takes a model and a sentence and makes inference on the sentence
	
	Parameters:
	model (torch.Module): an instance of a Detoxification NN
	cat2idx (dict{"str"->"int"}): a dictionary that maps words to integers
	idx2cat (dict{"int"->"str"}): a dictionary that maps integers to words
	sentence (str): the sentence to make an inference from
	seq_length (int): maximum sentence length; required for padding
	
	Returns:
	output (str): the result of model inference as a sentence
	"""
	# tokenizes and pads the sentence, casts it into a tensor
	tokenized_sentence = WPT().tokenize(sentence)
	encoded_sentence = [cat2idx.get(word, cat2idx["<UNK>"]) for word in tokenized_sentence]
	encoded_sentence += [cat2idx["<PAD>"]] * (seq_length - len(encoded_sentence))
	inp = torch.as_tensor(encoded_sentence).view(-1, 1)
	
	with torch.no_grad():
		model.eval()
		output = model(inp)
		output = output.argmax(dim=2).flatten()
		# transforms the list of predicted values back into words, removes padding
		output = ' '.join(([idx2cat[x.item()] for x in output])).replace(" <PAD>", "")
	
	return output
