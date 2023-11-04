import sys
import os
from classes import Detoxificator, predict_for_one
from nltk.tokenize import WordPunctTokenizer as WPT
import torch
import pickle
import torch.nn as nn
import pandas as pd

# reproducibility
torch.cuda.manual_seed(1)

# reading file with the model; the serialized file contains the model, word->integer dict, and maximum sentence length
with open(os.path.join(os.path.dirname(__file__), f"../../{sys.argv[2]}"), "rb") as file:
	data = pickle.load(file)
	model, vocab_cat2idx, seq_length = data

# creates integer->word mapping
vocab_idx2cat = {v: k for k, v in vocab_cat2idx.items()}

# reading the sentences
sentences = list()
if sys.argv[3] == "terminal": # takes sentences from the arguments passed in the command line
	print("Reading from terminal")
	sentences.append(' '.join(sys.argv[4:]))

elif sys.argv[3] == "txt": # takes sentences from the txt file
	print("Reading from txt file")
	with open(os.path.join(os.path.dirname(__file__), "../../data/references.txt"), "r") as file:
		sentences = file.readlines()
	# in this case, all lines have a /n character at the end; remove it
	sentences = [x[:-1] for x in sentences]

elif sys.argv[3] == "dataframe": # takes sentences from pandas dataframe
	print("Reading from pandas dataframe")
	df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/references.tsv"), sep="\t")
	sentences = df.references.to_list()

else:
	print("Wrong parameters. Please read the documentation!!")

# Makes and writes predictions for each sentence separately
for sentence in sentences:
	output = predict_for_one(model, vocab_cat2idx, vocab_idx2cat, sentence, seq_length)
	print(f"\nInput:  {sentence}")
	print(f"Output: {output}")
