# All necessary imports
from classes import DetoxDataset, collate_batch, Detoxificator, train
import pandas as pd
from nltk.tokenize import WordPunctTokenizer as WPT
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import os
import sys

# reproducibility
torch.cuda.manual_seed(1)

# reads the dataset of references and translations
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/raw.tsv"), sep="\t")
data = data[:100]

vocab = set(["<PAD>", "<UNK>"])
# stores the maximum length of one sentence
max_seq_length = 0

print("Scanning the vocabulary...")
pbar = tqdm(total=data.shape[0])

# tokenizes each sentence from the dataset and creates the vocab
for i, row in data.iterrows():
    ref, trn = row.reference, row.translation

    # Some of the rows have swapped 'reference' and 'translation'. Fix this
    if row.ref_tox < row.trn_tox:
        ref, trn = trn, ref
        data.loc[i, "ref_tox"], data.loc[i, "trn_tox"] = row.trn_tox, row.ref_tox
    
    ref = WPT().tokenize(ref.lower())
    trn = WPT().tokenize(trn.lower())
    [vocab.add(word) for word in ref+trn]
    # updates the max length
    max_seq_length = max(max_seq_length, len(ref))
    max_seq_length = max(max_seq_length, len(trn))

    data.loc[i, "reference"] = ' '.join(ref)
    data.loc[i, "translation"] = ' '.join(trn)

    pbar.update(1)

# creates the {str->int} and {int->str} mappings from the vocabulary
vocab_cat2idx = {word: i for i, word in enumerate(vocab)}
vocab_idx2cat = {v: k for k, v in vocab_cat2idx.items()}
VOCAB_SIZE = len(vocab)
PAD_IDX = vocab_cat2idx["<PAD>"]
print(f"Vocabulary size: {VOCAB_SIZE}")

print("\nTokenizing elements...")
pbar = tqdm(total=data.shape[0])
for i, row in data.iterrows():
    # tokenizes the senteces using the mapping
    ref = [str(vocab_cat2idx[i]) for i in row.reference.split()]
    trn = [str(vocab_cat2idx[i]) for i in row.translation.split()]
    
    data.loc[i, "reference"] = ' '.join(ref)
    data.loc[i, "translation"] = ' '.join(trn)

    pbar.update(1)

# creates the dataset
print("\nCreating dataset...")
dataset = DetoxDataset(data, max_seq_length, PAD_IDX)


batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# creates the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)

# defines the parameters for NN training
INPUT_DIM = VOCAB_SIZE
OUTPUT_DIM = VOCAB_SIZE
EMBED_DIM = 32
model = Detoxificator(INPUT_DIM, EMBED_DIM, OUTPUT_DIM, PAD_IDX).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn1 = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
loss_fn2 = nn.CrossEntropyLoss()

# trains the model
print("\nTraining the model...")
train(model, dataloader, optimizer, PAD_IDX, loss_fn1, loss_fn2, 10)

# saves the model
print("\nSaving the model...")

with open(os.path.join(os.path.dirname(__file__), f"../../{sys.argv[2]}"), 'wb') as file:
	pickle.dump([model, vocab_cat2idx, max_seq_length], file)

print(f'Model "{sys.argv[2]}" trained and saved successfully !!')
