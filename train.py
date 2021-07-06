import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import torch
import warnings
import logging

# create logger
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from torchtext.data import Field, RawField
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

from utils import *
from model import *


if __name__ == "__main__":
	# if torch.cuda.is_available():
	#     device = torch.device('cuda:0')
	#     torch.backends.cudnn.deterministic = True
	#     torch.backends.cudnn.benchmark = False
	# else:
	#     device = torch.device('cpu')

	device = torch.device('cpu')
	print(device)
	data_path = 'data'

	# Initialize tokenizer.
	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

	# Set tokenizer hyperparameters.
	MAX_SEQ_LEN = 256
	BATCH_SIZE = 16
	PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
	UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
	path = f"{data_path}/prep_tweets.csv"

	# Define columns to read.
	label_field = Field(sequential=False,  use_vocab=False, batch_first=True)

	text_field = Field(use_vocab=False, tokenize=tokenizer.encode, include_lengths=False, batch_first=True,
	                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

	fields = {'tweet' : ('tweet', text_field),'label' : ('label', label_field)}


	dataset = TabularDataset(path= path, format='CSV', fields=fields, skip_header=False)
	train_data, valid_data, test_data = dataset.split(split_ratio=[0.70, 0.2, 0.1],  stratified=True, strata_field='label')


	# Create train and validation iterators.
	train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
	                                               batch_size=BATCH_SIZE,
	                                               device=device,
	                                               shuffle=True,
	                                               sort_key=lambda x: len(x.tweet), 
	                                               sort=True, 
	                                               sort_within_batch=False)

	# Test iterator, no shuffling or sorting required.
	test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)


	output_path = data_path





	# Main training loop
	NUM_EPOCHS_1 = 6
	steps_per_epoch = len(train_iter)
	model = ROBERTAClassifier(0.4)
	model = model.to(device)
	results = ResultsSaver(len(train_iter), len(valid_iter), output_path, device = device)


	print(" ............. Training the added Layers only ............. ")
	optimizer = AdamW(model.parameters(), lr=1e-4)
	scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=steps_per_epoch*1,  
	                                            num_training_steps=steps_per_epoch*NUM_EPOCHS_1)

	train(model=model, train_iter=train_iter, valid_iter=valid_iter, optimizer=optimizer, 
	      results = results, scheduler=scheduler, num_epochs=NUM_EPOCHS_1, train_whole_model = False, pad_index = PAD_INDEX)


	print(" ............. Training the whole Model ............. ")
	NUM_EPOCHS_2 = 6
	optimizer = AdamW(model.parameters(), lr=2e-6)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*2, 
	                                            num_training_steps=steps_per_epoch*NUM_EPOCHS_2)

	train(model=model,  train_iter=train_iter,  valid_iter=valid_iter,  optimizer=optimizer, 
	      results = results,  scheduler=scheduler,  num_epochs=NUM_EPOCHS_2, train_whole_model=True, pad_index = PAD_INDEX)

