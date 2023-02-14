"""
Script for training, testing, and saving finetuned, binary classification models based on pretrained
BERT parameters, for the IMDB dataset.
"""

import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from utils.imdb import IMDB
from utils.model_new import distillation
from models.xlnet import XLNet
# Disable unwanted warning messages from pytorch_transformers
# NOTE: Run once without the line below to check if anything is wrong, here we target to eliminate
# the message "Token indices sequence length is longer than the specified maximum sequence length"
# since we already take care of it within the tokenize() function through fixing sequence length
logging.getLogger('pytorch_transformers').setLevel(logging.CRITICAL)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Define hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32

PRETRAINED_MODEL_NAME = 'bert-base-cased'
NUM_PRETRAINED_BERT_LAYERS = 4
MAX_TOKENIZATION_LENGTH = 512
NUM_CLASSES = 2
TOP_DOWN = True
NUM_RECURRENT_LAYERS = 0
HIDDEN_SIZE = 128
REINITIALIZE_POOLER_PARAMETERS = False
USE_BIDIRECTIONAL = False
DROPOUT_RATE = 0.20
AGGREGATE_ON_CLS_TOKEN = True
CONCATENATE_HIDDEN_STATES = False

APPLY_CLEANING = False
TRUNCATION_METHOD = 'head-only'
NUM_WORKERS = 0

BERT_LEARNING_RATE = 3e-5
CUSTOM_LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
BERT_WEIGHT_DECAY = 0.01
EPS = 1e-8

# Initialize to-be-finetuned Bert model
model = XLNet()

# Initialize train & test datasets
train_dataset = IMDB(input_directory='/data/jc/data/sentiment/IMDB_hapi/aclImdb/test',hapi_info='sa/imdb/amazon_sa/22-05-23')

test_dataset = IMDB(input_directory='/data/jc/data/sentiment/IMDB_hapi/aclImdb/train',hapi_info='sa/imdb/amazon_sa/22-05-23')

# Acquire iterators through data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,drop_last=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)



# Define optimizer
optimizer = AdamW(model.parameters(),
                  lr=BERT_LEARNING_RATE,
                  betas=BETAS,
                  eps=EPS,
                  weight_decay=BERT_WEIGHT_DECAY)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=NUM_EPOCHS)
 
distillation(module=model,num_classes=NUM_CLASSES,
             epochs=NUM_EPOCHS,optimizer=optimizer,
             lr_scheduler=lr_scheduler, 
             loader_train=train_loader,loader_valid=test_loader)
