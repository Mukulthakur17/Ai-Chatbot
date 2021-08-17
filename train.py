import json
from utils import * 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open ('intents.json', 'r') as f:
    intents = json.load(f) 

all_words = []
tags = []
labeled = []     # it's a list of tuples that will store the words with their tags

# Tokenizing the corpora and storing it
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # same tokenize fxn that we made in utils
        w = tokenize(pattern)
        all_words.extend(w)
        labels.append((w, tag))

# stemming all the words and ignoring punctuations
ignore_words = ['?', '!', ',', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # To sort & remove the duplicates

# Creating the training data
X_train = []
y_train = []

for (pattern_sentence, tag) in labeled:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

#Converting the lists into numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)


# Converting our dataset into a PyTorch dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
#print(input_size, output_size)


# Dataloader has been used to parallelize the data loading as this boosts up the speed and saves memory.
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

