import csv
from typing import Dict, Iterator, List, Union, Optional
from torch.nn.utils.rnn import pad_sequence
import torch
from gensim.models import KeyedVectors
import numpy as np
'''
Input : path to dataset
Output : List of dictionaries, each dictionary is a Id Sentence composed by "sentence" and "labels" field
'''

def dataset_creation(path):
  

  dataset = [] 
  words = []
  labels = []

  with open(path) as file:
      tsv_file = csv.reader(file, delimiter="\t")
      for line in tsv_file:
          
          #print(line)


          if line:


              if line[0] == '#':
                  #print('new line------------------------------')

                  new_sentence={
                      "sentence":words,
                      "labels":labels
                  }
                  
                  dataset.append(new_sentence)
                                  
                  words=[]
                  labels=[]
              else:
                  word=line[0]
                  label=line[1]
                  words.append(word)
                  labels.append(label)

          
  dataset.pop(0)
  return dataset



def vocabulary_and_lableDictionary(data):
    

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # vocabulary from word to index
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    # iterate over the words of the dataset
    for word in [word for sample in data for word in sample["sentence"]]:
        if word not in vocab:
            vocab[word] = len(vocab)

    label_dict = {}
    for label in [label for sample in data for label in sample["labels"]]:
        if label not in label_dict:
            label_dict[label] = len(label_dict)

    #print("Word to Index:")
    #print(vocab)

    #print("Label to Index:")
    #print(label_dict)

    return vocab, label_dict




def load_torch_embedding_layer(weights: KeyedVectors, padding_idx: int = 0, freeze: bool = False):

  vectors = weights.vectors
  # random vector for pad
  pad = np.random.rand(1, vectors.shape[1])
  print(pad.shape)
  # mean vector for unknowns
  unk = np.mean(vectors, axis=0, keepdims=True)
  print(unk.shape)
  # concatenate pad and unk vectors on top of pre-trained weights
  vectors = np.concatenate((pad, unk, vectors))
  # convert to pytorch tensor
  vectors = torch.FloatTensor(vectors)
  # and return the embedding layer
  return torch.nn.Embedding.from_pretrained(vectors, padding_idx=padding_idx, freeze=freeze)




  