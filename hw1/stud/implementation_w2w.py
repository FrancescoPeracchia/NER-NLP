from itertools import count
from shutil import register_unpack_format
import numpy as np
from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from model import Model
#from gensim.models import KeyedVectors
import csv
from pathlib import Path
import os
import torch.nn.functional as F
import numpy as np

#____________________________________________________________________________________
def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    model=StudentModel()
   
    return model
#____________________________________________________________________________________

#____________________________________________________________________________________

class StudentModel(Model):

    def __init__(self) -> None:
        super().__init__()

        self.index_map={'O': 0, 'B-LOC': 1, 'B-CW': 2, 'I-CW': 3, 'B-PER': 4, 'I-PER': 5, 'B-CORP': 6, 'I-CORP': 7, 'B-GRP': 8, 'I-GRP': 9, 'B-PROD': 10, 'I-PROD': 11, 'I-LOC': 12}
        self.hard_const={'O': [], 'B-LOC': [], 'B-CW': [], 'I-CW': ['B-CW','I-CW'], 'B-PER': [], 'I-PER': ['B-PER','I-PER'], 'B-CORP': [], 'I-CORP': ['B-CORP','I-CORP'], 'B-GRP': [], 'I-GRP': ['B-GRP','I-GRP'], 'B-PROD': [], 'I-PROD': ['B-PROD','I-PROD'], 'I-LOC': ['B-LOC','I-LOC']}
        self.vocabulary ,self.index_map = generate_vocabulary()
 
        

        #LOAD
        self.Dictionary = np.load('hw1/data/Dict_W2W.npy',allow_pickle='TRUE').item()
        weight = np.load('hw1/data/Weight_W2W.npy',allow_pickle='TRUE')
        self.weight_tensor=torch.from_numpy(weight)
  
        
    

        print(self.weight_tensor.size())
        embedding_dim=300
        hidden_dim=300

        tagset_size=len(self.index_map)

        self.Embedder = Embedding(self.Dictionary,self.index_map)
        self.Ge = W2W_Embedding(self.weight_tensor,self.Dictionary,True,True)
        self.model = LSTMTagger(embedding_dim, hidden_dim, tagset_size)

        #RESUME MODEL
        print(self.model)

        #Search
        self.search_model=Search(self.index_map,self.hard_const)
        
        #load weights on LSTMTtagger and select INFERENCE MODE
        self.PATH = 'model/model_w2w.pt'
 
        try : self.model.load_state_dict(torch.load(self.PATH))
        except :print('error loading')

        #model in inference mode
        self.model.eval()



    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        ''''
        
        for t in token 

        each sentece divided
        
        '''

        sentence_indexed,lenght_padding=self.Embedder.from_token_to_index_embedding(tokens)
        sentence_indexed = self.Ge.embedd(sentence_indexed.tolist())
        output_network=self.model.forward(sentence_indexed)



        

        #ALTERNATIVE WITH BELOW         
        out=torch.argmax(output_network, dim=2)
        out_without_padding = self.Embedder.remove_padding(out,lenght_padding)
        output =self.Embedder.from_index_to_label(out_without_padding)

          
        #ALTERNATIVE WITH SEARCHING 
        '''
        search_output=self.search_model.explore(output_network)
        out_without_padding = self.Embedder.remove_padding_list(search_output,lenght_padding)
        output =self.Embedder.from_index_to_label(out_without_padding)

        '''

        return output       

class Embedding():


    def __init__(self,vocabulary,mapping) -> None:

        self.UNK_TOKEN = "<UNK>"
        self.PAD_TOKEN = "<PAD>"
        self.vocabulary=vocabulary
        self.index_inv_map = {v: k for k, v in mapping.items()}


    def from_token_to_index_embedding(self,x):
        list_=[]
        #print('TOKEN SENTENCE :',x)

        for sentence in x:
            list_.append(len(sentence))

        x = [[self.vocabulary.get(word, self.vocabulary[self.UNK_TOKEN]) for word in sentence] for sentence in x]
        
        #print('TOKENIZED : ',x)
        
        
       

        x = pad_sequence(
                [torch.as_tensor(sample) for sample in x],
                batch_first=True,
                padding_value=self.vocabulary.get(self.PAD_TOKEN))

        #print('Padded : ',x)
        return x,list_


    def from_index_to_label(self,x):

        x = [[self.index_inv_map.get(index) for index in sentence] for sentence in x]
        return x
    
    def remove_padding(self, input , list):

        list_out=[]
       

        for i,max in enumerate(list):
            list_out.append(input[i,:max].tolist())
    
        return list_out
    
    def remove_padding_list(self, input , list):

        list_out=[]
        #print('input:',input)
        

        for i,max in enumerate(list):
            in_ = input[i]  
            #print(in_)
            list_out.append(in_[:max])
    
        return list_out

class W2W_Embedding():

    def __init__(self, weights, dict, batch, by_id ) -> None:

        self.weights = weights
        self.dict = dict
        self.batch = batch
        self.weights_leng = len(self.weights[0,:])
        self.by_id = by_id
    

    def get_weight(self,token):


        if(self.by_id == False):
            try : index_token = self.dict[token]
            except : 
                #print('missing')
                index_token = self.dict['UNK']
        else:
            return self.weights[token,:]

        
    
        return self.weights[index_token,:]


    def embedd(self,list_words):

        list=[]
        
        if(self.batch == False):

            for word in list_words:
                list.append(self.get_weight(word).tolist())
            

            

            return torch.tensor(list)
        
        elif(self.batch == True):

            max_pad = longest(list_words)
            #print('max padd',max_pad)

            

            for sentence in list_words:

                current_lenght = len(sentence)
                #print('current :',current_lenght)
                sencence=[]
                for word in sentence:
                    sencence.append(self.get_weight(word).tolist())
                
                

                if(max_pad>current_lenght):
                    #print('padding')
                    listofzeros = [0] * self.weights_leng

                    while max_pad>current_lenght:
                        #print('padding now')
                        sencence.append(listofzeros)
                        #print('new leng',len(sencence))
                        current_lenght = len(sencence)
                

                #print('append list',len(sencence))
                list.append(sencence)


            return torch.tensor(list)


    def embedd(self,list_words):

        list=[]
        
        if(self.batch == False):

            for word in list_words:
                list.append(self.get_weight(word).tolist())
            

            

            return torch.tensor(list)
        
        elif(self.batch == True):

            max_pad = longest(list_words)
            #print('max padd',max_pad)

            

            for sentence in list_words:

                current_lenght = len(sentence)
                #print('current :',current_lenght)
                sencence=[]
                for word in sentence:
                    sencence.append(self.get_weight(word).tolist())
                
                

                if(max_pad>current_lenght):
                    #print('padding')
                    listofzeros = [0] * self.weights_leng

                    while max_pad>current_lenght:
                        #print('padding now')
                        sencence.append(listofzeros)
                        #print('new leng',len(sencence))
                        current_lenght = len(sencence)
                

                #print('append list',len(sencence))
                list.append(sencence)


            return torch.tensor(list)

class Search():


    def __init__(self,mapping,hard_constraints) -> None:

        self.hard_constraints = hard_constraints
        self.mapping=mapping
        self.index_inv_map = {v: k for k, v in mapping.items()}

    

    def explore(self, batch):
        
        List_ = []

        cond = torch.zeros(13, dtype=torch.bool)
        cond = ~cond
        index_map={'O': 0, 'B-LOC': 1, 'B-CW': 2, 'I-CW': 3, 'B-PER': 4, 'I-PER': 5, 'B-CORP': 6, 'I-CORP': 7, 'B-GRP': 8, 'I-GRP': 9, 'B-PROD': 10, 'I-PROD': 11, 'I-LOC': 12}
        hard_const={'O': [], 'B-LOC': [], 'B-CW': [], 'I-CW': ['B-CW','I-CW'], 'B-PER': [], 'I-PER': ['B-PER','I-PER'], 'B-CORP': [], 'I-CORP': ['B-CORP','I-CORP'], 'B-GRP': [], 'I-GRP': ['B-GRP','I-GRP'], 'B-PROD': [], 'I-PROD': ['B-PROD','I-PROD'], 'I-LOC': ['B-LOC','I-LOC']}
        index_inv_map = {v: k for k, v in index_map.items()}

        #print('BATCH SIZE',batch.size())
                
    
    
        print('batch SIZE',batch.size())
        
        for x in batch:

            

            initial_lenght = len(x)
            print(x.size())
            
          

            flipped=torch.flip(x,[0])
            #print(flipped)
            list_max,list_,counter,list_index,list_max_index=recursive_call(flipped,0,flipped.size()[0],0.008,[],[],[],[],cond,0,index_map,index_inv_map,hard_const)
            list_max_index=list_max_index[::-1]

            if len(list_max_index) != initial_lenght:
                
                
                list_max_index=torch.argmax(x, dim=1).tolist()
                






            List_.append(list_max_index)
            

            #print('new sentence : ', List_ )
            




        return  List_

class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()

        #self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        
        #if bidirectional false
        #self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        #self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True,num_layers=2,bidirectional=True,dropout=0.3)
        self.hidden2tag = torch.nn.Linear(2*hidden_dim, hidden_dim)
        self.drop = torch.nn.Dropout(p=0.3)
        self.hidden3tag = torch.nn.Linear(hidden_dim, tagset_size)
        self.soft_max = torch.nn.Softmax(dim=2)


    def forward(self, sentence):


        #embeds = self.word_embeddings(sentence)
        #print('embeds size',embeds.size())
  
        lstm_out, _ = self.lstm(sentence)
        #print('lstm_out size',lstm_out.size())

        tag_space = self.hidden2tag(lstm_out)
        tag_space = self.drop(tag_space)
        tag_space = self.hidden3tag(tag_space)
        

        #print('tag_space size',tag_space.size())

        tag_scores=self.soft_max(tag_space)
        #print('tag_scores size',tag_scores.size())



        return tag_scores



       
#____________________________________________________________________________________





#________________UTILS______________________________________________________________
#____________________________________________________________________________________

def heuristic(list_):
    
    if(len(list_)==0):
        product=0
    else:
        product=1
        for i in list_:
            product=product*i
        #print(product)
    
    
    return product

def tag_conditioning(tag_index,mapping,inv_mapping,hard_constraints):
    cond = torch.zeros(13, dtype=torch.bool)
    label=inv_mapping[tag_index]
    list_constraints=hard_constraints[label]
    for costraint in list_constraints:
        cond[mapping[costraint]]=True
    
    if len(list_constraints)==0:
        cond = ~cond
    return cond
   
def recursive_call(sentence_inv,current_depth,depth,threshold,list,list_index,list_max,list_max_index,cond,counter,index_map,index_inv_map,hard_const):
    #print('Level :',current_depth)
    #print('Total :',depth)
    counter+=1
    #print('Ricorsioni',counter)
    


   #for all possible tag
    if(current_depth==depth):
        
        value=heuristic(list)
        value_max=heuristic(list_max)
        #print('Values:',value,' and ',value_max)
        if(value>value_max):
            #print(list_index)
            list_max_index=list_index
            list_max=list
        else:
            #print('Old')
            pass
        #print('\n______________________________________________________\n')
        #print('current tested list ',list)
        #print('Index of list max : ',list_max)
        #print('current ID tested list ',list_index)
        #print('Index of ID list max : ',list_max_index)
        #print('\n______________________________________________________\n')

        return list_max,list,counter,list_index,list_max_index
    
    
    for index_tag, tag in enumerate(sentence_inv[0]):

        list.append(float(tag))
        value=heuristic(list)
        value_max=heuristic(list_max)
        
        #print('last hy values added',tag,' TOTAL hy :',list)
        #print('values skip',value,'values skip max',value_max )
        list=list[:len(list)-1]
        #print(cond)


        

        if(tag>threshold and value>=value_max and cond[index_tag]==True) :
            list.append(float(tag))
            list_index.append(index_tag)
            #print('added element',tag)
            #print('list VALUES are : ',list)
            #print('list TAGS are : ', list_index)

            #conditions change on tag choice
            cond=tag_conditioning(index_tag,index_map,index_inv_map,hard_const)
            #print(index_tag)
            




            list_max,list,counter,list_index,list_max_index=recursive_call(sentence_inv[1:,:],current_depth+1,depth,threshold,list,list_index,list_max,list_max_index,cond,counter,index_map,index_inv_map,hard_const)
            cond=tag_conditioning(index_tag,index_map,index_inv_map,hard_const)
            list=list[:len(list)-1]
            list_index=list_index[:len(list_index)-1]
            #print('Index of list max : ',list_max_index)
            #print('current list max',list_max)
        else:
            #print('non added element',tag)
            pass
 
    

    #print('Back to previous level')
    return list_max,list,counter,list_index,list_max_index

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

def generate_vocabulary():

    print('current dir is',)
    [print(x) for x in os.walk(os.getcwd())]
    



    data_voc="hw1/data/train.tsv"



    data_train=dataset_creation(data_voc)
    vocab,label_dict=vocabulary_and_lableDictionary(data_train)
    #print(vocab['This'])
    print('\nVocabulary Created\n')

    return vocab,label_dict

#____________________________________________________________________________________


def generate_glove_voc():
    path="hw1/Glove/glove.6B.200d.txt"
    return  load_GLOVE(path)


def load_GLOVE(PATH):

    Dictionary = {}
    weights = []

    count=0
    
    with open(PATH, 'r', encoding="utf-8") as f:

       
        content = f.readlines()
       
        
        for row in content:

            vector = []
            
            
            s = row.split(' ')
            
            Dictionary[s[0]] = count
            count +=1
            values = s[1:]
            for v in values:
                v = v.rstrip()
                #print(type(float(v)))

                vector.append(float(v))
                #print(type(vector[0]))
                #print(count)




            
            
            weights.append(vector)

                
                
                

            #count+=1
            #Dictionary[line]
     
    

    return torch.tensor(weights),Dictionary
    #print(weight_tensor.size())
      
def longest(list1):
    longest_list = max(len(elem) for elem in list1)
    return longest_list
