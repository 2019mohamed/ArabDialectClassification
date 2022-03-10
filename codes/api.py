# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:21:00 2022

@author: M
"""

from flask import Flask , request , jsonify
from  preprocessing import clean_text
import pickle
import numpy as np
from scipy.sparse import coo_matrix
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler , BatchSampler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.linear_model import PassiveAggressiveClassifier


labels = {'IQ': 0, 'LY': 1, 'QA': 2, 'PL': 3, 'SY': 4
          , 'TN': 5, 'JO': 6, 'MA': 7, 'SA': 8, 'YE': 9, 'DZ': 10, 'EG': 11,
          'LB': 12, 'KW': 13, 'OM': 14, 'SD': 15, 'AE': 16, 'BH': 17}
langs = {v:k for k,v in labels.items()}


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            #for layer in range(num_layers - 1):
             #   self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.linears[i](h))
            return self.linears[-1](h)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        #m.bias.data.fill_(0.01)

def load_embedding ():
    vec = pickle.load(open('C:\\Users\\M\\Downloads\\vectorizer.pkl', 'rb'))
    return vec

def get_PA ():
    PA = pickle.load(open('C:\\Users\\M\\Downloads\\PA.pkl', 'rb'))
    return PA

def get_MLP_Classifier (dim_input):
    model = MLP(2,dim_input,1000,18)
    model.load_state_dict(torch.load('C:\\Users\\M\\Downloads\\MLP4.pth'))
    model.eval()
    return model
    

def reduce (copus , vec):
    X = vec.transform(copus)
    X = coo_matrix(X)
    tensor_X = torch.sparse_coo_tensor([X.row , X.col],X.data , dtype = torch.float)
    reducer = MLP(1,tensor_X.shape[1],1000,10000)
    reducer.apply(init_weights)
    Z = reducer(tensor_X)
    Z = Z.detach().numpy()
    return Z


#_, predicted = torch.max(output.data, 1)



def get_important_index ():
    ii = pickle.load(open('C:\\Users\\M\\Downloads\\important_indexs.pkl', 'rb'))
    return ii


'''
RuntimeError: [enforce fail at ..\c10\core\CPUAllocator.cpp:73] data. DefaultCPUAllocator: not enough memory: you tried to allocate 15681160000 bytes. Buy new RAM!

'''
copus = ["الله اكبر ", "اهلا بيك يا خوي"]
    
vec = load_embedding()
X = reduce(copus , vec)
ML = get_PA()
out = ML.predict(copus)
print(out)
'''
app = Flask(__name__)

@app.route("/PA", methods = ['POST'])
def PA ():    
   data = request.json
   copus = []
   for text in data :
       copus.append(clean_text(text))
    
   vec = load_embedding()
   X = vec.transform(copus)
   ML = get_PA()
   out = ML.predict(copus)
   
   return jsonify(data)


@app.route("/DL", methods = ['POST'])
def DL ():    
   data = request.json
   return jsonify(data)

app.run(debug=True)
'''