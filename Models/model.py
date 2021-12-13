from random import choice
import torch
import torch.nn    as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from string          import ascii_letters
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

# params
H = 32 #-- Hidden dim
V = 52 #-- Vocabulary size
D = 4 #-- Embedding dim
L = 2 #-- Number of layers
C = 18 #-- Number of classes

def build_model():

    mData   = np.load('NamesData.npy')

    lNames  = mData[:,0]
    lLabels = mData[:,1]
    N       = mData.shape[0]

    PATH = './ManyToOne_dict.pt'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    oVocab   = build_vocab_from_iterator([c for c in ascii_letters])

    oModel_state_dict = torch.load(PATH)
    oManyToOne = RNN()
    oManyToOne.load_state_dict(oModel_state_dict)

    return oManyToOne, np.unique(lLabels), lNames, oVocab


def Name2Tensor(oVocab, name):
    return torch.tensor(oVocab(list(name)))

def PackedAs(mX, mPack):
    return PackedSequence(mX, mPack.batch_sizes, mPack.sorted_indices, mPack.unsorted_indices)

def PackedAs(mX, mPack):
    return PackedSequence(mX, mPack.batch_sizes, mPack.sorted_indices, mPack.unsorted_indices)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
       
        # convert vocab to embedding (V - > D)
        self.oEmbeeding = nn.Embedding(V, D)

        # hidden layer 
        self.oGRU       = nn.GRU(D, H, L, dropout=0.15, bidirectional=True)

        # NN Layer
        # self.oLinear    = nn.Linear(H, C)
        ## error: mat1 and mat2 shapes cannot be multiplied (64x64 and 32x18)
        # bidirectional
        self.oLinear    = nn.Linear(H + H, C)
        
        ###
        torch.nn.init.xavier_normal_(self.oLinear.weight.data)
        
    def forward(self, mPackNames):
        #-- mPackNames.shape = (N*T,)
        
        # mE     = self.oEmbeeding(mPackNames)                             #-- mE        .shape = (N*T, D)
        ## error: embedding(): argument 'indices' (position 2) must be Tensor, not PackedSequence
        mE     = self.oEmbeeding(mPackNames.data) 

        mPackE = PackedAs       (mE, mPackNames)                  #-- mPackE    .shape = (N*T, D)
        
        _, mH  = self.oGRU      (mPackE)                             #-- mH        .shape = (2*L, N, H) 

        # since this is bidirectional we have to cat
        mH     = torch.cat      ([mH[-1,:,:], mH[-2,:,:]], dim=1) #-- mH        .shape = (N,   2*H)
               
        # mZ     = self.oLinear   (mH)                             #-- mZ        .shape = (N, C)
        ## error: mat1 and mat2 shapes cannot be multiplied (64x64 and 32x18)
        mZ     = self.oLinear   (mH) 
        
        return mZ

def predict_country(oManyToOne, name, oVocab, lCategories):
    with torch.no_grad():
        vName       = Name2Tensor(oVocab, name)
        mPackNames  = pack_sequence([vName], enforce_sorted=False)
        yhat        = oManyToOne(mPackNames)
        m = nn.Softmax(dim=1) 
        ySoftmax    = m(yhat)
        country     = lCategories[torch.argmax(yhat)]
        return country, np.array(ySoftmax[0]), yhat[0]







