import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
random.seed(42)

NAMES = open("C:\\Users\\samip\\Documents\\quick-maffs\\neural_nets\\names.txt",'r').read().split('\n')
random.shuffle(NAMES)
NAMES_COUNT = 5 
BLOCK_LENGTH = 5

#DATA SET
def load_dataset(NAMES, BLOCK_LENGTH = 3):
    CONTEXT_INPUT = []
    TARGETS = []
    CHARACTERS = ['.'] + sorted(set(list(''.join(NAMES))))
    STOI = {CHARACTER:INDEX for INDEX,CHARACTER in enumerate(CHARACTERS)}
    ITOS = {INDEX:CHARACTER for INDEX,CHARACTER in enumerate(CHARACTERS)}
    for NAME in NAMES[:NAMES_COUNT]:
        CONTEXT = [0] * BLOCK_LENGTH
        for CHAR in NAME + '.': 
            IX = STOI[CHAR]
            CONTEXT_INPUT.append(CONTEXT)
            TARGETS.append(IX)
            CONTEXT = CONTEXT[1:] + [IX]
    X = torch.tensor(CONTEXT_INPUT)
    Y = torch.tensor(TARGETS)
    return X, Y


##APPLY SPLIT
n1 = int(0.8*len(NAMES))
n2 = int(0.9*len(NAMES))

XTR, YTR = load_dataset(NAMES[:n1], BLOCK_LENGTH)
XDEV, YDEV = load_dataset(NAMES[n1:n2], BLOCK_LENGTH)
XTS, YTS = load_dataset(NAMES[n2:], BLOCK_LENGTH)

#SET UP MODEL FOR TRAINING

## 
G = torch.Generator(3232139832)
VOCAB_SIZE = 27
NUM_STEPS = 29299
EMB_DIM = 4
LAY1LEN = 100
W1 = torch.randn(BLOCK_LENGTH*EMB_DIM,LAY1LEN) #6x100
B1 = torch.randn(LAY1LEN)
W2 = torch.randn(LAY1LEN,VOCAB_SIZE)
B2 = torch.randn(VOCAB_SIZE)
C = torch.randn(VOCAB_SIZE,EMB_DIM, generator = G, requires_grad = True)
PARAMS = [C,W1,B2,W2,B2]
LEARNRATE = 0.01

for P in PARAMS:
    P.requires_grad = True

# training step
for i in NUM_STEPS: 
    EMB = C[XTR]
    #32,3,2
    HIDLAY = torch.tanh(EMB.view(-1,6) @ W1 + B1) #SUM VIA BROADCASTING
    LOGITS = HIDLAY@W2 + B2
    LOSS = F.cross_entropy(LOGITS,YTR[BATCH])
    for P in PARAMS: 
        P.grad = None 
    LOSS.backward()
    for P in PARAMS:
        P.data += -LEARNRATE*P.grad


    
@torch.no_grad()
def split_loss(split):
    x, y = {"train": (XTR, YTR), "val": (XDEV, YDEV), "test": (XTS, YTS)}[split]
    emb = C[x]
    h = torch.tanh(emb.view(-1,block_size*emd_dim)@W1+b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
    
    class MakeMoreNames:
        def __init__(self, names, block_length=3, vocab_size=27, emb_dim=4, lay1len=100, learn_rate=0.01, num_steps=29299, seed=42):
            random.seed(seed)
            self.names = names
            self.block_length = block_length
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
            self.lay1len = lay1len
            self.learn_rate = learn_rate
            self.num_steps = num_steps
            self.generator = torch.Generator().manual_seed(seed)
            self.stoi, self.itos = self._build_vocab(names)
            self.params = self._init_params()
            self.XTR, self.YTR, self.XDEV, self.YDEV, self.XTS, self.YTS = self._prepare_data()

        def _build_vocab(self, names):
            characters = ['.'] + sorted(set(list(''.join(names))))
            stoi = {char: idx for idx, char in enumerate(characters)}
            itos = {idx: char for idx, char in enumerate(characters)}
            return stoi, itos

        def _init_params(self):
            W1 = torch.randn(self.block_length * self.emb_dim, self.lay1len)
            B1 = torch.randn(self.lay1len)
            W2 = torch.randn(self.lay1len, self.vocab_size)
            B2 = torch.randn(self.vocab_size)
            C = torch.randn(self.vocab_size, self.emb_dim, generator=self.generator, requires_grad=True)
            params = [C, W1, B1, W2, B2]
            for p in params:
                p.requires_grad = True
            return params

        def _prepare_data(self):
            n1 = int(0.8 * len(self.names))
            n2 = int(0.9 * len(self.names))
            XTR, YTR = self._load_dataset(self.names[:n1])
            XDEV, YDEV = self._load_dataset(self.names[n1:n2])
            XTS, YTS = self._load_dataset(self.names[n2:])
            return XTR, YTR, XDEV, YDEV, XTS, YTS

        def _load_dataset(self, names):
            context_input = []
            targets = []
            for name in names:
                context = [0] * self.block_length
                for char in name + '.':
                    ix = self.stoi[char]
                    context_input.append(context)
                    targets.append(ix)
                    context = context[1:] + [ix]
            X = torch.tensor(context_input)
            Y = torch.tensor(targets)
            return X, Y

        def train(self):
            C, W1, B1, W2, B2 = self.params
            for i in range(self.num_steps):
                emb = C[self.XTR]
                hidlay = torch.tanh(emb.view(-1, self.block_length * self.emb_dim) @ W1 + B1)
                logits = hidlay @ W2 + B2
                loss = F.cross_entropy(logits, self.YTR)
                for p in self.params:
                    p.grad = None
                loss.backward()
                for p in self.params:
                    p.data += -self.learn_rate * p.grad

        @torch.no_grad()
        def split_loss(self, split):
            x, y = {"train": (self.XTR, self.YTR), "val": (self.XDEV, self.YDEV), "test": (self.XTS, self.YTS)}[split]
            C, W1, B1, W2, B2 = self.params
            emb = C[x]
            h = torch.tanh(emb.view(-1, self.block_length * self.emb_dim) @ W1 + B1)
            logits = h @ W2 + B2
            loss = F.cross_entropy(logits, y)
            print(split, loss.item())






