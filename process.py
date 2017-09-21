import pickle
import numpy as np
import random

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def initialize():
    return load_obj('protVec')

def split(start, model, seq, lis):
    for index in xrange(start,len(seq) - 2,3):
        kmer = seq[index:index+3].encode('utf-8')
        if kmer in model:
            lis.append(np.array(model[kmer]))
        else:
            lis.append(np.array(model['<unk>']))
    lis = np.mean(lis, axis=0).tolist()
    return lis   

def embedding(model, seq):
    first, second, third = [], [] , []
    #First Split
    first = split(0, model, seq, first)
    #Second Split
    second = split(1, model, seq, second)
    #Third Split
    third = split(2, model, seq, third)
    return first, second, third

model = initialize()
num = 0
batch = []
batch_size = 3
with open('pos.data') as file:
    for line in file:
        line = line.rstrip('\n')
        line = line.split()
        f,s,t = embedding(model, line[0])
        mean = (np.array(f) + np.array(s) + np.array(t))/3.0
        #print mean
        num += 1
        batch.append(mean)
        if num > 10:
            break
print random.sample(batch, batch_size)