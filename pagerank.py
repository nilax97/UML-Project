import tensorflow_datasets as tfds
import tensorflow as tf
import os
import struct
import hashlib
import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
import tensorflow_hub as hub
import tokenization
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from sklearn.cluster import KMeans
import pickle
from sknetwork.ranking import PageRank
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import sys

def tensor_to_string(x):
    return x.numpy().decode('UTF-8')

def get_sent_list(text,stem=None):
    sents = sent_tokenize(text)
    if stem == "None":
        return sents
    if stem == "EnglishStemmer":
        stemmer = EnglishStemmer()
    ans = []
    for sent in sents:
        words = word_tokenize(sent)
        word_stem = [stemmer.stem(w) for w in words]
        ans.append(str(" ".join(str(word_stem))))
    return ans

transformer = sys.argv[1]

models = dict()
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models[transformer] = SentenceTransformer(transformer,cache_folder='/mnt/disks/disk-1/data/models')
models[transformer]._target_device = device

# # V = list of embeddings. k = target size of summary
# # Returns a list of sentence indices

def sim(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def generate_summary(V,k):
    if k >= len(V):
        return list(range(len(V)))
    n = V.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i][i] = sim(V[i],V[i])
        for j in range(i+1,n):
            s = sim(V[i], V[j])
            adj[i][j] = s
            adj[j][i] = s

    pr = PageRank()
    scores = pr.fit_transform(adj)
    ind = np.argpartition(scores, -k)[-k:]
    return np.sort(ind)

stem = sys.argv[3]

def uml_summary(x,index,kind="cnn_dailymail",model="all-MiniLM-L6-v2"):
    if kind == "cnn_dailymail":
        key1 = 'article'
        key2 = 'highlights'
    elif kind == "scientific_papers/arxiv" or kind == "scientific_papers/pubmed":
        key1 = 'article'
        key2 = 'abstract'
    text = tensor_to_string(x[key1])
    text = get_sent_list(text,stem)
    summary = tensor_to_string(x[key2])
    summary = get_sent_list(summary,stem)
    text_emb = models[model].encode(text)
    filename = str(index) + "_" + model + "_" + stem + ".pickle"
    folderpath = os.path.join("/mnt/disks/disk-1/data/pickle",kind)
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    filepath = os.path.join("/mnt/disks/disk-1/data/pickle",kind,filename)
    with open(filepath, 'wb') as handle:
        pickle.dump(text_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gen_sum = [text[x] for x in generate_summary(text_emb,len(summary))]
    scores = scorer.score(" ".join(summary)," ".join(gen_sum))
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

dataset = sys.argv[2]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)

train, val, test = tfds.load(name=dataset, 
                      split=["train", "validation", "test"], 
                      data_dir="/mnt/disks/disk-1/data")

r1 = []
r2 = []
rl = []
index = 0
for x in list(test):
    r1_val,r2_val,rl_val = uml_summary(x,index,kind=dataset,model=transformer)
    index += 1
    r1.append(r1_val)
    r2.append(r2_val)
    rl.append(rl_val)
print(sys.argv,index)
print("Rouge 1 : ",np.round(np.mean(np.asarray(r1))*100,2))
print("Rouge 2 : ",np.round(np.mean(np.asarray(r2))*100,2))
print("Rouge L : ",np.round(np.mean(np.asarray(rl))*100,2))
print("___")
