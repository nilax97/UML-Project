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
        ans.append(str(" ".join(word_stem)))
    return ans

## Model features include an encode function -> takes a list of sentences. Returns a list of embeddings (all same dim)
transformers = [sys.argv[1]]

models = dict()
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for trans in transformers:
    models[trans] = SentenceTransformer(trans,cache_folder='/mnt/disks/disk-1/data/models')
    models[trans]._target_device = device

## V = list of embeddings. k = target size of summary
## Returns a list of sentence indices

def generate_summary(V, k):
    if k >= len(V):
        return list(range(len(V)))
    k -= 1
    centers = []
    cities = list(range(len(V)))
    centers.append(0)
    cities.remove(0)
    while k!= 0:
        city_dict = {}
        for cty in cities:
            min_dist = float("inf")
            for c in centers:
                min_dist = min(min_dist,np.linalg.norm(V[cty] - V[c]))
            city_dict[cty] = min_dist
        new_center = max(city_dict, key = lambda i: city_dict[i])
        centers.append(new_center)
        cities.remove(new_center)
        k -= 1
    return centers

def uml_summary(x,index,kind="cnn_dailymail",model="all-MiniLM-L6-v2"):
    if kind == "cnn_dailymail":
        key1 = 'article'
        key2 = 'highlights'
    elif kind == "scientific_papers/arxiv" or kind == "scientific_papers/pubmed":
        key1 = 'article'
        key2 = 'abstract'
        
    stemmer = sys.argv[3]
    text = tensor_to_string(x[key1])
    text = get_sent_list(text,stemmer)
    summary = tensor_to_string(x[key2])
    summary = get_sent_list(summary,stemmer)
    
    filename = str(index) + "_" + model + "_" + stemmer + ".pickle"
    folderpath = os.path.join("/mnt/disks/disk-1/data/pickle",kind)
    filepath = os.path.join("/mnt/disks/disk-1/data/pickle",kind,filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'rb') as handle:
            text_emb = pickle.load(handle)
    else:
        text_emb = models[model].encode(text)
        with open(filepath, 'wb') as handle:
            pickle.dump(text_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    gen_sum = [text[x] for x in generate_summary(text_emb,len(summary))]
    scores = scorer.score(" ".join(summary)," ".join(gen_sum))
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

datasets = [sys.argv[2]]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
for ds in datasets:
    for trans in transformers:
        train, val, test = tfds.load(name=ds, 
                              split=["train", "validation", "test"], 
                              data_dir="/mnt/disks/disk-1/data")
        
        r1 = []
        r2 = []
        rl = []
        index = 0
        for x in list(test):
            r1_val,r2_val,rl_val = uml_summary(x,index,kind=ds,model=trans)
            index += 1
            r1.append(r1_val)
            r2.append(r2_val)
            rl.append(rl_val)
        print(sys.argv,index)
        print("Rouge 1 : ",np.round(np.mean(np.asarray(r1))*100,2))
        print("Rouge 2 : ",np.round(np.mean(np.asarray(r2))*100,2))
        print("Rouge L : ",np.round(np.mean(np.asarray(rl))*100,2))
        print("___")
