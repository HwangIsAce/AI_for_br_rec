import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score

import gensim
from gensim import models
from gensim.models import fasttext
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec, KeyedVectors

import networkx as nx
from node2vec import Node2Vec

def dataset_preprocess(df_origin):
    
    df_origin['계'] = df_origin['원재료명'] + " " + df_origin['함량(%)']
    df_origin['함량(%)'] = 1
    
    df_trans = df_origin[['제품명', '계', '함량(%)', '분류']]
    df_trans.rename(columns = {'계': '원재료명'}, inplace = True)
    
    return df_trans

def train_fasttext(X_train, epoch=10):
    
    # load a model
    ft_model = fasttext.load_facebook_model('data/wiki.ko.bin')
    ft_model.build_vocab(sentences=X_train, update=True)
    ft_model.train(X_train, total_examples=len(X_train), epoch=epoch)
    
    return ft_model

def train_node2vec(df): 
    G = nx.Graph()
    
    for i in range(len(df)):
        G.add_edge(df['제품명'][i], df['원재료명'][i], relation='ingredients')
        
    node2vec = Node2Vec(G, dimensions=20, walk_length=16, num_walks=100)
    model = node2vec.fit(window=10, min_count=1)
    
    return model

if __name__ == "__main__":
    df_origin = pd.read_csv('data/pm_cls.csv')
    df_origin.reset_index(drop=True, inplace=True)
    
    df_origin.drop('제품 코드', axis=1, inplace=True)
    
    X_train1 = list(df_origin['제품명'].unique())
    X_train2 = list(df_origin['원재료명'].unique())
    X_train = X_train1 + X_train2
    
    # train fasttext model
    train_fasttext(X_train, epoch=10)
    
    # change dataset format
    df_trans = dataset_preprocess(df_origin)
    
    # train node2vec model - version1
    train_node2vec(df_trans)
    
    # train node2vec model - version2
    train_node2vec(df_origin)
    
    # train/test split 안 했었나?