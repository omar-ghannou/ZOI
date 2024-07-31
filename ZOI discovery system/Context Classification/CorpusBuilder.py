
import xml.etree.ElementTree as ET
from numpy.lib.polynomial import polyint
import geopandas as gpd
from shapely.geometry import Polygon, MultiLineString
from shapely import MultiPoint, LineString, Point, LinearRing
import random
from enum import Enum
import itertools
import pandas as pd
import numpy as np
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download
import warnings
import gensim
from gensim.models import Word2Vec
import io
import sys

def Parser4(Path=None):
    all_tags = {}
    if not Path:
      Path = './Demo.xml'

    print(Path)
    root = ET.parse(Path).getroot()
    for element in root:
      match element.tag:
        case 'node':
             key_value = {}
             has_tag = False
             for tag in element:
              key_value[tag.attrib['k']] = tag.attrib['v']
              has_tag = True
             if has_tag:
              key = "node_"+ element.attrib['id']
              all_tags[key] = key_value
        case 'way':
             key_value = {}
             has_tag = False
             for ele in element:
              if ele.tag == 'tag':
                key_value[ele.attrib['k']] = ele.attrib['v']
                has_tag = True
             if has_tag:
                    key = "way_"+ element.attrib['id']
                    all_tags[key] = key_value
        case 'relation':
             key_value = {}
             has_tag = False
             for ele in element:
              if ele.tag == 'tag':
                key_value[ele.attrib['k']] = ele.attrib['v']
                has_tag = True
             if has_tag:
                key = "relation_"+ element.attrib['id']
                all_tags[key] = key_value
        case _:
            pass
    return all_tags

def Parser3(Path=None):
    all_tags = {}
    if not Path:
      Path = './Demo.xml'

    print(Path)
    root = ET.parse(Path).getroot()
    for element in root:
      match element.tag:
        case 'node':
             key_value = {}
             has_tag = False
             for tag in element:
              key_value[tag.attrib['k']] = tag.attrib['v']
              has_tag = True
             if has_tag:
              key = "n"+ element.attrib['id']
              all_tags[key] = key_value
        case 'way':
             key_value = {}
             has_tag = False
             for ele in element:
              if ele.tag == 'tag':
                key_value[ele.attrib['k']] = ele.attrib['v']
                has_tag = True
             if has_tag:
                    key = "w"+ element.attrib['id']
                    all_tags[key] = key_value
        case 'relation':
             key_value = {}
             has_tag = False
             for ele in element:
              if ele.tag == 'tag':
                key_value[ele.attrib['k']] = ele.attrib['v']
                has_tag = True
             if has_tag:
                key = "r"+ element.attrib['id']
                all_tags[key] = key_value
        case _:
            pass
    return all_tags

def get_keys(all_tags):
  keys = []
  for k in all_tags.values(): 
     keys += k.keys()
     #print(k.keys())
  return list(set(keys))

def get_values(all_tags):
  values = []
  for k in all_tags.values(): 
     values += k.values()
     #print(k.keys())
  return list(set(values))

def build_table(all_tags):
   keys = get_keys(all_tags)

def build_corpus_stat(tags):
  s = ""
  for _,dic in tags.items():
     for k,v in dic.items():
        s += str(k) + " " + str(v) + " "
  with open('corpus.crps',encoding="utf-8", mode='a') as f:
    f.write(s)
  return s

def build_keys_corpus_stat(tags):
  s = ""
  for i in tags:
    s += str(i) + " "
  with open('corpus_keys.crps',encoding="utf-8", mode='a') as f:
    f.write(s)
  return s

def build_values_corpus_stat(tags):
  s = ""
  for i in tags:
    s += str(i) + " "
  with open('corpus_values.crps',encoding="utf-8", mode='a') as f:
    f.write(s)
  return s

def build_values_clustering(tags):
  corpus = []
  for _,dic in tags.items():
     s = ""
     for k,v in dic.items():
        s += str(v) + " "
     s+='\n'
     corpus.append(s)
  with open('corpus_values_clus.crps',encoding="utf-8", mode='a') as f:
    f.writelines(corpus)
  return corpus

def build_keys_clustering(tags):
  corpus = []
  for _,dic in tags.items():
     s = ""
     for k,v in dic.items():
        s += str(k) + " "
     s+='\n'
     corpus.append(s)
  with open('corpus_keys_clus.crps',encoding="utf-8", mode='a') as f:
    f.writelines(corpus)
  return corpus

def build_corpus_clustering(tags):
  corpus = []
  for _,dic in tags.items():
     s = ""
     for k,v in dic.items():
        s += str(k) + " " + str(v) + " "
     s+='\n'
     corpus.append(s)
  with open('corpus_clus_test.crps',encoding="utf-8", mode='a') as f:
    f.writelines(corpus)
  return corpus

def annotate(corpus):
  labels = []
  for s in corpus:
     print(s)
     labels.append(input("Label : "))
  with open('corpus.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(corpus)
  with open('labels.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(labels)

def word2vec_func():
  download('punkt')
  warnings.filterwarnings(action = 'ignore')
  sample = open('corpus.csv',encoding="utf-8")
  s = sample.readlines()
  #f = s.replace("\n", " ")
  data = []
 
  # iterate through each sentence in the file
  for i in s:
      temp = []
      # tokenize the sentence into words
      for j in word_tokenize(i):
          temp.append(j.lower())
  
      data.append(temp)
  
  # Create CBOW model
  model1 = gensim.models.Word2Vec(data, min_count = 1,
                                vector_size = 50, window = 5)
  return data, model1
  
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []
    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features
    


#nodes, ways, relations, tags = Parser2("data.xml")
#print(len(nodes),len(ways),len(relations),len(tags))
#center = list(nodes.values())[0]
#center = [center[1],center[0]]
#m = build_map(shapes, shape_type, pop)

#tags = Parser3("./P1.osm")
#build_corpus(tags)

#build_corpus(tags)
#tokenized_docs, model1 = word2vec_func()
#vectorized_docs = vectorize(tokenized_docs, model=model1)
#print(len(vectorized_docs), len(vectorized_docs[0]))
#
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA
#
#norm_X = MinMaxScaler().fit_transform(vectorized_docs)
#pca = PCA(n_components=2)
#X_pca = pca.fit_transform(norm_X)
#
#
#from sklearn.cluster import DBSCAN
#clustering = DBSCAN(eps=0.5, min_samples=5).fit(vectorized_docs)
#
#def find_indices(list_to_check, item_to_find):
#    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]
#
#colors = plt.cm.get_cmap('hsv', len(np.unique(clustering.labels_)))
#plot_color = [mcolors.rgb2hex(colors(i)) for i in clustering.labels_]
#
#plt.scatter(x=X_pca[:,0], y=X_pca[:,1],c=plot_color)
#plt.show()
#
#print([tokenized_docs[index] for index in find_indices(clustering.labels_, -1)])
#print(len(np.unique(clustering.labels_)))

#annotate()


