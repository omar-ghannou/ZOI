
import xml.etree.ElementTree as ET
from numpy.lib.polynomial import polyint
import geopandas as gpd
from shapely.geometry import Polygon, MultiLineString
from shapely import MultiPoint, LineString, Point, LinearRing
import folium
from folium import IFrame
import random
from enum import Enum
import itertools
from UI import Window
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

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets


class Shape(Enum):
    Point = 1
    Way = 2
    ComplexWay = 3
    Polygon = 4
    ComplexPolygon = 5


class relationMember:
    def __init__(self):
        self.memberType = ""
        self.memberId = ""
        self.coords = []
        self.memberRole = ""

class relation:
    def __init__(self):
        self.RelType = ""
        self.members = []

def Parser3(Path=None, stops = None ,center=None):
    all_tags = {}
    if not Path:
      Path = './Demo.xml'

    if not center:
      center = [18.949923, 72.942207] #To refine
    tiles = []

    map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=12)
    tiles.append(folium.TileLayer( tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                      attr = 'Esri', name = 'Satellite', overlay = False, control = True))

    tiles.append(folium.TileLayer(tiles='https://tile.thunderforest.com/transport/{z}/{x}/{y}.png?apikey=e22170de87894e0793ce487a27a69ee0', attr = 'Transport Map', name = 'Transport'))
    tiles.append(folium.TileLayer(tiles='https://tile.thunderforest.com/transport-dark/{z}/{x}/{y}.png?apikey=e22170de87894e0793ce487a27a69ee0', attr = 'Dark Transport Map', name = 'Dark Transport'))#, name = 'Map Quest Open', overlay = False, control = True))
    tiles.append(folium.TileLayer(tiles='https://tile.thunderforest.com/landscape/{z}/{x}/{y}.png?apikey=e22170de87894e0793ce487a27a69ee0', attr = 'Landscape Map', name = 'Landscape'))#, name = 'Aerial', overlay = False, control = True))

    tiles.append(folium.TileLayer('openstreetmap', attr = 'osm', name = 'Open Street Map', overlay = False, control = True))
    for tile in tiles:
        map.add_child(tile)

    del tiles

    print("Mark1")

    #map = folium.Map(center, zoom_start=17, tiles='cartodbpositron')
    get_colors = lambda n: "#%06x" % random.randint(0x0000FF, 0x0199FF)
    c = 0
    Stops = folium.FeatureGroup(name="Stops")
    Points = folium.FeatureGroup(name="Points")
    Ways = folium.FeatureGroup(name="Ways")
    Polygons = folium.FeatureGroup(name="Polygons")
    ComplexWays = folium.FeatureGroup(name="ComplexWays")
    ComplexPolygons = folium.FeatureGroup(name="ComplexPolygons")

    df = pd.read_csv('CLEAN_STOPS.csv')
    for ind in df.index:
      Stops.add_child(folium.GeoJson(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point([float(df['longitude'][ind]),float(df['latitude'][ind])])]),
                  style_function=lambda x: {'fillColor': '#0055ff', 'color': '#0055ff'},
                  popup=folium.Popup(getHTML("Trajectory", str("Stop Point") ,str("Stop point in the trajectory")), max_width=3000, parse_html=True)))
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
              Points.add_child(folium.GeoJson(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Point([float(element.attrib['lon']),float(element.attrib['lat'])])]),
                  style_function=lambda x: {'fillColor': get_colors(1), 'color': get_colors(1)},
                  popup=folium.Popup(getHTMLdict("Node", element.attrib['id'] ,key_value), max_width=3000, parse_html=True)))
              key = "n"+ element.attrib['id']
              all_tags[key] = key_value
        case 'way':
             way = []
             nds = 0
             has_tag = False
             key_value = {}
             for ele in element:
              if ele.tag == 'nd':
                nds += 1
                way.append([float(ele.attrib['lon']),float(ele.attrib['lat'])])
              elif ele.tag == 'tag':
                print(ele.attrib['k'],ele.attrib['v'])
                key_value[ele.attrib['k']] = ele.attrib['v']
                has_tag = True
             if nds > 1 :
               if has_tag:
                  if isClosed(way):
                    Polygons.add_child(folium.GeoJson(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Polygon(way)]),
                                                  style_function=lambda x: {'fillColor': get_colors(1), 'color': get_colors(1)},
                                                  popup=folium.Popup(getHTMLdict("Polygon", element.attrib['id'], key_value), max_width=3000, parse_html=True)))
                    key = "p"+ element.attrib['id']
                    all_tags[key] = key_value
                  else:
                    Ways.add_child(folium.GeoJson(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[LineString(way)]),
                                                  style_function=lambda x: {'fillColor': get_colors(1), 'color': get_colors(1)},
                                                  popup=folium.Popup(getHTMLdict("Way", element.attrib['id'], key_value), max_width=3000, parse_html=True)))
                    key = "w"+ element.attrib['id']
                    all_tags[key] = key_value

        case 'relation':
             rela = relation()
             key_value = {}
             has_tag = False
             for ele in element:
              if ele.tag == 'member':
                rm = relationMember()
                rm.memberType = ele.attrib['type']
                rm.memberId = ele.attrib['ref']
                rm.memberRole = ele.attrib['role']
                for nd in ele:
                  rm.coords.append([float(nd.attrib['lon']),float(nd.attrib['lat'])])
                rela.members.append(rm)
              elif ele.tag == 'tag':
                key_value[ele.attrib['k']] = ele.attrib['v']
                has_tag = True
                if ele.attrib['k'] == 'type':
                  rela.RelType = ele.attrib['v']
             if rela.RelType == "multipolygon" or rela.RelType == "site" or rela.RelType == "building" :#or rela.RelType == "boundary":
                shell = []
                holes = []
                for member in rela.members:
                   if member.memberType == "way":
                     if member.memberRole == "inner":
                       holes.append(member.coords)
                     else:
                       shell.append(member.coords)
                poly = None
                shell = list(itertools.chain.from_iterable(shell))
                if holes:
                  if shell:
                    hl = []
                    for hole in holes:
                      if(len(hole)>2):
                        hl.append(hole)
                    #if element.attrib['id'] == str(287077):
                    #holes = 
                    #print(len(holes))
                      #print(holes[24])
                      #print(holes[26])
                      #print(holes[27])
                      #[print(idx,LinearRing(ring),"\n\n") for idx,ring in enumerate(holes)]
                    #print(element.attrib['id'])
                    #[LinearRing(ring) for (idx,ring) in enumerate(holes)]
                    poly = Polygon(shell,hl)
                    #print(element.attrib['id'],rela.RelType)
                  else:
                    poly = Polygon(list(itertools.chain.from_iterable(holes))) #it arrives that relation contains holes only, it is kind of typing error in OSM
                else:
                  poly = Polygon(shell)
                ComplexPolygons.add_child(folium.GeoJson(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[poly]),
                                                  style_function=lambda x: {'fillColor': get_colors(1), 'color': get_colors(1)},
                                                  popup=folium.Popup(getHTMLdict("Complex Polygon", element.attrib['id'] ,key_value), max_width=3000, parse_html=True)))
                key = "cp"+ element.attrib['id']
                all_tags[key] = key_value
             else: #According to OSM conventions it is kind of way or variant, that is why we consider them as Lines
                waysList = []
                for member in rela.members:
                  if member.memberType == "way":
                    waysList.append(LineString(member.coords))
                #poly = Polygon(shell,holes)
                if waysList:
                  ComplexWays.add_child(folium.GeoJson(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[MultiLineString(waysList)]),
                                                  style_function=lambda x: {'fillColor': get_colors(1), 'color': get_colors(1)},
                                                  popup=folium.Popup(getHTMLdict("Complex Way", element.attrib['id'] ,key_value), max_width=3000, parse_html=True)))
                  key = "cw"+ element.attrib['id']
                  all_tags[key] = key_value
        case _:
            pass

    print("Mark2")
    map.add_child(Stops)
    map.add_child(Points)
    print(len(Stops._children))
    print(len(Points._children))
    print(len(Ways._children))
    print(len(Polygons._children))
    print(len(ComplexWays._children))
    print(len(ComplexPolygons._children))
    map.add_child(Ways)
    map.add_child(Polygons)
    map.add_child(ComplexWays)
    map.add_child(ComplexPolygons)
    #folium.LatLngPopup().add_to(map)
    map.add_child(folium.map.LayerControl())
    print("Mark3")
    return map, all_tags



def isClosed(way):
  if len(way) > 2:
    return way[0] == way[-1]
  else:
    return False

def getHTML(type, id, context):
  html="""<body style = "background:#87CEEB">
    <div>
      <h1>
        <span style = "color:white"> """ + type + """ : """ + id + """</span>
        </hr>
      </h1>
      <span style = "color:white; font-size:20px"> """ + context + """</span>
    </div>
    </body>
    """
  return IFrame(html=html, width=450, height=250)

def getHTMLdict(type, id, context):
  name = ""
  if "name" in context:
     name = context["name"]
  string = ""
  for k,v in context.items():
     string = string + k + " : " + v + "<br>"
  
  html="""<body style = "background:#87CEEB">
    <div>
      <h1>
        <span style = "color:white"> """ + type + """ : """ + name + """</span>
        </hr>
      </h1>
      <span style = "color:white; font-size:20px"> """ + string + """</span>
    </div>
    </body>
    """
  return IFrame(html=html, width=450, height=250)

def get_keys(all_tags):
  keys = []
  for k in all_tags.values(): 
     keys += k.keys()
     #print(k.keys())
  return list(set(keys))

def build_table(all_tags):
   keys = get_keys(all_tags)

def build_corpus(tags):
  corpus = []
  for tag in tags:
     s = ""
     for item in tags[tag].items():
        s += str(item[0]) + " " + str(item[1]) + " "
     s += s + "\n"
     corpus.append(s)
  with open('corpus.csv',encoding="utf-8", mode='w') as f:
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
    

App = QtWidgets.QApplication(sys.argv)

#nodes, ways, relations, tags = Parser2("data.xml")
#print(len(nodes),len(ways),len(relations),len(tags))
#center = list(nodes.values())[0]
#center = [center[1],center[0]]
#m = build_map(shapes, shape_type, pop)

map, tags = Parser3("./CMA.xml")

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



window = Window(Map=map, Tags=tags)
window.show()
sys.exit(App.exec())