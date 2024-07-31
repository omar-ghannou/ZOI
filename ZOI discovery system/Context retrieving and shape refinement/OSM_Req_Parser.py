from collections import defaultdict
from OSMPythonTools.overpass import overpassQueryBuilder
from OSMPythonTools.overpass import Overpass
from Utilities import stop_margin

stop_margin_lat , stop_margin_lon = stop_margin(2500,2500)

def is_node(elem):
  return elem.type() == 'node'

def is_way(elem):
  return elem.type() == 'way'

def is_relation(elem):
  return elem.type() == 'relation'

def get_nodes_tags(node):
  if node.tags() != None and type(node.tags()) is dict:
      return node.tags()
  else:
    return []

def get_way_tags(way):
  tags_w = []
  for node in way.nodes():
    res = get_nodes_tags(node)
    if res != []:
      tags_w.append(res)
  return merge_dict(tags_w)

def get_relation_tags(relation,_shallow=False):
  tags_r = []
  flag = True
  #tags += relation.tags()
  for member in relation.members():
    if is_node(member):
      if flag :
        tags_r = []
        flag = False
      res = get_nodes_tags(member)
      if res != []:
        tags_r.append(res)
        pass
    elif is_way(member):
        tags_r += get_way_tags(member)

  return merge_dict(tags_r)

def merge_dict(dict_list):
  dd = defaultdict(list)
  for d in dict_list: # you can list as many input dicts as you want here
    for key, value in d.items():
      dd[key].append(value)
  return dd

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def analyse(data):
  tags_a = []
  for i in data.elements():
    if is_node(i):
      if(get_nodes_tags(i) != []):
        tags_a.append(get_nodes_tags(i))
    elif is_way(i):
      tags_a.append(get_way_tags(i))
    elif is_relation(i):
      tags_a.append(get_relation_tags(i))
  final_dict = {}
  p = 0
  for k,v in merge_dict(tags_a).items():
    v = flatten(v)
    final_dict[k] = np.unique(v)
  return final_dict


def get_context_bbox(bbox,timeout=20000):
  query = overpassQueryBuilder(bbox=[bbox['latLower'],bbox['lonLower'],bbox['latHigher'],bbox['lonHigher']], elementType=['node','way','relation'],
                                  out='body',includeGeometry=True)
  overpass = Overpass()
  data = overpass.query(query, timeout=timeout)
  return analyse(data)



def get_context_poly(poly,timeout=20000):
  query_poly = '(node(' + poly + ');way(' + poly + ');rel(' + poly + ');); out body geom;'
  overpass = Overpass()
  data_poly = overpass.query(query_poly, timeout=timeout)
  return analyse(data_poly)

def get_context_around(around,timeout=20000):
  query_around = '(node(' + around + ');way(' + around + ');rel(' + around + ');); out body geom;'
  overpass = Overpass()
  data_around = overpass.query(query_around, timeout=timeout)
  return analyse(data_around)

def All_stops_context_bbox(stops):
  context = {}
  for i in range(len(stops)):
    bbox = {'latLower':stops[i][1]-stop_margin_lat,'lonLower':stops[i][0]-stop_margin_lon,'latHigher':stops[i][1]+stop_margin_lat,'lonHigher': stops[i][0]+stop_margin_lon}
    context[i] = get_context_bbox(bbox)
  return context

def All_stops_context_poly(bboxs,geom):
  context = {}
  for i in range(len(bboxs[:9])):
    context[i] = get_context_poly(bboxs[i])
  for j in range(0,len(geom[:3])):
    key = len(bboxs[:9]) + j
    context[key] = get_context_poly(geom[j])
  return context

def All_stops_context_around(arounds):
  context = {}
  for i in range(len(arounds)):
    context[i] = get_context_around(arounds[i])
  return context

#context = All_stops_context_bbox(get_stops(nontemporal_trajectory,tdb_labels)) ## porto
#context = All_stops_context_bbox(stops) ## traxens
#context = All_stops_context_poly(bboxs,geom)
#context = All_stops_context_around(around)
context = All_stops_context_around(around2)
context