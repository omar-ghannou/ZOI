import requests
from LoadData import Generate_stopBbox, Load_Stops


def execute_query(shapes):
    prefix = "[out:xml] [timeout:25];("
#node(22.491456595835217, 113.86777080811105, 22.500500313164785, 113.87675391988894); 
#way(22.491456595835217, 113.86777080811105, 22.500500313164785, 113.87675391988894); 
#relation(22.491456595835217, 113.86777080811105, 22.500500313164785, 113.87675391988894);
    suffix = ");(._;>;); out body geom;"

    url = r"http://overpass-api.de/api/interpreter" 

    for shape in shapes:
        node = "node(" + str(shape[0]) + ", " + str(shape[1]) + ", " + str(shape[2]) + ", " + str(shape[3]) + ");"
        way = "way(" + str(shape[0]) + ", " + str(shape[1]) + ", " + str(shape[2]) + ", " + str(shape[3]) + ");"
        relation = "relation(" + str(str(shape[0])) + ", " + str(shape[1]) + ", " + str(shape[2]) + ", " + str(shape[3]) + ");"
        prefix = prefix + node + way + relation
    query = prefix + suffix
    #print(query)
    #x = requests.post(url, data = query)
    with open('data.xml', 'w', encoding="utf-8") as f:
        f.write(requests.post(url, data = query).text)
    #return x.text

bboxs = Generate_stopBbox(Load_Stops())
execute_query(bboxs)