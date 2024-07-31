import pandas as pd
import re
bboxs = []
geom = []
Load_ZOI_From_File = True
if Load_ZOI_From_File:
  ZOI_Shapes = pd.read_csv("CLEAN_ZOI.csv")
  i = 0
  geom_elem = 'poly:"'
  while i < len(ZOI_Shapes):
    #print(str(i) + "/" + str(len(ZOI_Shapes)))
    number = re.findall(r'\d+', ZOI_Shapes['zoiId'][i])
    if ZOI_Shapes['zoiId'][i][:-len(number[0])] == 'uniquePoint::':
      bboxs.append('poly:"'+ str(ZOI_Shapes['latitude'][i]) +' '+ str(ZOI_Shapes['longitude'][i]) + ' '
                   + str(ZOI_Shapes['latitude'][i+1]) +' '+ str(ZOI_Shapes['longitude'][i+1]) + ' '
                   + str(ZOI_Shapes['latitude'][i+2]) +' '+ str(ZOI_Shapes['longitude'][i+2]) + ' '
                   + str(ZOI_Shapes['latitude'][i+3]) +' '+ str(ZOI_Shapes['longitude'][i+3]) + '"')
      i+=4
    elif ZOI_Shapes['zoiId'][i][:-len(number[0])] == 'cluster::':
      j=i
      while(j<len(ZOI_Shapes)-1):
        if ZOI_Shapes['zoiId'][j] == ZOI_Shapes['zoiId'][j+1]:
          j+=1
        else:
          break
      for k in range(i,j+1):
        point = str(ZOI_Shapes['latitude'][k]) +' '+ str(ZOI_Shapes['longitude'][k]) + ' '
        geom_elem += point
      geom_elem = geom_elem[:-1] + '"'
      geom.append(geom_elem)
      geom_elem = 'poly:"'
      i = j+1

for g in geom:
    print("geom : ", g)
for b in bboxs:
    print("bbox : ", b)


