import math

def stop_margin(latDistance,longDistance):
  lat = latDistance / (1000 * 110.574) # meter
  lon = longDistance / (1000 * (111.320 * math.cos(lat * math.pi / 180))) # meter
  return lat,lon