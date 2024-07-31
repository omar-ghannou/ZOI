import io
import sys

import folium

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5 import QtWebEngineWidgets

from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem
from PyQt5.QtCore import pyqtSlot
import sys

class TableView(QTableWidget):
    def __init__(self, data, *args):
        QTableWidget.__init__(self, *args)
        self.data = data
        self.keys = ['id'] + self.get_keys()
        self.setRowCount(len(self.data)+1)
        self.setColumnCount(len(self.keys))
        self.setData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.setHorizontalHeaderLabels(self.keys)
 
    def setData(self):
        for n, key in enumerate(self.keys):
            for m, item in enumerate(self.data.items()):
                if n == 0:
                    item = item[0]
                else:
                    if key in item[1].keys():
                        item = item[1][key]
                    else:
                        item = "NULL"
                
                newitem = QTableWidgetItem(item)
                self.setItem(m, n, newitem)

    def get_keys(self):
        keys = []
        for k in self.data.values():
           keys += k.keys()
           #print(k.keys())
        return list(set(keys))
    
 
 

    #def build_table(self):
    #    keys = self.get_keys()
 


class Window(QtWidgets.QMainWindow):

    shortPathButton = None
    button2 = None
    button3 = None
    view = None
    central_widget = None
    button_container = None
    lay = None
    vlay = None
    map = None
    data = None
    tiles = []

    def __init__(self, Map=None,Tags=None,Keys=None):
        super().__init__()
        self.map = Map
        self.data = Tags
        self.initWindow()
        self.fdata = dict(filter(self.filtering, self.data.items()))

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def initWindow(self):
        self.setWindowTitle(self.tr("MAP PROJECT"))
        self.setMinimumSize(1500, 800)
        self.SetUI()

    def filtering(self, pair):
        key, value = pair
        if key[0] == 'p':
            return True  # filter pair out of the dictionary
        else:
            return False  # keep pair in the filtered dictionary

    def Set_trajectory(self, _Trajectory):
        self.Trajectory = _Trajectory
        self.map.location = [self.Trajectory[0][0],self.Trajectory[0][1]]
        trajectories_group = folium.FeatureGroup(name="Trajectories").add_to(self.map)
        for i in range(len(self.Trajectory)):
            trajectories_group.add_child(folium.Marker(location=[self.Trajectory[i][0],self.Trajectory[i][1]]))

    def Set_Points(self, _Points):
        self.map.location = [_Points[0][0], _Points[0][1]]
        points_group = folium.FeatureGroup(name="Points").add_to(self.map)
        for i in range(len(_Points)):
            points_group.add_child(folium.Marker(location=[_Points[i][0], _Points[i][1]]))
    
    def Set_Stops(self, _Stops):
        self.map.location = [_Stops[0][0], _Stops[0][1]]
        stops_group = folium.FeatureGroup(name="Points").add_to(self.map)
        for i in range(len(_Stops)):
            stops_group.add_child(folium.Marker(location=[_Stops[i][0], _Stops[i][1]]))

    def Set_Ways(self, _Ways):
        ways_group = folium.FeatureGroup(name="Ways").add_to(self.map)
        for i in range(len(self.Trajectory)):
            ways_group.add_child(folium.Marker(location=[self.Trajectory[i][0],self.Trajectory[i][1]]))

    def Set_Polygons(self, _Polygons):
        polygons_group = folium.FeatureGroup(name="Polygons").add_to(self.map)
        for i in range(len(self._Polygons)):
            polygons_group.add_child(folium.Marker(location=[self.Trajectory[i][0],self.Trajectory[i][1]]))

    def show_attributes(self):
        global table
        table = TableView(self.fdata)
        table.show()

    def SetUI(self):
        self.shortPathButton = QtWidgets.QPushButton(self.tr("Show Attributes"))
        self.button2 = QtWidgets.QPushButton(self.tr("Show Satelite Map"))
        self.button3 = QtWidgets.QPushButton(self.tr("Show Trajectory"))

        self.shortPathButton.setFixedSize(120, 50)
        self.button2.setFixedSize(120, 50)
        self.button3.setFixedSize(120, 50)

        self.shortPathButton.clicked.connect(self.show_attributes)
        #self.button2.clicked.connect(self.Visualize_sat_map)
        #self.button3.clicked.connect(self.Show_trajectory)

        self.view = QtWebEngineWidgets.QWebEngineView()
        #self.view.setContentsMargins(50, 50, 50, 50)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.lay = QtWidgets.QHBoxLayout(self.central_widget)

        self.button_container = QtWidgets.QWidget()
        self.vlay = QtWidgets.QVBoxLayout(self.button_container)
        self.vlay.setSpacing(20)
        self.vlay.addStretch()
        self.vlay.addWidget(self.shortPathButton)
        self.vlay.addWidget(self.button2)
        self.vlay.addWidget(self.button3)
        self.vlay.addStretch()
        self.lay.addWidget(self.button_container)
        self.lay.addWidget(self.view, stretch=1)

        if not self.map:
            self.map = folium.Map(
                location=[45.5236, -122.6750], tiles="Stamen Toner", zoom_start=13)
            self.tiles.append(folium.TileLayer( tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                              attr = 'Esri', name = 'Satellite', overlay = False, control = True))

            self.tiles.append(folium.TileLayer(tiles='https://tile.thunderforest.com/transport/{z}/{x}/{y}.png?apikey=e22170de87894e0793ce487a27a69ee0', attr = 'Transport Map', name = 'Transport'))

            self.tiles.append(folium.TileLayer(tiles='https://tile.thunderforest.com/transport-dark/{z}/{x}/{y}.png?apikey=e22170de87894e0793ce487a27a69ee0', attr = 'Dark Transport Map', name = 'Dark Transport'))#, name = 'Map Quest Open', overlay = False, control = True))

            self.tiles.append(folium.TileLayer(tiles='https://tile.thunderforest.com/landscape/{z}/{x}/{y}.png?apikey=e22170de87894e0793ce487a27a69ee0', attr = 'Landscape Map', name = 'Landscape'))#, name = 'Aerial', overlay = False, control = True))

            self.tiles.append(folium.TileLayer('openstreetmap', attr = 'osm', name = 'Open Street Map', overlay = False, control = True))

            for tile in self.tiles:
                self.map.add_child(tile)

            self.map.add_child(folium.map.LayerControl())
        
        #self.data = io.BytesIO()
        #self.map.save(self.data, close_file=False)
        #self.view.setHtml(self.data.getvalue().decode())

        url = "C:\\Users\\Omar Ghannou\\Documents\\Reasearch\\thesis\\ZOIs\\ZOIs Identification\\map.html"
        self.map.save(url)

        #webView = QtWebEngineWidgets.QWebEngineView()
        html_map = QtCore.QUrl.fromLocalFile(url)
        self.view.load(html_map)


#if __name__ == "__main__":
#App = QtWidgets.QApplication(sys.argv)
#window = Window()
#window.show()
#sys.exit(App.exec())