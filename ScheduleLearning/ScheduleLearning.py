# Importing all the required modules
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import gmplot
from sklearn.neighbors import NearestCentroid

def loadData(path):
    with open(path) as f:
        data = json.load(f)
    return data
    print("JSON Load Successful!")

def convertToTimestamp(datetimeObject):
    return datetimeObject.replace(tzinfo=timezone.utc).timestamp()

def convertToUTC(timestamp):
    return datetime.utcfromtimestamp(timestamp)

def getMeanFromList(values):
    mean = sum(values)/len(values)
    return mean

def roundTimestamp(dt_series, seconds=60, up=False):
    return dt_series // seconds * seconds + seconds * up

def getAddress(latitude, longitude):
    coordinates = str(latitude) + ", " + str(longitude)
    location = locator.reverse(coordinates)
    return location.address

def getCentroids(data, predictions):
    clf = NearestCentroid()
    clf.fit(data, predictions)
    return clf.centroids_

#states = ['STILL', 'IN_VEHICLE', 'ON_FOOT', 'ON_BICYCLE', 'TILTING', 'UNKNOWN', 'EXITING_VEHICLE']
def createDataStructure(data):
    latitudes = []
    longitudes = []
    timestamps = []
    allowedStates = ['STILL']
    transitionalState = False
    for i in range(len(data['locations'])):
        entry = data["locations"][i]
        try:
            if(entry['activity'][0]['activity'][0]['type'] not in allowedStates):
                transitionalState = True
            else:
                transitionalState = False
        except:
            pass
        timeToAppend = roundTimestamp(int(entry["timestampMs"][:10]), 60*60)
        latitudeToAppend = entry["latitudeE7"]/(10**7)
        longitudeToAppend = entry["longitudeE7"]/(10**7)
        while(timeToAppend > day8):
            timeToAppend -= oneWeek
        if(len(latitudes) == 0 and len(timestamps) == 0 and timeToAppend is not None and not transitionalState):
            timestamps.append(timeToAppend)
            latitudes.append(latitudeToAppend)
            longitudes.append(longitudeToAppend)
        else:
            try:
                if(timeToAppend > timestamps[-1] and not transitionalState):
                    timestamps.append(timeToAppend)
                    latitudes.append(latitudeToAppend)
                    longitudes.append(longitudeToAppend)
            except IndexError:
                print("No Values that match criteria!")
                pass
    dataStructure = np.array(list(zip(timestamps, latitudes, longitudes)))
    return dataStructure, latitudes, longitudes, timestamps

def createAndShowDendogram(data):
    dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Points')
    plt.ylabel('Euclidean distances')
    plt.show()

def createModelAndPredict(data, clusters):
    model = AgglomerativeClustering(n_clusters = clusters, affinity = 'euclidean', linkage = 'ward')
    modelPredictions = model.fit_predict(X)
    return model, modelPredictions



#import plotly.plotly as py
import plotly.graph_objs as go
from plotly import offline
offline.init_notebook_mode()

def plotandShow3dGraph(data, predictions):

    scatter_data = [go.Scatter3d(x=timestamps, y=latitudes, z=longitudes)]

    layout = go.Layout(
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(
                spikecolor='#1fe5bd',
                spikesides=False,
                spikethickness=6,
            ),
            yaxis=go.layout.scene.YAxis(
                spikecolor='#1fe5bd',
                spikesides=False,
                spikethickness=6,
            ),
            zaxis=go.layout.scene.ZAxis(
                spikecolor='#1fe5bd',
                spikethickness=6,
            ),
        ),
    )
    fig = go.Figure(data=scatter_data, layout=layout)
    fig.update_layout(scene = dict(
                        xaxis_title='Timestamps',
                        #xaxis = dict(range=[timestampstart,timestampend]),
                        yaxis_title='Latitudes',
                        zaxis_title='Longitudes'),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10))
    offline.iplot(fig)

    fignum = 1
    fig = plt.figure(fignum, figsize=(10, 9))
    colors = ['red', 'blue', 'yellow', 'violet']
    centroid_color = "green"
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=50)
    for i in range(len(centroids)):
        ax.scatter(data[predictions == i, 0], data[predictions == i, 1], data[predictions == i, 2], s=5, c=colors[i], label = 'Cluster ' + str(i+1), picker=True)
        ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], s=500, c=centroid_color, label = 'Centroids')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Time')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Longitude')
    ax.set_title('Location over time')
    ax.dist = 10
    ax.legend()

    plt.show()

def createMap(filePath, latitudes, longitudes):
    zoom = 10
    meanLatitude = getMeanFromList(latitudes)
    meanLongitude = getMeanFromList(longitudes)
    gmap3 = gmplot.GoogleMapPlotter(meanLatitude, meanLongitude, zoom)
    gmap3.scatter(latitudes, longitudes, 'cornflowerblue', size = 1000, marker = False)
    for i in range(len(centroids)):
        gmap3.scatter([float(centroids[i][1])], [float(centroids[i][2])], 'green', size = 5000, marker = False)
    gmap3.draw(filePath)

locator = Nominatim(user_agent='google')
filePath = "TakeoutAR/Location History/Location History.json"
data = loadData(filePath)
len(data['locations'])
state = ['']
for i in range(len(data['locations'])):
    try:
        if(data['locations'][i]['activity'][0]['activity'][0]['type'] not in state):
            state.append(data['locations'][i]['activity'][0]['activity'][0]['type'])
    except:
        pass

startDate = datetime(2018, 6, 1, 0, 0, 0)
endDate = datetime(2018, 7, 1, 0, 0, 0)
timestampstart = convertToTimestamp(startDate)
timestampend = convertToTimestamp(endDate)
day1 = roundTimestamp(int(timestampstart), 24*60*60)
oneWeek = 7*24*60*60
day8 = day1 + oneWeek
X_init, latitudes_init, longitudes_init, timestamps = createDataStructure(data)
X_init[0]
X = []
latitudes = []
longitudes = []
for i in range(len(X_init)):
    if(X_init[i][0] > timestampstart and X_init[i][0] < timestampend):
        X.append(X_init[i])
        latitudes.append(X_init[i][1])
        longitudes.append(X_init[i][2])
X = np.array(X)

convertToUTC(X_init[0][0])
convertToUTC(X_init[-1][0])
len(X)
#convertToUTC(X[-1][0])
#convertToUTC(int(data['locations'][0]['timestampMs'][:10]))
createAndShowDendogram(X)
model, predictions = createModelAndPredict(X, 2)
centroids = getCentroids(X, predictions)
plotandShow3dGraph(X, predictions)
createMap("hc_map_2020_07_2020_08_AR_test.html", latitudes, longitudes)
X[-1]
for i in range(len(centroids)):
    print(getAddress(centroids[i][1], centroids[i][2]))
