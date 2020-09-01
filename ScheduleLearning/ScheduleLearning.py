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
    datetimeObject = datetime.utcfromtimestamp(timestamp)
    return datetimeObject.strftime("%Y-%m-%d")

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
    centroidLocator = NearestCentroid()
    centroidLocator.fit(data, predictions)
    centroids = pd.DataFrame(centroidLocator.centroids_, columns = ["Timestamps", "Latitudes", "Longitudes"])
    for i in range(len(centroids)):
        centroids.iloc[i, 0] = convertToUTC(centroids.iloc[i, 0])
    return centroids

#states = ['STILL', 'IN_VEHICLE', 'ON_FOOT', 'ON_BICYCLE', 'TILTING', 'UNKNOWN', 'EXITING_VEHICLE']
def createDataStructure(data):
    timestamps = []
    latitudes = []
    longitudes = []
    states = []
    allowedStates = ['STILL', 'IN_VEHICLE', 'ON_FOOT', 'ON_BICYCLE', 'TILTING', 'UNKNOWN', 'EXITING_VEHICLE', 'UNDEFINED']
    for i in range(len(data['locations'])):
        entry = data["locations"][i]
        timeToAppend = roundTimestamp(int(entry["timestampMs"][:10]), 60*60)
        latitudeToAppend = entry["latitudeE7"]/(10**7)
        longitudeToAppend = entry["longitudeE7"]/(10**7)
        timestamps.append(timeToAppend)
        latitudes.append(latitudeToAppend)
        longitudes.append(longitudeToAppend)
        try:
            states.append(entry['activity'][0]['activity'][0]['type'])
        except KeyError:
            states.append("UNDEFINED")
    dataStructure = pd.DataFrame(list(zip(timestamps, latitudes, longitudes, states)), columns=['Timestamps', 'Latitudes', 'Longitudes', 'States'])
    dataStructure.sort_values("Timestamps", inplace = True)
    dataStructure = dataStructure[dataStructure['Timestamps'].between(timestampstart, timestampend)]
    dataStructure = dataStructure[dataStructure['States'].isin(allowedStates)]
    dataStructure.drop_duplicates(subset = ["Timestamps"], keep = 'first', inplace = True)
    dataStructure.reset_index(drop=True, inplace=True)
    for i in range(len(dataStructure)):
        timeToCorrect = dataStructure.iloc[i, 0]
        while(timeToCorrect > day8):
            timeToCorrect -= oneWeek
        dataStructure.iloc[i, 0] = timeToCorrect
    dataStructure.reset_index(drop=True, inplace=True)
    return dataStructure, latitudes, longitudes, timestamps

def createAndShowDendogram(data):
    dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Points')
    plt.ylabel('Euclidean distances')
    plt.show()

def createModelAndPredict(data, clusters):
    model = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')
    modelPredictions = model.fit_predict(data)
    return model, modelPredictions

import plotly.express as px
def plotandShow3dGraph(data, predictions):
    fig = px.scatter_3d(data, x=dates, y='Latitudes', z='Longitudes', color=predictions)
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    centroidPlot = px.scatter_3d(centroids, x='Timestamps', y='Latitudes', z='Longitudes')
    centroidPlot.update_traces(marker=dict(size=20, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.add_trace(centroidPlot.data[0])
    fig.update_layout(font_family="Akrobat", title_font_family="Akrobat", hoverlabel=dict(font_family="Akrobat"), scene=dict(xaxis_title='Dates'))
    fig.show()

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
filePath = "TakeoutSR/Location History/Location History.json"
data = loadData(filePath)
startDate = datetime(2014, 6, 1, 0, 0, 0)
endDate = datetime(2014, 7, 1, 0, 0, 0)
timestampstart = convertToTimestamp(startDate)
timestampend = convertToTimestamp(endDate)
day1 = roundTimestamp(int(timestampstart), 24*60*60)
oneWeek = 6*24*60*60
day8 = day1 + oneWeek
X, latitudes, longitudes, timestamps = createDataStructure(data)
X_np = X.iloc[:, :-1].values
#createAndShowDendogram(X_np)
model, predictions = createModelAndPredict(X_np, 2)
centroids = getCentroids(X_np, predictions)
dates = X.iloc[:, 0]
for i in range(len(dates)):
    dates.iloc[i] = convertToUTC(dates.iloc[i])

plotandShow3dGraph(X, predictions)
createMap("hc_map_SR_test.html", latitudes, longitudes)
for i in range(len(centroids)):
    print(getAddress(centroids[i][1], centroids[i][2]))
