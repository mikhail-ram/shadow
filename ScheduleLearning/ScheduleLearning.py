import json
from datetime import datetime, timezone
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import geopy
from geopy.geocoders import Nominatim

locator = Nominatim(user_agent='google')
def get_address(lat, long):
    coordinates = str(lat) + ", " + str(long)
    location = locator.reverse(coordinates)
    return location.address

#place your history file in your project deirectory
with open("Takeout/Location History/Location History.json") as f:
    data = json.load(f)
    print("JSON Load Successful!")


dtstart = datetime(2020, 7, 1, 1, 0, 0)
dtend = datetime(2020, 8, 1, 1, 0, 0)
timestampstart = dtstart.replace(tzinfo=timezone.utc).timestamp()
timestampend = dtend.replace(tzinfo=timezone.utc).timestamp()
#print(timestamp)

lats = []
longs = []
tmstps = []

def round_unix_date(dt_series, seconds=60, up=False):
    return dt_series // seconds * seconds + seconds * up

#retrieving the lats and longs
for entry in data["locations"]:
    if(len(lats) == 0 and int(entry["timestampMs"][:10]) > timestampstart and int(entry["timestampMs"][:10]) < timestampend):
        tmstps.append(round_unix_date(int(entry["timestampMs"][:10]), 60*60))
        lats.append(entry["latitudeE7"]/(10**7))
        longs.append(entry["longitudeE7"]/(10**7))
    else:
        if(int(entry["timestampMs"][:10]) > timestampstart and int(entry["timestampMs"][:10]) < timestampend and round_unix_date(int(entry["timestampMs"][:10]), 60*60) > tmstps[-1]):
            tmstps.append(round_unix_date(int(entry["timestampMs"][:10]), 60*60))
            lats.append(entry["latitudeE7"]/(10**7))
            longs.append(entry["longitudeE7"]/(10**7))
            #time = datetime.utcfromtimestamp(ts)

X = np.array(list(zip(tmstps, lats, longs)))

#print(X)
#print(data['locations'][0]['latitudeE7']/(10**7))

'''import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Points')
plt.ylabel('Euclidean distances')
plt.show()'''

from sklearn.cluster import AgglomerativeClustering
clusters = 2
hc = AgglomerativeClustering(n_clusters = clusters, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X, y_hc)
centroids = clf.centroids_

# Plot the 3D representation
fignum = 1
fig = plt.figure(fignum, figsize=(10, 9))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=50)

ax.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], X[y_hc == 0, 2], s=5, c='red', label = 'Cluster 1')
ax.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], X[y_hc == 1, 2], s=5, c='blue', label = 'Cluster 2')
#ax.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], X[y_hc == 2, 2], s=5, c='yellow', label = 'Cluster 3')
#ax.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], X[y_hc == 3, 2], s=5, c='violet', label = 'Cluster 4')
for i in range(clusters):
    ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], s=500, c='green', label = 'Centroids')
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
for i in range(clusters):
    print(get_address(centroids[i][1], centroids[i][2]))
#print(get_address(hc.cluster_centers_[3, 1], hc.cluster_centers_[3, 2]))

#hc.cluster_centers_[2, 1]
#hc.cluster_centers_[2, 2]
#import plotly.express as px
#px.set_mapbox_access_token(open(".mapbox_token").read())
#coords = pd.DataFrame(hc.cluster_centers_, columns=['time', 'lat', 'long'])
#map = px.scatter_mapbox(coords, lat='lat', lon='long', size_max=15, zoom=10)
#map.update_layout(mapbox_style="open-street-map")
#map.show()

# import gmplot package
import gmplot

gmap3 = gmplot.GoogleMapPlotter(sum(lats)/len(lats), sum(longs)/len(longs), 7)

# scatter method of map object
# scatter points on the google map
gmap3.scatter(lats, longs, 'cornflowerblue', size = 2000, marker = False)
for i in range(clusters):
    gmap3.scatter([float(centroids[i][1])], [float(centroids[i][2])], 'green', size = 20000, marker = False)
gmap3.draw("hc_map_2020_07_2020_08_MR.html")
