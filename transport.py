# import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
import random
from math import radians, sin, cos, sqrt, atan2
import os
# generate synthetic data 
num_stops = 300
stops = pd.DataFrame({
    "stop_id": range(num_stops),
    "latitude": np.random.uniform(-1.30, -1.20, num_stops),
    "longitude": np.random.uniform(36.75, 36.90, num_stops)
})
# simulating ridership data
ridership = pd.DataFrame({
    "stop_id": np.random.choice(stops["stop_id"], 1000, replace=True),
    "hour": np.random.randint(5, 23, 1000),
    "passengers": np.random.randint(1, 60, 1000)
})
# merge stop $ ridership data
# Merge stop and ridership data
data = pd.merge(
    stops,
    ridership.groupby("stop_id", as_index=False).agg({"passengers": "mean"}),
    on="stop_id"
)
data.rename(columns={"passengers": "avg_passengers"}, inplace=True)

# clustring bus stop (K-Means)
n_clusters = 25
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data["cluster"] = kmeans.fit_predict(data[["latitude", "longitude"]])
# aggregating cluster-level demand
cluster_summary = data.groupby("cluster").agg({
    "latitude": "mean",
    "longitude": "mean",
    "avg_passengers": ["sum", "mean", "count"]
}).reset_index()

cluster_summary.columns = ["cluster", "latitude", "longitude", "total_passengers", "avg_passengers", "num_stops"]
# Help function to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
# create simplofied route
def create_routes(clusters, route_size=5):
    clusters = clusters.sort_values("total_passengers", ascending=False).reset_index(drop=True)
    routes, visited = [], set()

    for i in range(0, len(clusters), route_size):
        route_clusters = clusters.iloc[i:i+route_size]["cluster"].tolist()
        routes.append(route_clusters)
        visited.update(route_clusters)
    return routes

routes = create_routes(cluster_summary, route_size=5)
# visuliation with folium map
m = folium.Map(location=[data.latitude.mean(), data.longitude.mean()], zoom_start=12)
colors = ["red", "blue", "green", "orange", "purple", "pink", "gray", "brown"]

for _, row in data.iterrows():
    folium.CircleMarker([row.latitude, row.longitude],
                        radius=3,
                        color = colors[int(row.cluster) % len(colors)],
                        fill=True).add_to(m)

for _, row in cluster_summary.iterrows():
    folium.Marker([row.latitude, row.longitude],
                  popup=f"Cluster {row.cluster}, Demand: {row.total_passengers}").add_to(m)

m.save("transport_routes_map.html")
# saving outputs
os.makedirs("output", exist_ok=True)
data.to_csv("output/stops_with_clusters.csv", index=False)
cluster_summary.to_csv("output/clusters_summary.csv", index=False)

