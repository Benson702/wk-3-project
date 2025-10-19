import streamlit as st
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

# --- Title and description ---
st.title("🚍 Sustainable Cities: Public Transport Route Optimization")
st.markdown("""
This demo uses **K-Means clustering** to group nearby bus stops
and optimize potential transport routes for sustainable urban planning.
""")

# --- Simulated bus stop data ---
np.random.seed(42)
data = pd.DataFrame({
    "stop_id": range(1, 101),
    "latitude": np.random.uniform(-1.3, -1.1, 100),
    "longitude": np.random.uniform(36.7, 36.9, 100),
    "passengers": np.random.randint(10, 200, 100)
})

# --- Sidebar controls ---
st.sidebar.header("🔧 Settings")
n_clusters = st.sidebar.slider("Number of Clusters (Routes)", 2, 15, 5)

# --- K-Means clustering ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data["cluster"] = kmeans.fit_predict(data[["latitude", "longitude"]])

cluster_summary = (
    data.groupby("cluster")
    .agg({
        "latitude": "mean",
        "longitude": "mean",
        "passengers": ["sum", "mean", "count"]
    })
    .reset_index()
)
cluster_summary.columns = ["cluster", "latitude", "longitude", "total_passengers", "avg_passengers", "num_stops"]

# --- Visualization ---
m = folium.Map(location=[data.latitude.mean(), data.longitude.mean()], zoom_start=12)
colors = ["red", "blue", "green", "orange", "purple", "pink", "gray", "brown"]

# plot stops
for _, row in data.iterrows():
    folium.CircleMarker(
        [row.latitude, row.longitude],
        radius=4,
        color=colors[int(row.cluster) % len(colors)],
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# plot cluster centers
for _, row in cluster_summary.iterrows():
    folium.Marker(
        [row.latitude, row.longitude],
        popup=f"🚏 Cluster {int(row.cluster)}<br>🧍 Demand: {int(row.total_passengers)} passengers",
        icon=folium.Icon(color="green", icon="bus", prefix="fa")
    ).add_to(m)

# --- Display the map ---
st.subheader("🗺️ Optimized Public Transport Routes")
st_folium(m, width=700, height=500)

# --- Show summary ---
st.subheader("📊 Cluster Summary")
st.dataframe(cluster_summary)
