# K-Means
import matplotlib
import matplotlib.pyplot as plot
import pandas
from scipy.cluster.hierarchy import linkage, dendrogram

matplotlib.use("TkAgg")
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
import warnings

warnings.filterwarnings("ignore")
df = pandas.read_csv("Dataset/USArrests.csv", index_col=0)
df.head()
df.isnull().sum()
df.info
df.describe().T

scaler = MinMaxScaler((0, 1))
df = scaler.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

kmeans.inertia_

# Optimum küme sayısı belirleme
kmeans = KMeans()
ssd = []
K = range(1, 30)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)
plot.plot(K, ssd, "bx-")
plot.xlabel("Farklı")
plot.show(block=True)

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_

# Final Cluster Oluşturma

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters = kmeans.labels_
df = pandas.read_csv("Dataset/USArrests.csv", index_col=0)
df["cluster"] = clusters
df.head()
df["cluster"] = df["cluster"] + 1
df.head()

df[df["cluster"] == 6]

df.groupby("cluster").agg(["count", "mean", "median"])

# Hierachical Cluster

df = pandas.read_csv("Dataset/USArrests.csv", index_col=0)
scaler = MinMaxScaler((0, 1))
df = scaler.fit_transform(df)

hc_average = linkage(df, "average")

plot.figure(figsize=(10, 5))
plot.title("Hiyerarşik Kümeleme Dendogramı")
plot.xlabel("Gözlem Birimleri")
plot.ylabel("Uzaklıklar")
dendrogram(hc_average, leaf_font_size=10)
plot.show(block=True)

plot.figure(figsize=(10, 5))
plot.title("Hiyerarşik Kümeleme Dendogramı")
plot.xlabel("Gözlem Birimleri")
plot.ylabel("Uzaklıklar")
dendrogram(hc_average, truncate_mode="lastp", p=10, show_contracted=True, leaf_font_size=10)
plot.show(block=True)

# Final model oluşturma

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
clusters = cluster.fit_predict(df)
