'''

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from clustering import Dbscan, Kmeans

fig = plt.figure(1)
"""
plt.suptitle("DBSCAN clustering")
ax1 = fig.add_subplot(221)
ax1.set_title("our implementation")
ax1.set_ylabel("y")
ax2 = fig.add_subplot(222)
ax2.set_title("sklearn")
ax3 = fig.add_subplot(223)
ax3.set_ylabel("y")
ax3.set_xlabel("x")
ax4 = fig.add_subplot(224)
ax4.set_xlabel("x")
"""
ax1 = fig.add_subplot(121)
ax1.set_title("our implementation")
ax1.set_ylabel("y")
ax1.set_xlabel("x")
ax2 = fig.add_subplot(122)
ax2.set_title("sklearn")
ax2.set_xlabel("x")


def plot(ax, X, l):
    # Coloriamo i punti in base alle etichette dei cluster
    unique_labels = set(l)

    # Definiamo un colore per ogni cluster (differenti cluster avranno colori distinti)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # I punti etichettati come rumore saranno neri
            col = [0, 0, 0, 1]

        class_member_mask = (l == k)
        
        # Plotta i core points (grandi punti)
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k')

#========================================================================================
# Test dbscan with blobs
#========================================================================================

X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.6, random_state=0)
#X    = StandardScaler().fit_transform(X)

db = Dbscan(eps=0.3, min_samples=10)
db.fit(X)
my_labels = db.labels

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_labels = db.labels_

print(f"Difference with scikit learn : {np.linalg.norm(my_labels - skl_labels)}")
plot(ax1, X, my_labels)
plot(ax2, X, skl_labels)
"""
#========================================================================================
# Test dbscan with moons
#========================================================================================

X, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
X    = StandardScaler().fit_transform(X)

db = Dbscan(eps=0.3, min_samples=10)
db.fit(X)
my_labels = db.labels

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_labels = db.labels_

print(f"Difference with scikit learn : {np.linalg.norm(my_labels - skl_labels)}")

plot(ax3, X, my_labels)
plot(ax4, X, skl_labels)

#========================================================================================

fig = plt.figure(2)
plt.suptitle("Kmeans clustering")
ax1 = fig.add_subplot(221)
ax1.set_title("our implementation")
ax1.set_ylabel("y")
ax2 = fig.add_subplot(222)
ax2.set_title("sklearn")
ax3 = fig.add_subplot(223)
ax3.set_ylabel("y")
ax3.set_xlabel("x")
ax4 = fig.add_subplot(224)
ax4.set_xlabel("x")

#========================================================================================
# Test Kmeans with bloobs
#========================================================================================

X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.4, random_state=0)
X    = StandardScaler().fit_transform(X)

km = Kmeans(n_clusters=4)
km.fit(X)
my_labels = km.labels

km = KMeans(n_clusters=4).fit(X)
skl_labels = km.labels_

print(f"Difference with scikit learn : {np.linalg.norm(my_labels - skl_labels)}")
plot(ax1, X, my_labels)
plot(ax2, X, skl_labels)

#========================================================================================
# Test Kmeans with moons
#========================================================================================

X, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
X    = StandardScaler().fit_transform(X)

km = Kmeans(n_clusters=2)
km.fit(X)
my_labels = km.labels


km = KMeans(n_clusters=2).fit(X)
skl_labels = km.labels_

print(f"Difference with scikit learn : {np.linalg.norm(my_labels - skl_labels)}")
plot(ax3, X, my_labels)
plot(ax4, X, skl_labels)
"""
plt.show()   