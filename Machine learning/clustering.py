import numpy as np

np.random.seed(69420)

class Dbscan:
    '''
    Density-Based Spatial Clustering of Applications with Noise.
    Class that implement DBSCAN algorithm for clustering.
    '''

    def __init__(self, eps=0.5, min_samples=5):
        '''
        Parameters
        ----------
        eps : float
            threshold distance for classification
        min_samples : int
            required number of neighbors
        '''
        self.eps         = eps
        self.min_samples = min_samples
        self.labels      = None
    
    def fit(self, data):
        '''
        Function that cluster the data

        Parameters
        ----------
        data : list
            list of vector (i.e. list of points) that we want to classify
        '''
        
        n_data      = len(data)      # Number of data
        self.labels = [0] * n_data   # Init label
        cluster_id  = 0              # Id of current cluster

        # For each point in the dataset
        for idx in range(n_data):
            
            # Only points that have not already been claimed
            # can be picked as new seed points.
            if self.labels[idx] != 0:
                continue # go to the next point
            
            # Find all Neighbor of point associated to idx
            Neighbor_pts = self.region_query(data, idx)

            # If True the point is noise
            if len(Neighbor_pts) < self.min_samples:
                self.labels[idx] = -1 #noise
            
            # Otherwise use the point as the seed for a new cluster.
            else:
                cluster_id += 1
                self.grow_cluster(data, idx, Neighbor_pts, cluster_id)
        
        # For Ccikit learn comparison
        # Scikit learn uses -1 to for noise,
        # and starts cluster labeling at 0
        for i in range(n_data):
            if self.labels[i] != -1:
                self.labels[i] -= 1
        
        self.labels = np.array(self.labels)
    
    def grow_cluster(self, data, idx, Neighbor_pts, cluster_id):
        '''
        Grow a new cluster

        Parameters
        ----------
        data : list
            list of vector (i.e. list of points) that we want to classify
        idx : int
            index of the seed point
        Neighbor_pts : list
            all neighbor of the point idx
        cluster_id :  int
            label for the cluster
        '''

        # Assign the label to the seed point
        self.labels[idx] = cluster_id

        i = 0
        while i < len(Neighbor_pts):
            
            # Go to the next point 
            next_p = Neighbor_pts[i]

            # If the point is a noise for the seed search
            # it can still be a member of cluster as a leaf
            if self.labels[next_p] == -1:
                self.labels[next_p] = cluster_id
            
            # If next_p isn't already claimed, we assign it to cluster.
            elif self.labels[next_p] == 0:
                self.labels[next_p] = cluster_id

                # Find all Neighbor of point associated to next_p
                next_p_Neighbor_pts = self.region_query(data, next_p)

                # If the point has enough neighbors then it is not a leaf
                # but becomes a branching point of the cluster,
                # so we add all the neighbors found from which we then start again
                if len(next_p_Neighbor_pts) >= self.min_samples:
                    Neighbor_pts = Neighbor_pts + next_p_Neighbor_pts

                # Otherwise, next_p is a leaf point so we do nothing
            
            i += 1 # go to the next point

    def region_query(self, data, idx):
        '''
        Find all points in dataset within distance self.eps of point idx.

        Parameters
        ----------
        data : list
            list of vector (i.e. list of points) that we want to classify
        idx : int
            index of the seed point
        
        Returns
        -------
        neighbors : list
            list of the neighbors of idx
        '''

        neighbors = []

        # For each point in the dataset
        for j in range(len(data)):
            
            # If True we add it to the list
            if np.linalg.norm(data[idx] - data[j]) < self.eps:
                neighbors.append(j)

        return neighbors


class Kmeans:
    '''
    Class for kmeans algorithm for clustering.
    '''

    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        '''
        Parameters
        ----------
        n_cluster : int
            number of claster
        max_iter : int, optional, default 100
            maximum number of iteration
        tol : float
            required tollerance for convergence
        '''
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        self.centroids  = None
        self.labels     = None
    
    def fit(self, data):
        '''
        Function that cluster the data

        Parameters
        ----------
        data : list
            list of vector (i.e. list of points) that we want to classify
        '''
        # Random Initializzation of the centroid
        random_idxs    = np.random.choice(len(data), self.n_clusters, replace=False)
        self.centroids = data[random_idxs]

        for i in range(self.max_iter):
            # Assign point to the nearest centroid
            self.labels = self.assign_clusters(data)
            
            # Compute new centroid
            new_centroids = self.update_centroids(data)
            
            # Convergence criteria
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            # Update centroids
            self.centroids = new_centroids

    def assign_clusters(self, data):
        '''
        Assign each point to the closest centroid

        Parameters
        ----------
        data : list
            list of vector (i.e. list of points) that we want to classify
        
        Return
        ------
        labels : 1darray
            labels for each points
        '''
        labels = []

        # For each point in the dataset
        for point in data:
            # Compute distances with all centroids
            distances = np.linalg.norm(point - self.centroids, axis=1)
            # Assign to the closest
            labels.append(np.argmin(distances))
        
        return np.array(labels)
    
    def update_centroids(self, data):
        '''
        Update the centroids as the average of the points assigned to each cluster

        Parameters
        ----------
        data : list
            list of vector (i.e. list of points) that we want to classify
        
        Return
        ------
        new_centroids: 2darray
            new centroids of each cluster
        '''
        new_centroids = np.zeros((self.n_clusters, data.shape[1]))

        for i in range(self.n_clusters):
            # Select points of i-th cluster
            cluster_points = data[self.labels == i]

            # Compute mean
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            
        return new_centroids
    