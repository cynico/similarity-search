from typing import Dict, List, Annotated
import numpy as np
from npy_append_array import NpyAppendArray
from math import ceil, floor 
from sklearn.cluster import MiniBatchKMeans
import os
from joblib import dump, load
from shutil import rmtree
import hnswlib
import tempfile
import uuid

verbose = False
def printIfVerbose(message):
    if (verbose): print(message)

class VecDB:
    def __init__(self, new_db = True, file_path=".") -> None:

        # If no file_path is passed, default file_path is current directory.
        # If that is the case, create a temporary directory and make it the parent directory for the db.
        if file_path == ".": file_path = tempfile.mkdtemp()

        # File paths
        self.parent_path = file_path
        self.data_path = f"{self.parent_path}/data"
        self.metadata_path = f"{self.parent_path}/metadata"

        self.monolith_file_path = f"{self.data_path}/monolith"
        self.sample_path = f"{self.data_path}/clustering_sample.npy"
        self.centroids_path = f"{self.data_path}/centroids"
        self.kmeans_path = f"{self.data_path}/kmeans"
        self.regions_path = f"{self.data_path}/regions"
        self.temp_regions_path = f"{self.data_path}/new-regions"

        self.num_vectors_db_path = f"{self.metadata_path}/num_vectors_db.txt"
        self.num_vectors_clustered_path = f"{self.metadata_path}/num_vectors_clustered.txt"

        # Ratios and parameters
        self.vectors_per_cluster = 500 # How many vectors per cluster
        
        self.nprobe_ratio = 0.05 # Controls the number of clusters with closest centroids to search during query retrieval.
        self.minimum_clusters_to_search = 40 # Minimum number of clusters to search during query retrieval.
        self.maximum_clusters_to_search = 500 # Maximum number of clusters to search during query retrieval.

        self.clustering_ratio_threshold = 1.5 # Controls when the re-clustering should happen. 
        self.hnsw_efConstruction_ratio = 0.2 # An HNSW-related graph construction parameter.
        self.hnsw_efSearch_ratio = 4 # An HNSW-related graph query parameter. This (roughly) controls how many nearest neighbour to search.

        if new_db:

            if os.path.exists(self.parent_path):
                if os.path.isdir(self.parent_path): rmtree(self.parent_path)
                else: os.remove(self.parent_path)

            for directory in [self.parent_path, self.metadata_path, self.data_path, self.monolith_file_path]: os.mkdir(directory)
            with open(self.num_vectors_db_path, "w") as file: file.write("0\n")

    # def __del__(self):
    #     if os.path.exists(self.parent_path):
    #         if os.path.isdir(self.parent_path): rmtree(self.parent_path)
    #         else: os.remove(self.parent_path)

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]], build_index=True):
        
        chunk_size = 5*(10**5)
        num_iterations = ceil(len(rows)/chunk_size)

        for i in range(num_iterations):
            start = i * chunk_size
            end = (i+1) * chunk_size if (i != num_iterations - 1) else len(rows)
            
            vectors = np.array([np.array([row["id"]] + row["embed"]) for row in rows[start:end]])
            with NpyAppendArray(f"{self.monolith_file_path}/{str(uuid.uuid4())}") as file:
                file.append(vectors)

        # Update the present number of vectors in the database
        with open(self.num_vectors_db_path, "r+") as file:
            present_vec_num = int(file.readline()) + len(rows)
            file.seek(0)
            file.write(f"{present_vec_num}\n")
        
        # We can delay building the index if we are iteratively inserting chunks of records, 
        # by setting the parameter build_index to False
        if build_index: self._build_index()

    # Returns a list of vectors loaded from the given list of files.
    # isGraph indicates whether those files are graph files (during reclustering), or npy files (first time
    # clustering).
    def get_vectors_from_files(self, files: List[str], isGraph: bool) -> np.ndarray:
        
        data = np.ndarray(shape=(0, 71))
        for file in files:
            if isGraph:
                regionGraph = load(file)
                regionVectors = np.append(np.array(regionGraph.get_ids_list()).reshape((regionGraph.element_count, 1)), regionGraph.get_items(regionGraph.get_ids_list()), axis=1)
                data = np.append(data, regionVectors, axis=0)
            else:
                data = np.append(data, np.load(file), axis=0)

        return data

    # Return a list of list of files to use the vectors within for clustering.
    # This is supposed to hide the implementation detail of whether we are clustering for the first time
    # and thus from the files under data/monolith, or reclustering with files of existing regions under
    # data/regions. It tries to return files containing up to 5mil vectors, or the whole dataset if its size is 
    # less than 5mil.
    # Every list has files containing around 500k vectors, used as one batch in MiniBatchKMeans
    def get_files_for_clustering(self, reclustering: bool, predicting: bool) -> List[List[str]]:
        
        batch_size = 5*(10**5)
        batches = 10
        files = None

        if reclustering:

            num_vectors_db = 0
            with open(self.num_vectors_db_path, "r") as file: num_vectors_db = int(file.readline())

            num_existing_regions = len(os.listdir(self.regions_path))
            
            # Approximately how many /existing/ vectors per region
            true_vectors_per_cluster = floor(num_vectors_db / num_existing_regions)

            if predicting:
                files = [ f"{self.regions_path}/{file}" for file in os.listdir(self.regions_path) ]
                files = [ arr.tolist() for arr in np.array_split(files, ceil(num_existing_regions / ceil(batch_size/true_vectors_per_cluster))) ]
            else:

                # Regions per 5mil
                regions_per_5mil = ceil((batches*batch_size) / true_vectors_per_cluster)
                used_regions = min(regions_per_5mil, num_existing_regions)

                files = np.random.choice(os.listdir(self.regions_path), size=used_regions, replace=False).tolist()
                files = [ f"{self.regions_path}/{file}" for file in files ]

                if used_regions == num_existing_regions:
                    files = [ arr.tolist() for arr in np.array_split(files, ceil(used_regions / ceil(batch_size/true_vectors_per_cluster))) ]
                else:
                    files = [ arr.tolist() for arr in np.array_split(files, ceil(len(files)/10)) ]

        else:
            if predicting:
                files = os.listdir(self.monolith_file_path)
            else:
                # Maximum of 10 files (10*0.5mil = 5mil vectors), or the dataset size.
                files = np.random.choice(os.listdir(self.monolith_file_path), size=min(batches, len(os.listdir(self.monolith_file_path))), replace=False).tolist()

            files = [ [ f"{self.monolith_file_path}/{file}"] for file in files ]

        return files

    def retrive(self, query: Annotated[List[float], 70], top_k = 5):

        query = np.reshape(query, newshape=(70,))

        kmeans : MiniBatchKMeans = load(self.kmeans_path)
        nprobe = ceil(self.nprobe_ratio * kmeans.cluster_centers_.shape[0])
        
        # Probing self.nprobe of the clusters, with a lower limit of self.minimum_clusters_to_search and an upper limit of self.maximum_clusters_to_search
        nprobe = min(max(self.minimum_clusters_to_search, min(nprobe, self.maximum_clusters_to_search)), kmeans.cluster_centers_.shape[0])

        # Get the closest nprobe centroids
        printIfVerbose(f"Probing {nprobe} clusters out of {kmeans.cluster_centers_.shape[0]} clusters")
        centroidsGraph = load(self.centroids_path)
        centroidsGraph.set_ef(int(self.hnsw_efSearch_ratio * nprobe))
        closest_centroids, _ = centroidsGraph.knn_query(query, k = nprobe)
        del centroidsGraph

        closest_vectors = np.ndarray(shape=(1,))
        closest_vectors_distances = np.full(shape=(1,), fill_value=float("+inf"))

        for centroid in closest_centroids[0]:
            
            regionGraph = load(f"{self.regions_path}/{centroid}")
            regionGraph.set_ef( int(self.hnsw_efSearch_ratio * top_k) )
            labels, distances = regionGraph.knn_query(query, k = top_k)

            # Appending the k closest distances and vectors to the existing np arrays.
            closest_vectors_distances = np.append(closest_vectors_distances, distances) # now it contains 2*top_k
            closest_vectors = np.append(closest_vectors, labels)

            # Choosing k closest out of 2*k vectors. mergesort is better because every k is already sorted.
            t = np.argsort(closest_vectors_distances, kind='mergesort')[:top_k]
            closest_vectors = closest_vectors[t]
            closest_vectors_distances = closest_vectors_distances[t]

        return closest_vectors.astype('int').tolist()

    def _build_index(self):
       
        reclustering : bool = False
        num_vectors_clustered : int
        num_vectors_db : int

        with open(self.num_vectors_db_path, "r") as file: num_vectors_db = int(file.readline())

        # If previously clustered
        if os.path.exists(self.num_vectors_clustered_path):

            with open(self.num_vectors_clustered_path, "r") as file: num_vectors_clustered = int(file.readline())

            kmeans : MiniBatchKMeans = load(self.kmeans_path)

            # If there exists some data that do not yet belong to any region, cluster them
            for file in os.listdir(self.monolith_file_path):

                unclusteredData = np.load(f"{self.monolith_file_path}/{file}")
                prediction = kmeans.predict(unclusteredData[:,1:])

                for centroid_index in range(len(kmeans.cluster_centers_)):

                    belonging_samples = np.where(prediction == centroid_index)[0]
                    
                    # If there are no samples belonging to this region
                    if belonging_samples.shape[0] == 0: continue

                    # Load the region graph
                    regionGraph : hnswlib.Index = load(f"{self.regions_path}/{centroid_index}")

                    # If the currenet graph capacity is not enough to add the new vectors, increase the graph size
                    required_size = regionGraph.element_count + belonging_samples.shape[0] 
                    if (required_size  > regionGraph.max_elements): regionGraph.resize_index(required_size)

                    # Add the items to the graph
                    regionGraph.add_items(data=unclusteredData[belonging_samples, 1:], ids=unclusteredData[belonging_samples, 0])
                    dump(regionGraph, f"{self.regions_path}/{centroid_index}")

                os.remove(f"{self.monolith_file_path}/{file}")

            # If the ratio constraint is violated, re-cluster
            if (num_vectors_db / num_vectors_clustered) >= self.clustering_ratio_threshold:
                printIfVerbose("Low threshold ratio violated. Reclustering..")                  
                reclustering = True
            else:  
                printIfVerbose("No need to recluster now, low threshold ratio not violated. Appending to existing regions..")
                return
        else:
            printIfVerbose("First time clustering..")
        
        #### 
        # If this is the first time indexing, or the ratio of vectors involved
        # in clustering in the current index to the vectors currently
        # present in the database in smaller than a certain threshold: redo the clustering
        ####

        ncentroids = ceil(num_vectors_db / self.vectors_per_cluster)
        batch_size = 5*(10**5)
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=ncentroids, verbose=verbose, n_init=1, batch_size=batch_size, compute_labels=False, reassignment_ratio=0.01)
        
        files = self.get_files_for_clustering(reclustering=reclustering, predicting=False)
        for files_list in files:
            data = self.get_vectors_from_files(files=files_list, isGraph=reclustering)
            printIfVerbose(f"Clustering vectors from file(s) {files_list}. Data shape = {data.shape}")
            kmeans = kmeans.partial_fit(data[:, 1:])

        # Saving kmeans model to disk, for use later when inserting new records.
        if os.path.exists(self.kmeans_path): os.remove(self.kmeans_path)
        dump(kmeans, self.kmeans_path)

        # Creating and saving the centroids graph to disk.
        centroidsGraph = hnswlib.Index(space='cosine', dim=70)
        max_elements = kmeans.cluster_centers_.shape[0]
        centroidsGraph.init_index(max_elements = max_elements, ef_construction = int(self.hnsw_efConstruction_ratio * max_elements), M = 16)
        centroidsGraph.add_items(data=kmeans.cluster_centers_, ids=range(max_elements))
        if os.path.exists(self.centroids_path): os.remove(self.centroids_path)
        dump(centroidsGraph, self.centroids_path)
        del centroidsGraph

        # Create num_vectors_clustered (holding the number of vectors at the time of clustering)
        with open(self.num_vectors_clustered_path, "w") as file: file.write(f"{num_vectors_db}\n")

        # Predict the clusters of all data points and create region files.
        os.mkdir(self.temp_regions_path)
        for files_list in self.get_files_for_clustering(reclustering=reclustering, predicting=True):
            
            data = self.get_vectors_from_files(files=files_list, isGraph=reclustering)
            prediction = kmeans.predict(data[:, 1:])
            for centroid_index in range(len(kmeans.cluster_centers_)):

                belonging_samples = np.where(prediction == centroid_index)[0]
                if belonging_samples.shape[0] == 0: continue

                # Construct the HNSW graph for this region
                if os.path.exists(f"{self.temp_regions_path}/{centroid_index}"):
                    regionGraph = load(f"{self.temp_regions_path}/{centroid_index}")
                    required_size = regionGraph.element_count + belonging_samples.shape[0] 
                    if (required_size  > regionGraph.max_elements): regionGraph.resize_index(required_size)
                    os.remove(f"{self.temp_regions_path}/{centroid_index}")
                else:
                    regionGraph = hnswlib.Index(space='cosine', dim=70)
                    max_elements = max(self.vectors_per_cluster, belonging_samples.shape[0])
                    regionGraph.init_index(max_elements = max_elements, ef_construction = int(self.hnsw_efConstruction_ratio * max_elements), M = 16)
                
                regionGraph.add_items(data=data[belonging_samples, 1:], ids=data[belonging_samples, 0])
                # Dumping the graph to the disk
                dump(regionGraph, f"{self.temp_regions_path}/{centroid_index}")
                del regionGraph

            for file in files_list: os.remove(file)

        # Deleting the old region directory. rmtree() instead of rmdir() for extra caution,
        # even though region files should've been already deleted above.
        if os.path.exists(self.regions_path): rmtree(self.regions_path)
        os.rename(self.temp_regions_path, self.regions_path)

        printIfVerbose("Finished building index")
