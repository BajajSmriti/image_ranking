# Spectral Clustering
We sought inspiration from Google’s reverse image search wherein the query is in the form of an image and the output is a set of relevant images. To decide the relevance or similarity among these images, we chose to rank them based on two methods. First, we extract the features from the training dataset and the query image using the VGG-16 (Simonyan and Zisserman) pre-trained model. With the features of the training image set, we create a Laplacian matrix and generate eigenvectors, with which we perform Spectral Clustering to get the cluster labels and their respective centroids. 
The query image features are then measured against the centroids using the L2 norm or Euclidean distance. From these distances, we chose the cluster corresponding to the minimum distance. That is the cluster where our query image is expected to belong. We again took the Euclidean distance between the query image and the feature vectors of the images belonging to that cluster. Then, we finally ranked these images on the basis of the non-decreasing distances. Optionally, we can repeat this process to achieve stable clusters and get definite ranks. 
We have deployed the algorithm on the localhost for ranking real-time images using a Python-based web framework, Flask. It has helped to implement a bare-minimum web server which is useful for the end users to search from a database of ever-growing Flickr images.

## Dependencies
1.Python 3.7+

## How to run
1. python feature_extractor.py
2. python data_cluster.py
3. python test.py
4. python main.py (Administrator because of Flask)

