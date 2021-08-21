# Transformed-Based Clustering
Sentence-Transformers can be used in different ways to perform clustering of small or large set of sentences. This repository we evaluate 5 baselines based on transformed deep learning.
* Baselines
    * Mass(paraphrase-mpnet-base-v2)
    * Multi Language BERT(distiluse-base-multilingual-cased-v2)
    * XLM-RoBERTa(xlm-r-100langs-bert-base-nli-stsb-mean-tokens)
    * ALBERT (paraphrase-albert-base-v2)
    * BERT(bert-large-nli-stsb-mean-tokens)


## k-Means
[kmeans.py](kmeans.py) contains an example of using [K-means Clustering Algorithm](https://scikit-learn.org/stable/modules/clustering.html#k-means). K-Means requires that the number of clusters is specified beforehand. The sentences are clustered in groups of about equal size.
 
## Agglomerative Clustering
[agglomerative.py](agglomerative.py) shows an example of using [Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) using the [Agglomerative Clustering  Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering). In contrast to k-means, we can specify a threshold for the clustering: Clusters below that threshold are merged. This algorithm can be useful if the number of clusters is unknown. By the threshold, we can control if we want to have many small and fine-grained clusters or few coarse-grained clusters.

## Gaussian Mixture

Agglomerative Clustering for larger datasets is quite slow, so it is only applicable for maybe a few thousand sentences.

In [fast_clustering.py](fast_clustering.py) we present a clustering algorithm that is tuned for large datasets (50k sentences in less than 5 seconds). In a large list of sentences it searches for local communities: A local community is a set of highly similar sentences. 

You can configure the threshold of cosine-similarity for which we consider two sentences as similar. Also, you can specify the minimal size for a local community. This allows you to get either large coarse-grained clusters or small fine-grained clusters. 

We apply it on the [Quora Duplicate Questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset and the output looks something like this:
