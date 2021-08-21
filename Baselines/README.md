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
[agglomerative.py](agglomerative.py) shows an example of using [Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) using the [Agglomerative Clustering  Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering).
## Gaussian Mixture

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.
