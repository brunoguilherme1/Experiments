# Transformed-Based Clustering
Sentence-Transformers can be used in different ways to perform clustering of large set of sentences. In this repository, we evaluate 5 baselines based on transformed deep learning.
* Baselines
    * Mass(paraphrase-mpnet-base-v2)
    * Multi Language BERT(distiluse-base-multilingual-cased-v2)
    * XLM-RoBERTa(xlm-r-100langs-bert-base-nli-stsb-mean-tokens)
    * ALBERT (paraphrase-albert-base-v2)
    * BERT(bert-large-nli-stsb-mean-tokens)


## k-Means
[kmeans](https://scikit-learn.org/stable/modules/clustering.html#k-means) contains an example of using [K-means Clustering Algorithm](https://scikit-learn.org/stable/modules/clustering.html#k-means). K-Means requires that the number of clusters is specified beforehand. The sentences are clustered in groups of about equal size.
 
## AffinityPropagation Clustering
[AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation) creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples.
## Gaussian Mixture

A [Gaussian mixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

## Optimal Number of Clusters (non Ground Truth)
The optimal choice of clusters for each baseline was made by analyzing the SI, DB and CA  values of the datasets that do not have labels, as shown in the figures below, which show that all our baselines have a value of 2 as the optimal number of clusters. 



