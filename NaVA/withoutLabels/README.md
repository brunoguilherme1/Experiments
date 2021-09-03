# Text Clustering without Ground Truth
In this section, we will present text clustering experiments on datasets where there is no pre-defined manual labeling. Therefore, in these experiments, we will evaluate the ability of NaVA to find consistent clusters by evaluating unsupervised learning metrics where the labels are not known.

In our first experiment we will study the impact of varying the amount of clusters([2-32]) on each dataset not labeled by \model. With this experiment, we intend to also identify possible gains and losses when the number of clusters is varied, instead of coldly analyzing a numerical value found by a metric.

Yelp| Isis |Quora
:-------------------------:|:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/pink_yelp(1).png  "Title") |![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/quora(1).png  "Title")|![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/tweets.png  "Title")

With this in mind, we now present figures above, in which we can observe the density distribution of the values found by the Silhouette metric when we vary the number of clusters between 2 and 32 (Y axis) for each unlabeled dataset. Thereafter, we can observe that for smaller numbers of clusters (<6) the densities of the three figures are concentrated around 0 and 0.5, and as the number of clusters is increased the densities tend to shift to -0.5 and 0.0.
The saturation of the amount of clusters in the space formed by NaVA on the Quora dataset and Isis Datset is less smooth than Yelp dataset, in which we can observe an abrupt change in the shape of the curves with number of clusters < 4. Thus, these results demonstrate that the Yelp dataset could support a larger number of clusters, while the Quora and Isis datasets should present the values of 2 or 3 as the optimal number of clusters. 

 
Yelp| Isis |Quora
:-------------------------:|:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/yelp_box.png  "Title") |![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/isis.png  "Title")|![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/quora.png  "Title")


Yelp| Isis |Quora
:-------------------------:|:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/yelp-line.png  "Title") |![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/isis-line.png  "Title")|![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/quora-line.png  "Title")

In boxplot and line graphs, we also studied the Silhouette's behavior in the three datasets studied. As we can see, the optimal values of the silhoeutte are in the quantities of lower clusters. This factor was also observed in our deep learning baselines [see](https://github.com/NaVaClustering/Experiments/tree/main/Baselines). In our experiments we used 3 unsupervised learning metrics: Silhouette, Davies Bouldin and Calinski Harabasz.The Silhouette metric is calculated using the average intra-cluster distance and the average distance from the nearest cluster for each sample. So, this metric presents values close to 1 when the clusters found are far from each other and the examples of each cluster are close to each other. The Davies Bouldin metric is defined as the measure of average similarity of each cluster with its most similar cluster, where similarity is the ratio between the distances between the clusters. Therefore, more distant and purer clusters will result in a better score. Finally, the Calinski Harabasz, also known as the Variance Ratio Criterion, is a measure defined as the ratio between the dispersion within the cluster and the dispersion between the clusters.  





