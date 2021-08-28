# Text Clustering without Ground Truth
In this section, we will present text clustering experiments on datasets where there is no pre-defined manual labeling. Therefore, in these experiments, we will evaluate the ability of NaVA to find consistent clusters by evaluating unsupervised learning metrics where the labels are not known.

In our first experiment we will study the impact of varying the amount of clusters([$2-32$]) on each dataset not labeled by \model. With this experiment, we intend to also identify possible gains and losses when the number of clusters is varied, instead of coldly analyzing a numerical value found by a metric. 

Silhouette | Davies-Bouldin |Calinski-Harabasz
:-------------------------:|:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/si.png  "Title") |![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/db.png  "Title")|![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/ca.png  "Title")

With this in mind, we now present figures above, in which we can observe the density distribution of the values found by the Silhouette metric when we vary the number of clusters between 2 and 32 (Y axis) for each unlabeled dataset. Thereafter, we can observe that for smaller numbers of clusters (<6) the densities of the three figures are concentrated around 0 and 0.5, and as the number of clusters is increased the densities tend to shift to -0.5 and 0.0.
The saturation of the amount of clusters in the space formed by NaVA on the Quora dataset and Isis Datset is less smooth than Yelp dataset, in which we can observe an abrupt change in the shape of the curves with number of clusters < 4. Thus, these results demonstrate that the Yelp dataset could support a larger number of clusters, while the Quora and Isis datasets should present the values of 2 or 3 as the optimal number of clusters. 
