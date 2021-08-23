#                          NaVA Clustering
authors are removed for double blind review 
***

In this repository we present NaVA(Neural attention Variational Autoencoder) a new framework for general text clustering. In this repository we compare our technique with various state-of-the-art baselines in NLP. All of our experiments were executed on Google Colab, a NaVA model could be trained within a few minutes on this platform, its implementation based on neural networks, is also suitable for parallelization via GPU/TPU.

### Requirements
1. Python>=3.5
2. tensorflow>=1.16
3. numpy==1.19.5
4. scikit-learn==0.24.1


###  Datasets used in the experiments. 

|name | task | train set | test set | classes  |
|----------	|------------------------------	|-----------:|----------:|:-----------:|
|[20NewsGroup](https://github.com/davidsbatista/Aspect-Based-Sentiment-Analysis/tree/master/datasets/CR)  | User review polarity | 5176 | small | 2 |
|[IMDB](https://github.com/zeerakw/hatespeech)  | Hate speech detect| 23739 | large |3  |
|[TREC6](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)| Sentece polarity | 18179| large |  2 |
|[Subjectivity](https://github.com/hallr/DAT_SF_19/blob/master/data/yelp_labelled.txt)  | Movie and TV Review | 1000 | small|  2|
|[Biomedical](https://github.com/hallr/DAT_SF_19/blob/master/data/yelp_labelled.txt)  | Food review polarity | 1000| small| 2|
|[SearchSnippets](http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz) | Subjectivity and objectivity | 18179 | large | 2 |
|[Stackoverflow](https://github.com/hallr/DAT_SF_19/blob/master/data/yelp_labelled.txt)  | User product review  | 1000| small | 2  |
|[Quora DataSet](http://cogcomp.org/Data/QA/QC/)  | Question Answering  | 16504 | small | 6  |
|[Yelp Dataset](https://drive.google.com/file/d/0B8yp1gOBCztyN0JaMDVoeXhHWm8/)  | Movie and TV Review | 74337| large| 2 | 
|[Isis Tweets](https://drive.google.com/file/d/0B8yp1gOBCztyN0JaMDVoeXhHWm8/)  | Movie and TV Review | 74337| large| 2 | 

\
We propose a novel framework based on MVAE for general text clustering. NaVa(Neural attention Variational Autoencoder) is an unsupervised generative model of text which aims to extract a continuous semantic latent variable for each document. In this way, our model presents a new structure capable of increasing its latent representation using a deep architecture and attention mechanism.

<p align="center">
<img align="center" src="https://github.com/NaVaClustering/Experiments/blob/main/figs/a.png">
  Figure 1
</p>

### Input Data
NaVA receive as input any sequence of strings(sentences), each string of trainning corpus will be map to a vector.
Each file should have one sentence per line as follows (space delimited): \
`...`\
`weaknesses minor feel layout remote control show complete file names mp3s`\
`normal size sorry ignorant way get back 1x quickly` \
`It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`\
`many disney movies n play dvd player` \
`The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested.`\
`...`


After, for training NaVa you need specify his Hyperparameter and then you could use his text features( latent representation) for clustetring general text. Check this code to see how NaVA could be trained in a toy sample scenerio.


###  NaVa attention mechanism


In Figure 1  depicts a general idea for the recognition and generative models. A vector x representing a document passes through two attention layers in parallel, then each attention output pass through a deep neural network to obtain the latent representations c and h used by the mixture of Boltzmann machines.

Each attention layer will be responsible for measuring the importance of each word in our vocabulary for generating latent representations of documents h and clusters c. In this work our attention layer [[1]](#1) for input x is modelled as:

<p align="center">
<img class="center" src="https://github.com/NaVaClustering/Experiments/blob/main/figs/Screenshot%20from%202021-08-20%2013-50-59.png">
</p>

Where the key matrix, <a href="https://www.codecogs.com/eqnedit.php?latex=W\in\mathbf{R}^{|V|\times&space;D}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W\in\mathbf{R}^{|V|\times&space;D}" title="W\in\mathbf{R}^{|V|\times D}" /></a> and the query matrix, <a href="https://www.codecogs.com/eqnedit.php?latex=Q\in\mathbf{R}^{|V|\times&space;D}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q\in\mathbf{R}^{|V|\times&space;D}" title="Q\in\mathbf{R}^{|V|\times D}" /></a> with <a href="https://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D" title="D" /></a> dimensions of features will be used to gauge the importance of each <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a> word. Our attention function plays a key role in filtering out the most important words given input. As we work with bag-of-words that can contain high dimensions, our attention layer helps to reduce our high dimensionality by giving weights to each word in the vector. After that, we use multiple layers of neural networks to assess the latent values h and c, which will be used as latent representation of documents and cluster assign.

\
In our experiments, we performed some analysis to measure the impact of our attention engine on our models. Thus, in addition to noting a gain in unsupervised metrics, the space formed by the NaVA features presents a strong correlation with the groups found.Thus, we present in the figures below the visualization of the space formed in the 20newsGroup dataset using the TSNE dimensionality reduction technique. 

20News MVAE-BM| 20News NaVA|
:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/newplot(4).png  "Title") |  ![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/newplot(3).png  "Title")


\
In the figures above we have trained the MVAE-BM, a simplified version of NaVA, and in the figure on the right, the NaVA. So from the figures we can see that NaVA has a space that can better distinguish the clusters within the 20newsGroup dataset. Our attention function here plays a fundamental role in achieving better results than the MVAE-BM. 

Words Most important | TSNE Tweets Dataset|
:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/go2.png  "Title") |  ![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/cluster_tw.png  "Title")


We propose a novel framework based on MVAE for general text clustering. NaVa(Neural attention Variational Autoencoder) is an unsupervised generative model of text which aims to extract a continuous semantic latent variable for each document. In this way, our model presents a new structure capable of increasing its latent representation using a deep architecture and attention mechanism. We propose a novel framework based on MVAE for general text clustering. NaVa(Neural attention Variational Autoencoder) is an unsupervised generative model of text which aims to extract a continuous semantic latent variable for each document. In this way, our model presents a new structure capable of increasing its latent representation using a deep architecture and attention mechanism.


###  Unsupervised Clustering, without Ground Truth

In this section, we will present text clustering experiments on datasets where there is no pre-defined manual labeling. Therefore, in these experiments, we will evaluate the ability of \model~ and our baselines to find consistent clusters by evaluating unsupervised learning metrics where the labels are not known.

For these experiments we used 3 unsupervised learning metrics: Silhouette, Davies Bouldin and Calinski Harabasz.

The Silhouette metric is calculated using the average intra-cluster distance and the average distance from the nearest cluster for each sample. So, this metric presents values close to 1 when the clusters found are far from each other and the examples of each cluster are close to each other. The Davies Bouldin metric is defined as the measure of average similarity of each cluster with its most similar cluster, where similarity is the ratio between the distances between the clusters. Therefore, more distant and purer clusters will result in a better score. Finally, the Calinski Harabasz, also known as the Variance Ratio Criterion, is a measure defined as the ratio between the dispersion within the cluster and the dispersion between the clusters. 

<p align="center">
<img align="center" src="https://github.com/NaVaClustering/Experiments/blob/main/figs/Untitled%20presentation(4).jpg">
</p>

In our first experiment we will study the impact of varying the amount of clusters([$2-32$]) on each dataset not labeled by \model. With this experiment, we intend to also identify possible gains and losses when the number of clusters is varied, instead of coldly analyzing a numerical value found by a metric. Thereby, we now present table 3, where we can see the comparison between \model~ and our deep text representation baselines on our 3 unlabeled datasets. On the Quora dataset, all baselines performed better with a number of clusters equal to 2, while the NaVA equals 3, although the number of 2 clusters also presented similar results, corroborating the results shown in the previous graphs. For the Yelp and Isis datasets all methods presented 2 as the optimal number of clusters.



### Reference

Please make sure to cite the papers when its use for represents document for clustering or classification.
Please cite the following paper if you use this implementation:\
`@InProceedings{ Submitted to WSDM'22,`\
  `author    = {authors are removed for double blind review},`\
  `title     = {Neural attention Variational Autoencoder for Text Clustering},`\
  `booktitle = {WSDM'21},`\
  `year      = {2021} }`


<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.



