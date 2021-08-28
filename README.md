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
|[20NewsGroup]()  | User review polarity |11314 | 7531 | 20 |
|[IMDB]()  | Sentiment Analysis|35000 | 15000 |2  |
|[TREC6](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)| Question Answering  | 3816| 1636 |  6 |
|[Subjectivity]()  | Movie and TV Review | 9756 |  3323|  2|
|[Biomedical]()  | biomedical text clustering |20000| -| 20|
|[SearchSnippets]() | short text clustering | 19245 | - | 8 |
|[Stackoverflow])  | short text clustering | 18543| - | 20  |
|[Quora DataSet](http://cogcomp.org/Data/QA/QC/)  | Question Answering  | 19500 |5512 | -  |
|[Yelp Dataset]() |user review| 25000 | 10000| -| 
|[Isis Tweets]()  | Movie and TV Review |15000| 6000| - | 

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


After, for training NaVa you need specify his Hyperparameter and then you could use his text features( latent representation) for clustetring general text. Check this [code](https://github.com/NaVaClustering/Experiments/blob/main/NaVA/NaVA_Example.ipynb) to see how NaVA could be trained in a toy sample scenario.


###  NaVa attention mechanism


In Figure 1  depicts a general idea for the recognition and generative models. A vector x representing a document passes through two attention layers in parallel, then each attention output pass through a deep neural network to obtain the latent representations c and h used by the mixture of Boltzmann machines.

Each attention layer will be responsible for measuring the importance of each word in our vocabulary for generating latent representations of documents h and clusters c. In this work our attention layer [[1]](https://arxiv.org/pdf/2006.00988.pdf) for input x is modelled as:

<p align="center">
<img class="center" src="https://github.com/NaVaClustering/Experiments/blob/main/figs/Screenshot%20from%202021-08-20%2013-50-59.png">
</p>

Where the key matrix, <a href="https://www.codecogs.com/eqnedit.php?latex=W\in\mathbf{R}^{|V|\times&space;D}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W\in\mathbf{R}^{|V|\times&space;D}" title="W\in\mathbf{R}^{|V|\times D}" /></a> and the query matrix, <a href="https://www.codecogs.com/eqnedit.php?latex=Q\in\mathbf{R}^{|V|\times&space;D}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q\in\mathbf{R}^{|V|\times&space;D}" title="Q\in\mathbf{R}^{|V|\times D}" /></a> with <a href="https://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D" title="D" /></a> dimensions of features will be used to gauge the importance of each <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a> word. Our attention function plays a key role in filtering out the most important words given input. As we work with bag-of-words that can contain high dimensions, our attention layer helps to reduce our high dimensionality by giving weights to each word in the vector. After that, we use multiple layers of neural networks to assess the latent values h and c, which will be used as latent representation of documents and cluster assign.

\
In our experiments, we performed some analysis to measure the impact of our attention engine on our models. Thus, in addition to noting a gain in unsupervised metrics, the space formed by the NaVA features presents a strong correlation with the groups found.Thus, we present in the figures below the visualization of the space formed in the 20newsGroup dataset using the TSNE dimensionality reduction technique. 

Without Attention| With Attention|
:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/newplot(4).png  "Title") |  ![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/newplot(3).png  "Title")


\
In addition, NaVA allows the most important words to be highlighted using the energy function within our model. So, now we present the figures below, where within the Tweets dataset we present the most important words of each example just like the TSNE visualization.

Words Most important | TSNE ISIS Dataset|
:-------------------------:|:-------------------------:|
![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/go2.png  "Title") |  ![Figure 1 ](https://github.com/NaVaClustering/Experiments/blob/main/figs/cluster_tw.png  "Title")


###  Unsupervised Clustering, without Ground Truth

We will present text clustering experiments on datasets where there is no pre-defined manual labeling. Therefore, in our experiments, we will evaluate the ability of NaVa and our baselines to find consistent clusters by evaluating unsupervised learning metrics where the labels are not known. For these experiments we used 3 unsupervised learning metrics: Silhouette, Davies Bouldin and Calinski Harabasz. In our first experiment we will study the impact of varying the amount of clusters(2-32) on each dataset not labeled by NaVA. With this experiment, we intend to also identify possible gains and losses when the number of clusters is varied, instead of coldly analyzing a numerical value found by a metric. 

<p align="center">
<img align="center" src="https://github.com/NaVaClustering/Experiments/blob/main/figs/table3.png">
</p>

In table 3, where we can see the comparison between NaVA and our deep text representation baselines on our 3 unlabeled datasets. On the Quora dataset, all baselines performed better with a number of clusters equal to 2, while the NaVA equals 3, although the number of 2 clusters also presented similar [results](https://github.com/NaVaClustering/Experiments/tree/main/NaVA/withoutLabels), corroborating the results shown in the previous graphs. For the Yelp and Isis datasets all methods presented 2 as the optimal number of clusters. So, comparing the values found for each metric in table 3, we can observe that the NaVA presented the best results on the three evaluated datasets, being surpassed only by ALBERT in CA metric on Yelp dataset.



### Reference

Please make sure to cite the papers when its use for represents document for clustering.
Please cite the following paper if you use this implementation:\
`@InProceedings{Submitted to WSDM'22,`\
  `author    = {authors are removed for double blind review},`\
  `title     = {Neural attention Variational Autoencoder for Text Clustering},`\
  `booktitle = {WSDM'22},`\
  `year      = {2022} }`
