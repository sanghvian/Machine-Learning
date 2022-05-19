# Machine-Learning
A repository of ML notes, projects, libraries and applications

### A brief introduction


> The simplest 2 word explanation of ML - Curve Fitting

Basically, given a dataset, is a computer able to learn & interpret, what it ideally should i.e. learn it like a human would.

This is achieved by mathematically encoding everything from images, text, analytics into vectors that can be passed through a function that we want to come up with, that would actually give an output close to what a human brain would do for the same input information.

The accuracy of-course would be measured by drawing curves & see how well the output by the machine, matches real-world thinking i.e. how well does the predicted curve, fit the actual curve.


### Basic Algorithms

I've personally been a developer, most of my career, hence most technologies I learn or stumble across new technologies across are actually when I run into a business use case that demands a new solution or approach as the most optimal way forward.

Hence, in this post, instead of explaining in depth what each algorithm does & the mathematics of it, I will primarily be focussing on explaining the basic algorithms used in Supervised & Unsupervised Learning & when to use which algorithm.

We will be speaking in short about :

1. Linear Regression
2. Logistic Regression (Classification)
3. Neural Networks
4. Support Vector Machine
5. K-Means Clustering - deriving structure from data
6. Principal Component Analysis

For each algorithm, discuss :
1. Optimization Objective
2. How to actually use it
3. When to use it

Also, as a bonus, I'll point you to some resources on actually debugging learning algorithms so that you can improve upon an implemented solution. 🚀


#### 1. Linear Regression

`Regression` `Multivariate` `Gradient Descent` `Normalization` `Regularization`

**Used when** - We have a labelled dataset & we wish to devise a function that uses the features + parameters & predicts the most accurate output. Since, the results are continuous & not into classified buckets, hence, the term "regression". 

**Eg application** - Guessing the price of a house, starting with a labelled dataset of various houses’ features such as area, locality etc. & their actual prices

"linear", since we mostly only deal with features independent of each other. Eg : size of a house does not ideally depend directly on how near its location is to a school. In case we do have dependent features we use - "Multivariate regression" a technique used to measure the degree to which the various independent variable and various dependent variables are linearly related to each other

We optimize the function we come up with using the "Gradient Descent" approach, by which we come up with better coefficients for each feature in our function.

Also, we do something called "regularization" to prevent underfitting or overfitting the predictions we give to the dataset we have. Get it just right :)


#### 2. Logistic Regression

`Classification` `Sigmoid` `Gradient Descent` `One-vs-all`

**Use case** - We have a labelled dataset & we wish to devise a function that uses features + parameters & predicts the most accurate classified output

**Eg application** - Looking at a series of emails & classifying as "spam" or "not-spam". eg of multiclass - Looking at a series of vehicle images, predict which of them is a bike, car, ship, bus etc

We basically predict the probability (between 0 to 1) of a given data point to lie in one of the classes. We use a "sigmoid" function as part of the function we come up with outputs lying in the range 0 to 1

For multiclass classification, we predict for each data point, what is the probability that it lies in 1 class vs its probability of not lying in that class (i.e. probability of it lying in any class except that class). This approach is called "one-vs-all" approach & we calculate this for all classes for all data points


#### 3. Neural Networks

`Complex non-linear hypotheses` `Forward Propagation` `Backward Propagation`

**Use case** - We have a labelled dataset & we wish to learn interesting features starting from initial features, & perform classification. Basically, cases where we can’t use logistic regression for complex hypotheses as it is only a linear classifier

**Eg application** - Handwriting recognition

Each set of interesting features, that we learn, using the initial input features & training examples & building up on those - constitute a layer
Eg : Using 32x32 pixels black & white images & from a collection of pixels, learning what a line, arc, curve is & then using those, understanding how Arabic Numbers are constructed as a combination of these newly learnt, interesting features.  

We use "Forward Propagation" algorithm to predict output based on the hypothesis function the layers of the neural network collectively. We use "Backward Propagation" algorithm to optimize the accuracy of the neural network


#### 4. Support Vector Machine

`Kernels` `Gaussian Kernels`

**Use case** - Reducing feature set for ***labelled*** training data

Gives a more powerful & cleaner way of learning complex non-linear functions as compared to regression, neural networks - by learning only on basis of most important features in the given set.
Can be used to increase & decrease feature set size

**Eg application** - Feature set optimization in literally any machine learning implementation

We use kernels i.e. predefined methodologies to use existing features & come up with a smaller but more impactful feature set that can be less compute-heavy but more important to the actual output


#### 5. K-Means Clustering

`Cluster Assignment` `Move Centroid`

**Use case** - Given an unlabelled dataset, that we have to transform into discrete classes/clusters/groups of data

**Eg application** - Grouping of Tshirts, by using dimension values, into S, M, L, XL categories

Basically we start by initializing the random points as "cluster leaders" (aka centroids) & then based on the similarity of other data points, with respect to these, add the other points to the clusters. This is known as "cluster assignment" step

Once this is done, we recompute the "cluster leaders' " values to be the average of values of all points in the cluster. This is known as "move centroid" step as on an actual graph, this average would be a centroid that would move upon recomputation

We keep alternating "move centroid" & "cluster assignment" step for a couple iterations till the "cluster leaders' " values aren't changing anymore or changing negligibly

#### 6. Principal Component Analysis

`Dimensionality Reduction` `Eigen Vector` `Reconstruction`

**Use case** - Reducing feature set for ***unlabelled*** & ***labelled*** training examples.


It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation.

**Eg application** - Reduce memory needed to store data & to speed up learning algorithm

## Word of advice
Sometimes, by just looking at a ML problem, we might not know what ML algorithm to use. More than the algorithm itself, the following play a big part in training a model :

1. How much data do you have
2. How skilled are you at error analysis, debugging learning algorithms, designing new features, figuring out what features give you learning algorithms & so on
