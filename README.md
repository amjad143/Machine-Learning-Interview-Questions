# A collection of useful ML questions(with answers yet to be added)

### Questions:

#### 0. What is Bias, what is Variance and what does Bias-Variance trade-off(or decomposition) mean?

This is a concept that is well known in the context of supervised learning where we have some labeled data and we want to estimate an unknown function **c(X)**
using a function with known format and parameters, called hypothesis function (i.e. **h(X)**).

[Wikipedia's](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) definitions for bias and variance are as follows:

* The bias is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
* The variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting)

Consider h(X) ~ c(X) then we have: c(X) = h(X) + bias error + variance error + irreducible error; apart from the third term (i.e. irreducible error) we can reduce the first two types of errors.
bias_error originates from the assumptions we make about the characteristics of our models, for example we assume that the relationship between input and output is linear (like in linear regression);
while creating out prediction models, we have a subset of all labeled data (training data) and our guide for knowing how good our model is based on its 
performance on this limited set, this creates a problem where the training data set is relatively small (many real world problems) because the variance of error on the unseen data (test data) could be huge. In fact by putting all our effort in improving the 
training score and lowering training error (we have no other choice!) we are doomed to overfit :( ). 

Machine learning algorithms are influenced differently based on their assumptions (bias) about input and output and consequently have different error variances. 
Algorithms that have a high variance are strongly influenced by the characteristics of the training data. This means that the characteristics of the data 
have influences the number and types of parameters used to characterize the hypothesis function [[https://machinelearningmastery.com](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)]. 

The bias-variance trade-off is a central problem in supervised learning. Ideally, 
one wants to choose a model that both accurately captures the regularities in its training data,
 but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with low variance typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities [[Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)]. 
 
#### 1. What is the main difference between the ordinary algorithms and machine learning algorithms?
* A traditional algorithm takes some input and some logic in the form of code and drums up the output. As opposed to this, a Machine Learning Algorithm takes an input and an output and gives the some logic which can then be used to work with new input to give one an output.
#### 2. What is SVM? how does it work? Explain the math behind it
* SVM is a supervised machine learning algorithm which can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs.
#### 3. What are L1 and L2 regularizations? Why we may need regularization?
* A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression. The key difference between these two is the penalty term. Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.
#### 4. How does Decision Trees algorithm work?
Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. ... The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.
#### 5. What are some major types of machine learning problems?
#### 6. What is Probably Approximately Correct(or PAC learning framework)?
* Probably approximately correct (PAC) learning is a theoretical framework for analyzing the generalization error of a learning algorithm in terms of its error on a training set and some measure of complexity. The goal is typically to show that an algorithm achieves low generalization error with high probability.
#### 7. What are the example applications of machine learning?
#### 8. What are Features, Labels, Training samples, Validation samples, Test samples, Loss function, Hypothes set and Examples?
* What is labeled training data?
Labeled data, used by Supervised learning add meaningful tags or labels or class to the observations (or rows). These tags can come from observations or asking people or specialists about the data. Classification and Regression could be applied to labelled datasets for Supervised learning.
* What is validation set used for?
A validation set is a set of data used to train artificial intelligence (AI) with the goal of finding and optimizing the best model to solve a given problem. Validation sets are also known as dev sets. A supervised AI is trained on a corpus of training data.
* What is training and validation loss?
One of the most widely used metrics combinations is training loss + validation loss over time. The training loss indicates how well the model is fitting the training data, while the validation loss indicates how well the model fits new data.
#### 9. What are Overfitting and Underfitting?
* Overfitting: Good performance on the training data, poor generliazation to other data. 
* Underfitting: Poor performance on the training data and poor generalization to other data.
#### 10. What is Cross-validation? how does it help reduce Overfitting?
* Cross-validation is a powerful preventative measure against overfitting. ... In standard k-fold cross-validation, we partition the data into k subsets, called folds. Then, we iteratively train the algorithm on k-1 folds while using the remaining fold as the test set (called the “holdout fold”).
* Definition. Cross-Validation is a statistical method of evaluating and comparing learning algorithms by dividing data into two segments: one used to learn or train a model and the other used to validate the model.
#### 11. How much math is involved in the machine learning? what are the pre-requisites of ML?
* Which Mathematical Concepts Are Implemented in Data Science and Machine Learning. Machine learning is powered by four critical concepts and is Statistics, Linear Algebra, Probability, and Calculus. While statistical concepts are the core part of every model, calculus helps us learn and optimize a model.
* What is the math prerequisite for machine learning?
Math concepts are still prerequisites for machine learning. A thorough understanding of mathematical concepts like linear algebra, calculus, probability theory and statistics is necessary to gain a solid understanding of the internal working of the algorithms.
#### 12. What is Kernel Method(or Kernel Trick)?
* A Kernel Trick is a simple method where a Non Linear data is projected onto a higher dimension space so as to make it easier to classify the data where it could be linearly divided by a plane. This is mathematically achieved by Lagrangian formula using Lagrangian multipliers.
* Kernel Function is a method used to take data as input and transform into the required form of processing data. “Kernel” is used due to set of mathematical functions used in Support Vector Machine provides the window to manipulate the data.
#### 13. What is Ensemble Learning?
* An ensemble is a machine learning model that combines the predictions from two or more models. The predictions made by the ensemble members may be combined using statistics, such as the mode or mean, or by more sophisticated methods that learn how much to trust each member and under what conditions.
#### 14. What is Manifold Learning?
* Manifold learning is a popular and quickly-growing subfield of machine learning based on the assumption that one's observed data lie on a low-dimensional manifold embedded in a higher-dimensional space.
* Manifold Learning is basically learning finding a basis set, the manifold, that explains maximal variations in a dataset. For example, principal component analysis (PCA) finds eigenvectors explaining the maximal variations in a dataset, in a sense, PCA has found a "manifold" that explains the dataset. When you have high-dimensional (N dimensions) datasets, you want to learn the underlying manifold, often, explaining the N-dimensional dataset in M dimensions (M < N). Basically, manifold learning attempts to do dimensionality reduction and tries to learn this reduction. Manifolds generally consider properties of neighboring points, such as distance. Swiss Roll dataset is an initial test to your manifold learning algorithm. This is still an active area of research.
#### 15. What is Boosting?
* Definition: The term 'Boosting' refers to a family of algorithms which converts weak learner to strong learners.
* In machine learning, boosting is an ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones.
#### 16. What is Stochastic Gradient Descent? Describe the idea behind it
* Gradient descent is an iterative algorithm, that starts from a random point on a function and travels down its slope in steps until it reaches the lowest point of that function.” This algorithm is useful in cases where the optimal points cannot be found by equating the slope of the function to 0.
* According to a senior data scientist, one of the distinct advantages of using Stochastic Gradient Descent is that it does the calculations faster than gradient descent and batch gradient descent. ... Also, on massive datasets, stochastic gradient descent can converges faster because it performs updates more frequently.
#### 17. What is a Statistical Estimator?
* An estimator is a statistic that estimates some fact about the population. You can also think of an estimator as the rule that creates an estimate. For example, the sample mean(x̄) is an estimator for the population mean, μ. The quantity that is being estimated (i.e. the one you want to know) is called the estimand.
#### 18. What is Rademacher complexity?
* In computational learning theory (machine learning and theory of computation), Rademacher complexity, named after Hans Rademacher, measures richness of a class of real-valued functions with respect to a probability distribution.
#### 19. What is VC-dimension?
* In Vapnik–Chervonenkis theory, the Vapnik–Chervonenkis dimension is a measure of the capacity of a set of functions that can be learned by a statistical binary classification algorithm.
#### 20. What are Vector, Vector Space and Norm?
* In mathematics, a normed vector space or normed space is a vector space over the real or complex numbers, on which a norm is defined. A norm is the formalization and the generalization to real vector spaces of the intuitive notion of "length" in the real world.
#### 21. Why Logistic Regression algorithm has the word regression in it?
* Logistic regression uses the same basic formula as linear regression but it is regressing for the probability of a categorical outcome. Linear regression gives a continuous value of output y for a given input X. ... That's the reason, logistic regression has “Regression” in its name.
#### 22. What is Hashing trick?
* In machine learning, feature hashing, also known as the hashing trick, is a fast and space-efficient way of vectorizing features, i.e. turning arbitrary features into indices in a vector or matrix.
* Definition: A hash algorithm is a function that converts a data string into a numeric string output of fixed length. ... MD5 Message Digest checksums are commonly used to validate data integrity when digital files are transferred or stored.
#### 23. How does Perceptron algorithm work?
* In machine learning, the perceptron is an algorithm for supervised learning of binary classifiers. ... It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.
#### 24. What is Representation learning(or Feature learning)?
* In machine learning, feature learning or representation learning is a set of techniques that allows a system to automatically discover the representations needed for feature detection or classification from raw data.
#### 25. How does Principal Component Analysis(PCA) work?
* Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.
#### 26. What is Bagging?
* Bagging, also known as bootstrap aggregation, is the ensemble learning method that is commonly used to reduce variance within a noisy dataset.
* Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting.
#### 27. What is Feature Embedding?
* Feature embedding is an emerging research area which intends to transform features from the original space into a new space to support effective learning. ... The learned numerical embedding features can be directly used to represent instances for effective learning.
#### 28. What is Similarity Learing?
* Similarity learning is an area of supervised machine learning in artificial intelligence. It is closely related to regression and classification, but the goal is to learn a similarity function that measures how similar or related two objects are.
#### 29. What is Feature Encoding? How an Autoencoder Neural Network work?
* Autoencoder is an unsupervised artificial neural network that learns how to efficiently compress and encode data then learns how to reconstruct the data back from the reduced encoded representation to a representation that is as close to the original input as possible.
#### 30 Does ML have any limit? What those limits may be?
* The Limitations of Machine Learning
* Each narrow application needs to be specially trained.
* Require large amounts of hand-crafted, structured training data.
* Learning must generally be supervised: Training data must be tagged.
* Require lengthy offline/ batch training.
* Do not learn incrementally or interactively, in real time.
#### 31 What does the word naive in the name of Naive Bayes family of algorithms stand for?
* In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features (see Bayes classifier).
* Naive Bayes is called naive because it assumes that each input variable is independent. This is a strong assumption and unrealistic for real data; however, the technique is very effective on a large range of complex problems.
#### 32. Describe various strategies to handle an imbalanced dataset?
* Random undersampling with RandomUnderSampler.
* Oversampling with SMOTE (Synthetic Minority Over-sampling Technique)
* A combination of both random undersampling and oversampling using pipeline.
#### 33. Describe various strategies to tune up the hyper-parameters of a particular learning algorithm in general
* Grid search is arguably the most basic hyperparameter tuning method. With this technique, we simply build a model for each possible combination of all of the hyperparameter values provided, evaluating each model, and selecting the architecture which produces the best results.
#### 34. What are some general drawbacks of tree based learning algorithms?
* Disadvantages of decision trees: They are unstable, meaning that a small change in the data can lead to a large change in the structure of the optimal decision tree. They are often relatively inaccurate. Many other predictors perform better with similar data.
* What are the issues faced by decision tree algorithm?
* Issues in Decision Tree Learning
* Overfitting the data: ...
* Guarding against bad attribute choices: ...
* Handling continuous valued attributes: ...
* Handling missing attribute values: ...
* Handling attributes with differing costs:
#### 35. WHat are the differences between ordinary Gradient Boosting and XGBOOST?
#### 36. What is the main difference between Time Series Analysis and Machine Learning?
#### 37. What is the main difference between Data Mining and Machine Learning? Are they the same?
#### 38. What is a Generative learning model?
#### 39  What is a Discriminative learning model?
#### 40. What is the difference between Generative and Discriminative models?
#### 41. What is Case-Based Learning?
#### 42. What is Covariance Matrix?
#### 43. What is the difference between Correlation and Causation?
#### 44. What is the Curse of Dimensionality? How does it may hinder the learning process?
#### 45. How Dimensionality Reduction help improve the performance of the model?
#### 46. What is Feature Engineering?
#### 47. What is Transfer Learning?
#### 48. What do (Multi-)Collinearity, Autocorrelation, Heteroskedasticity and Homoskedasticity mean?
#### 49. Explain Backpropagation, What are some of its shortcomings?
#### 50. How do Boltzmann Machines work?
#### 51. What is the difference between In-sample evaluation and Holdout evaluation of a learning algorithm?
#### 52. What is Platt Scaling?
#### 53. What is Monotonic(or Isotoonic) Regression?
#### 54. How BrownBoost algorithm works?
#### 55. How Multivariate Adaptive Regression Splines(MARS) algorithm works?
#### 56. What are K-Fold Cross Validataion and Stripified Cross Validation?
#### 57. What is K-Scheme (Categorical Feature) Encoding? What are some drawbacks of one-hot categorical feature encoding?
#### 58. What is Locality Sensitive Hashing(LSH)?
#### 59. What are the difrerences of Arithmatic, Geometric and Harmonic means?
#### 60. What is a Stochastic Process?
#### 61. How Bayesian Optimization work?
#### 62. What is the difference between Bayesian and frequentist approaches?
#### 63. Why sometimes it is needed to Scale or to Normalise features?
#### 65. What is Singular-Value Decomposition(SVD)? What are some applications of SVD?
#### 66. Define Eigenvector, Eigenvalue, Hessian Matrix, Gradient
#### 67. What is an (Intelligent) Agent?
#### 68. What is Q-Learning?
#### 69. Define Markov Chain and Markov Process
#### 70. Explain how K-means clustering alorithm works, Why is it so popular?
#### 71. How Hierchial clustering algorithm works?
#### 72. What is Discriminant Analysis?
#### 73. What is Multivariate Analysis?
#### 74. What is the Rank of a matrix?
#### 75. How does Balanced Iterative Reducing and Clustering using Hierarchies(BIRCH) algorithm work?
#### 76. What is a Mixture Model?
#### 77. How does Machine Learning modeling fare against old-school mathematical modeling such as modeling a real-world system via use of differential equations?
#### 78. How to pick a suitable kernel for SVM?
#### 79. What is Query-Based Learning?
#### 80. What is Algorithmic Probability?
#### 81. What are  Occam's Razor and Epicurus' principle of multiple explanations?
#### 82. What are Filter and Wrapper based feature selection methods?
#### 83. What is Graph Embedding? What are some of its applications?
#### 84. What is Multi-Armed Bandit Problem?
#### 85. What is Active Learning?
#### 86. What are Hierarchical Temporal Memory algorithms(HTMs)?
#### 87. What are Factorization Machines?
#### 88. What are Probabilistic Graphical Models(PGM)?
#### 89. What is Entity Embedding?
#### 90. What is Feature Augmentation?
