# Shikha_Repo
**introduction**
Nowadays people try to lead a luxurious life. They tend to use the things either for show off or for
their daily basis. These days the consumption of red wine is very common to all. So it became important to
analyze the quality of red wine before its consumption to preserve human health. Hence this research is a step
towards the quality prediction of the red wine using its various attributes. Red wine quality and style are
highly influenced by the qualitative and quantitative composition of aromatic compounds having various
chemical structures and properties and their interaction within different red wine matrices. The understanding
of interactions between the wine matrix and volatile compounds and the impact on the overall flavor as well
as on typical or specific aromas is getting more and more important for the creation of certain wine styles.
Based on the data visualisation of python processing, classical visualization tools such as boxplot, correlation
matrix, jointplot and various algorithms for the result.

**Objectives**
The objectives of this project are as follows:
 1.To experiment with different classification methods to see which yields the highest accuracy
 2.To determine which features are the most indicative of a good quality wine 

**Problem Definition:**
 The red wine dataset contains different chemical information about red wine. It has 1599 instances
with 12 variables each. The dataset is good for classification and regression tasks. The model can be used to
predict red wine quality. Perform various different algorithms like regression, decision tree, random forests,
etc and differentiate between the models and analyse their performances.
Here I am Performing various different Classification algorithms like Logistics Regression, Decision Tree
Classifier, Random Forest Classifier, Stochastic Gradient Descent Classifier, Naive Bayes Classifier, KNearestNeighbours Classifier and Support Vector Machine(SVM) and trying to differentiate between the
models and analyse their performances. 

**Libraries Imported:**
_import pandas as pd -_ pandas is a popular Python-based data analysis toolkit which can be imported
using import pandas as pd. It presents a diverse range of utilities, ranging from parsing multiple file formats
to converting an entire data table into a NumPy matrix array. This makes pandas a trusted ally in data
science and machine learning. Similar to NumPy, pandas deals primarily with data in 1-D and 2-D arrays;
however, pandas handles the two differently.
_import matplotlib.pyplot as plt -_ matplotlib.pyplot is stateful, in that it keeps track of the current
figure and plotting area, and the plotting functions are directed to the current axes and can be imported
using import matplotlib.pyplot as plt.
import seaborn as sns - Seaborn is a library for making statistical graphics in Python. It builds on top
of matplotlib and integrates closely with pandas data structures.Seaborn helps you explore and understand
your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally
perform the necessary semantic mapping and statistical aggregation to produce informative plots. Its
dataset-oriented, declarative API lets you focus on what the different elements of your plots mean, rather
than on the details of how to draw them.
_import numpy as np_ - Numpy provides a large set of numeric datatypes that you can use to construct
arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually
also include an optional argument to explicitly specify the datatype.
_%matplotlib inline_ - %matplotlib inline sets the backend of matplotlib to the 'inline' backend: With this
backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook,
directly below the code cell that produced it.
_from sklearn.linear_model import LogisticRegression- _This class implements
regularized logistic regression using the 'liblinear' library, from sklearn.datasets
from sklearn.model_selection import train_test_split- train_test_split is a function in Sklearn
model selection for splitting data arrays into two subsets: for training data and for testing data. With this
function, you don't need to divide the dataset manually. By default, Sklearn train_test_split will make
random partitions for the two subsets. 
