# AI, Map Prediction
```AI Supervised learning from data```
```Created in July and August 2021```
```Professional Holiday Practice 2021```

---

### Project Description

Using AI and Machine Learning AI Model has to recognise specific places on the production hall map. 
Program must perform a pipeline of operations:
1. Clustering
2. Classification
3. Regression

Thanks to clustering Machine Learning algorithms can learn how the specific places on the production hall
are placed on the map.

After Classification AI Model learn and best reflect the time spent by an employee in a 
given area (see the description of the input data).

The whole project is also intended to study and review the available machine learning algorithms.

---

### Data details and description

1. Train and test data stores information about employees moving inside the production hall. 
Data file stores information about their ID, coordinates (x and y), measurement time and other (unnecessary for description) data.


2. Train and test data was created from the breakdown of data held in the database. 
Database can export data to 2 different files formats - json and csv, so the program has to
be able to read files in both these formats.
   

3. Data is truly used to analyze employee movements. They are not made up or designed only for machine learning, 
   which makes it difficult to use them for this purpose
   

4. The data consists of slightly more than 28,000 records
   which must still be broken down into 3 different employees during the learning process.
   

5. Test data represents 20% of the data and training data 80%.
   

6. Train data shape is always (*n*, 3), which *n* is number of data rows.

---

### Clustering

During clustering process **5** different algorithms were used:
1. Agglomerative Clustering
2. DBScan
3. KMeans
4. OPTICS
5. Spectral clustering

The main problem using all algorithms was to select the appropriate number of clusters.
The more clusters there were, the smaller the mean square error of the entire population was, but it did not provide a satisfactory solution.
The sum of mean square errors resembled a hyperbola like f(x) = 1/x (with an asymptote at infinity equal to 0 and with positive values for x in range (0, +inf)).

Considering that I chose the number of clusters "by eye" that looked most likely - it was 6.
After tuning the parameters, not every algorithm could cope with proper clustering. 
***Agglomerative Clustering***, ***KMeans*** and ***Spectral Clustereing*** has worked the best.
The results of which are presented below.

![best_clusterizers](img/best_clusterizers.png)

---

### Classification

During classification all labels of data were taken from the best clustering algorithm - ***Agglomerative Clustering***. 

During classification process **8** different algorithms were used:
1. Decision Tree
2. K-Nearest Neighbors
3. Logistic Regression
4. Multilayer Perceptron
5. Random Forest
6. SVM
7. Team Classification (team of classifiers)
8. Deep Neural Network (tensorflow)

All algorithms parameters were tuned using Grid Search method (which allows to automatically choices of params).
For all algorithms result chart of decision regions was created.

![classifiers_comparison](img/classifiers_comparison.png)


---

### Regression

---

### Knowledge and skills acquired

---

### Summary and general thoughts

---

### Technology used
+ Python 3.9.5
    + scikit-learn
    + pandas
    + tensorflow
    + matplotlib
    + numpy
    + pyprind

---

## License & copyright
© All rights reserved
