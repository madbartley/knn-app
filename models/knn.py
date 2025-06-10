# This project is based on the code from this Kaggle project: https://www.kaggle.com/code/danishyousuf19/breast-cancer-detection-using-knn-svc/notebook

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# for saving the trained model
import pickle

# for saving our standard scaler
from joblib import dump, load

'''
# Loading and preprocessing the data
# Import the csv file with all the data, and call it "data" - we're going to create some data frames to quickly get some info about the data in the csv
data = pd.read_csv("KNNAlgorithmDatasetTrainTest.csv")
# View the first 10 rows of the file + column names
data.head(10)
# For each column, we will get info on mean, std dev, and percentiles
data.describe()
# We want to make sure that each data point contains an entry for each column, there are no "null" entries, and see the data types for each column
data.info()
# Labelling our data with either 1 or 0
# Now we want to "map" our classification to the kind of diagnosis - Here, they're using "M" to mean there was a cancer diagnosis, and "B" to mean that there was NOT a cancer diagnosis for each data point
# We want to change that "M" vs "B" classification into something that our program can understand - we'll map the "M" to "1", meaning cancer, and the "B" to "0", meaning not cancer

# NOTE: When mapping, the mapped values must match the file values EXACTlY - case sensitive etc - and if double quotes doesn't work, try single quotes
# If you wind up with a column of "NaN"s, it probably means you used the wrong quotation or missed a case or a space somwhere

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

# And we can check that this was successful by viewing another data frame, this time from the end of the file, since the first 10 entries were only "M" before
# We'll also specify that we only want to view the "diagnosis" column, since we're just verifying our mapping right now

data['diagnosis'].tail(10)
# We saw earlier in our earlier data frame that there was an unnamed column at the end of the csv, full of NaN - let's remove that
data = data.dropna(axis=1)
# Let's also drop the "id" column, since the ID of each data point doesn't tell us anything useful about the samples
data = data.drop('id',axis=1)
data.head(10)
# Testing and Training the data
# Now we want to separate the data into our samples, X, without the labels, and our known labels, Y
X = data.drop('diagnosis',axis=1)
y = data['diagnosis']
# We can now see that our X is just the samples' data, without the labels
X.head(10)
# We can also look at our Y to make sure we got the labels correctly split
y.tail(10)

# To create the testing and training datasets, we'll use a scikit-learn method called "train_test_split()"
# This will randomize our datasets, and let us customize our sets. More info here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print(X_train.shape)
print(y_train.shape)

# Next, scaling the data
# KNN is sensitive to the size of data inputs - with StandardScaler() we perform the following transformation: z = (x - mean) / (std. dev.)
# This converts the mean to be 0, and normalizes the standard deviation to 1, which makes the data easier to work with.
# After scaler = StandardScaler() has been declared, we use it with fit_transform() and the training data, and then transform the tst data
# We only need to fit to the training data, as "fit" means learning the calculations to do to compute the mean and std. dev. - we wouldn't want to relearn a different set to transform the testing data in a different way than the training data, or this will skew our result

scaler = StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
dump(scaler, 'std_scaler.bin', compress=True)

# Finally, we can specify the number of neighbors and create the model

### uncomment to retrain the model 
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_std,y_train)


##### uncomment this code after training the model to go into load-mode
knnPickle = open('knnpickle_file', 'wb')

# source, destination 
pickle.dump(model, knnPickle) 

# close the file
knnPickle.close()

'''


##### commenting this out since our model is trained - this is just for reference
# Predicting
# Now we can try out our predictions using the test data
# We'll test the data, then we'll look at the data by reshaping our array into 10 samples of just 1 dimension, the predicition
'''

predictions = model.predict(X_test_std)
prnt = predictions[:10].reshape(10,1)
prnt
print(metrics.classification_report(y_test,predictions))
newDataM = [[17.99,	10.38,	122.8,	1001, 0.1184, 0.2776, 0.3001,  0.1471,	0.2419,	0.07871, 1.095,	0.9053,	8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,	0.006193, 25.38, 17.33,	184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]
newDataMS = scaler.transform(newDataM)
predictions = model.predict(newDataMS)
prnt = predictions.reshape(1,1)
prnt

newDataB = [[12.36,	21.8, 79.78, 466.1, 0.08772, 0.09445, 0.06015, 0.03745, 0.193, 0.06404, 0.2978, 1.502, 2.203, 20.95, 0.007112, 0.02493, 0.02703, 0.01293, 0.01958, 0.004463, 13.83, 30.5, 91.46, 574.7, 0.1304, 0.2463, 0.2434, 0.1205, 0.2972, 0.09261]]
newDataBS = scaler.transform(newDataB)

predictions2 = model.predict(newDataBS)
prnt = predictions2.reshape(1,1)
prnt



### taking predicitions from loaded model

# load the model from disk
loaded_model = pickle.load(open('knnpickle_file', 'rb'))
scaler = StandardScaler()
# make sure to scale the data
input_std = scaler.transform(user_input)
# predictions = loaded_model.predict(input_std)
# print(predictions)
'''