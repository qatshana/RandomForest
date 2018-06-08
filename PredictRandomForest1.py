'''
81 samples. using Random Forests

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Load dataset from csv file
df = pd.read_csv('kyphosis.csv')


# get basic information about the dataset
df.head()
df.describe()
df.info()

# get dataset columns
df.columns


# Visualize data
sns.pairplot(df,hue='Kyphosis',palette='Set1')



#Training the model

# define inputs (ignore address)
X=df.drop('Kyphosis',axis=1)

#define output file
y=df['Kyphosis']

#Create train and est sets using Train Test Split, 40% test data and 60% training data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train Model -Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Perform predictions

pred = dtree.predict(X_test)

# Model Evaluation - 
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# Tree Visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot  #conda install -c rmg pydot    
				# conda install graphviz

features = list(df.columns[1:])
features

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  
