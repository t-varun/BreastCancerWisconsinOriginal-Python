import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

classes = ['Benign', 'Malignant']
df = pd.read_csv('Breast_Cancer.data')
df.replace('?', '-99999', inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)

example = np.array([9,4,5,4,6,1,2,3,7])
example = example.reshape(len(example), -1)
prediction = classifier.predict(example)

if prediction[0] == 2:
	print(classes[0])
elif prediction[0] == 4:
	print(classes[1])
