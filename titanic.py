###   0    1                               2        3      4          5         6
###  "", "Name",                        "PClass", "Age", "Sex",   "Survived","SexCode"
### "1", "Allen, Miss Elisabeth Walton","1st",     29,   "female", 1,            1

import csv
import pandas
import numpy
import scipy
import types
import math
NumberTypes = (types.IntType, types.LongType, types.FloatType, types.ComplexType)

colnames = ['number', 'name', 'clas', 'age', 'sex', 'survived', 'sexcode']
datad = pandas.read_csv('titanic_data.csv', names=colnames)

#remove first row because it's a header
data = datad.iloc[1:]


#data = data[:13] ###################REMOVE THIS TODO


#extract survived information and use as a label
labels = data.survived.tolist()
for l in labels:
	l = float(l)
	#print isinstance(l, NumberTypes)
#print labels
print "-"


#delete useless column for feature
del data['number']
del data['name']
del data['sex']
del data['survived']
feature = data.values.tolist()
for f in feature:
	#print f[0]
	f[0] = f[0][0]
	#print f[0]
	if f[0] == "*":
		f[0] = -1
	f[0] =  float(f[0])
	#print f[0]

	f[1] =  float(f[1])
	if math.isnan(f[1]):
		f[1] = 0
	#print f[1]

	f[2] =  float(f[2])
	#print f[2]
	
	#print isinstance(f[0], NumberTypes)
	#print isinstance(f[1], NumberTypes)
	#print isinstance(f[2], NumberTypes)
#print feature
print "--"


from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(feature, labels)#, shuffle=False)
#print features_train
#print labels_train
#print
#print features_test
#print labels_test
#print "---"


from sklearn import svm
clf_svc = svm.LinearSVC()
clf_svc.fit(features_train, labels_train)
print "--LinearSVC--"
print "Accuracy on trainig data", clf_svc.score(features_train, labels_train)
print "Accuracy on test data", clf_svc.score(features_test,labels_test)
print


from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(features_train, labels_train)
print "--DecisionTree--"
print "Accuracy on train: ", clf_dt.score(features_train, labels_train)
print "Accuracy on test: ", clf_dt.score(features_test, labels_test)



print
print "Predicting male of 26yo travelling in 3rd class"
print(clf_dt.predict([[3.0, 26.0, 0.0]]))
print "Predicting female of 26yo travelling in 3rd class"
print(clf_dt.predict([[3.0, 26.0, 1.0]]))

print
print "Predicting male of 26yo travelling in 2rd class"
print(clf_dt.predict([[2.0, 26.0, 0.0]]))
print "Predicting female of 26yo travelling in 2rd class"
print(clf_dt.predict([[2.0, 26.0, 1.0]]))

print
print "Predicting male of 26yo travelling in 1rd class"
print(clf_dt.predict([[1.0, 26.0, 0.0]]))
print "Predicting female of 26yo travelling in 1rd class"
print(clf_dt.predict([[1.0, 26.0, 1.0]]))