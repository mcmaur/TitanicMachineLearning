###   0    1                               2        3      4          5         6
###  "", "Name",                        "PClass", "Age", "Sex",   "Survived","SexCode"
### "1", "Allen, Miss Elisabeth Walton","1st",     29,   "female", 1,            1

import csv
import pandas
import numpy
import scipy
import types
import math
from sklearn import model_selection
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

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
# composition of feature(clas, age, sexcode)
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


features_train, features_test, labels_train, labels_test = model_selection.train_test_split(feature, labels)#, shuffle=False)
#print features_train
#print labels_train
#print
#print features_test
#print labels_test
#print "---"


clf_svc = svm.LinearSVC()
clf_svc.fit(features_train, labels_train)
print "--LinearSVC--"
print "Accuracy on trainig data", clf_svc.score(features_train, labels_train)
print "Accuracy on test data", clf_svc.score(features_test,labels_test)
print


clf_dt = DecisionTreeClassifier()
clf_dt.fit(features_train, labels_train)
print "--DecisionTree--"
print "Accuracy on train: ", clf_dt.score(features_train, labels_train)
print "Accuracy on test: ", clf_dt.score(features_test, labels_test)
print



print "--EXAMPLES--"
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


#generating all possible combination
	#clas 1 - 2 - 3
	#age 0 --> 99
	#sexcode 0 - 1

resultFemales = []
resultMales = []

resultClass1 = []
resultClass2 = []
resultClass3 = []
for i in range(1, 100):
	# composition of feature(clas, age, sexcode)

#clas 1
	#sexcode 0
	res = clf_dt.predict([[1.0, i, 0.0]])
	resultMales.append(res)
	resultClass1.append(res)

	#sexcode 1
	res = clf_dt.predict([[1.0, i, 1.0]])
	resultFemales.append(res)
	resultClass1.append(res)

#clas 2
	#sexcode 0
	res = clf_dt.predict([[2.0, i, 0.0]])
	resultMales.append(res)
	resultClass2.append(res)

	#sexcode 1
	res = clf_dt.predict([[2.0, i, 1.0]])
	resultFemales.append(res)
	resultClass2.append(res)

#clas 3
	#sexcode 0
	res = clf_dt.predict([[3.0, i, 0.0]])
	resultMales.append(res)
	resultClass3.append(res)

	#sexcode 1
	res = clf_dt.predict([[3.0, i, 1.0]])
	resultFemales.append(res)
	resultClass3.append(res)

print
print "--RESULTS--"
print("{:.1%} of the females would have survived".format(resultFemales.count('1') / float(len(resultFemales))))
print("{:.1%} of the males would have survived".format(resultMales.count('1') / float(len(resultMales))))
print
print("{:.1%} of first class would have survived".format(resultClass1.count('1') / float(len(resultClass1))))
print("{:.1%} of second class would have survived".format(resultClass2.count('1') / float(len(resultClass2))))
print("{:.1%} of third class would have survived".format(resultClass3.count('1') / float(len(resultClass3))))


#data = numpy.random.rand(6, 3)
#print data

#df = pandas.DataFrame(data, columns=['A', 'B', 'C'])
#row = df.ix[5]
#row.plot(kind='bar')#, filename='bar-chart-row')

#import matplotlib.pyplot as plt
#plt.show()