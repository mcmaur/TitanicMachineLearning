###   0    1                               2        3      4          5         6
###  "", "Name",                        "PClass", "Age", "Sex",   "Survived","SexCode"
### "1", "Allen, Miss Elisabeth Walton","1st",     29,   "female", 1,            1

import csv
import pandas
colnames = ['number', 'name', 'clas', 'age', 'sex', 'survived', 'sexcode']
datad = pandas.read_csv('titanic_data.csv', names=colnames)

#remove first line of data because it's a header
data = datad.iloc[1:]


data = data[:5] ###################REMOVE THIS TODO

#extract survived information and use as a label
labels = data.survived.tolist()
print labels
print "-"


#delete useless column for feature
del data['number']
del data['name']
del data['sex']
del data['survived']
feature = data.values.tolist()
for f in feature:
	f[0] = f[0][0]
print feature
print "--"



from sklearn import model_selection
feature_train, feature_test, label_train, label_test = model_selection.train_test_split(feature, labels)#, shuffle=False)
print feature_train
print label_train
print
print feature_test
print label_test
print "---"



#clf = DecisionTreeClassifier()
#clf.fit(features_train, labels_train)


from sklearn import svm
clf = svm.LinearSVC()
clf.fit(feature_train, label_train)
print(clf.coef_)
print(clf.intercept_) 

#print "Accuracy on trainig data", clf.score(feature_train, label_train)
#print "Accuracy on test data", clf.score(feature_test,label_test)

































### Task 1: Select what features you'll use.
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### Task 6: Dump your classifier, dataset, and features_list so anyone can