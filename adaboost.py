from commonFunction import *
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
trainFeature = dataFeature[:800, :]
trainLabel = dataLabel[:800]
testFeature = dataFeature[800:, :]
testLabel = dataLabel[800:]

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm = 'SAMME', n_estimators = 200)
bdt.fit(trainFeature, trainLabel)
predictedLabel = bdt.predict(testFeature)
diff = list(testLabel - predictedLabel)
type_one_error = diff.count(1)/float(list(testLabel).count(1))
type_two_error = diff.count(-1)/float(list(testLabel).count(0))
print(1-type_one_error, 1-type_two_error)
#print(list(np.array(predictedLabel)-testLabel).count(0)/float(200))