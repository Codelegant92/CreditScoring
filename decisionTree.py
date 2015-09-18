from commonFunction import *
import numpy as np
import time
from sklearn import tree
from sklearn.externals.six import StringIO
import os
import pydot

def decision_Tree(trainFeature, trainLabel, testFeature, testLabel):
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    '''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('./Data/test.pdf')
    '''

    #return(list(predictedLabel == testLabel).count(True)*1.0/len(testLabel))
    diff = list(testLabel - predictedLabel)
    type_one_error = diff.count(1)/float(list(testLabel).count(1))
    type_two_error = diff.count(-1)/float(list(testLabel).count(0))
    return(1-type_one_error, 1-type_two_error)

    '''
    return(predictedLabel)
    '''
def crossValidation_decisionTree(dataFeature, dataLabel, folderNum):
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    cross_accuracy = []
    for i in range(folderNum):
        testFeature = featureFolder[i]
        testLabel = labelFolder[i]
        trainFeature = []
        trainLabel = []
        for j in range(folderNum):
            if(j != i):
                trainFeature.extend(list(featureFolder[j]))
                trainLabel.extend(list(labelFolder[j]))
        trainFeature = np.array(trainFeature)
        trainLabel = np.array(trainLabel)
        cross_accuracy.append(decision_Tree(trainFeature, trainLabel, testFeature, testLabel))
    accu1 = 0
    accu2 = 0
    for j in range(len(cross_accuracy)):
        accu1 += cross_accuracy[j][0]
        accu2 += cross_accuracy[j][1]
    return(accu1/float(j+1), accu2/float(j+1))    
#print(decision_Tree(dataFeature[:500], dataLabel[:500], dataFeature[500:1000], dataLabel[500:1000]))

folderNum = 5
dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
#dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
trainFeature = dataFeature[:500, :]
trainLabel = dataLabel[:500]
testFeature = dataFeature[500:, :]
testLabel = dataLabel[500:]
a = dataFeature[:, 9]
'''
print(list(set(a)))
accu1, accu2 = decision_Tree(trainFeature, trainLabel, testFeature, testLabel)
print(accu1, accu2)
'''

accu1 = 0
accu2 = 0
for i in range(10):
    accu1 += crossValidation_decisionTree(dataFeature, dataLabel, folderNum)[0]
    accu2 += crossValidation_decisionTree(dataFeature, dataLabel, folderNum)[1]
print(accu1/10.)
print(accu2/10.)
print(accu1/20. + accu2/20.)

'''
(accu1, accu2) = crossValidationFunc(dataFeature, dataLabel, folderNum, decision_Tree)
print(accu1)
print(accu2)
print((accu1 + accu2)/2.0)
'''
# print(np.array(accu).mean(0))