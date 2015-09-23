from commonFunction import *
import numpy as np
import time
from sklearn import tree
from sklearn.externals.six import StringIO
import os
import pydot

def decision_Tree(trainFeature, trainLabel, testFeature):
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    '''
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('./Data/test.pdf')
    '''
    return(predictedLabel)


if __name__ == "__main__":
    folderNum = 5
    #dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
    #dataFeature, dataLabel = read_GermanData('./Data/german/german.data-numeric')
    dataFeature, dataLabel = read_GermanData20('./Data/german/german.data')
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    (accu1, accu2) = crossValidationFunc(featureFolder, labelFolder ,decision_Tree)
    print(accu1, accu2)