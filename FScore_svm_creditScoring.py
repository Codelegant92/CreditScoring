import numpy as np
from commonFunction import *
from svm_classification import svm_GridSearch_creditScore

#function: sort a list, select first k elements and corresponding Index
def kMaximumElem(numList, K):
    numTupleList = []
    for i, item in enumerate(numList):
        numTupleList.append((i, item))
    numTupleList = sorted(numTupleList, key = lambda x : -x[1]) #from large to small
    return(numTupleList[:K])

#function: select high score features to be new training and validation data
def fScore_SVM(dataFeature, dataLabel, fScore, K):
    first_K_fScore = sorted(kMaximumElem(fScore, K), key = lambda x: x[0])
    print(first_K_fScore)
    selected_dataFeature = np.array([[rows[item[0]] for item in first_K_fScore] for rows in dataFeature])
    selected_dataLabel = dataLabel
    svm_GridSearch_creditScore(selected_dataFeature, selected_dataLabel)

dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
fScore = FScore(dataFeature, dataLabel)

K = 24
fScore_SVM(dataFeature, dataLabel, fScore, 24)
'''
fScore_SVM(dataFeature, dataLabel, fScore, 12)
fScore_SVM(dataFeature, dataLabel, fScore, 6)
fScore_SVM(dataFeature, dataLabel, fScore, 3)
fScore_SVM(dataFeature, dataLabel, fScore, 1)
'''