#This is a series of classification algorithms implementation
#SVM for Classification

from commonFunction import *
from sklearn import svm, grid_search
import numpy as np
import time
from functools import wraps

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" % (function.func_name, str(t1-t0)))
        return result
    return function_timer()

#@fn_timer
def svm_GridSearch_creditScore(dataFeature, dataLabel):
    start = time.clock()
    C_s = np.logspace(-5, 12, 18, True, 2)
    gamma_para = np.logspace(-12, 5, 18, True, 2)
    parameters = {'C':C_s, 'gamma':gamma_para}
    svc = svm.SVC(cache_size=600)
    clf = grid_search.GridSearchCV(svc, parameters, cv = 5)
    clf.fit(dataFeature, dataLabel)
    scoreList = clf.grid_scores_[0]
    print(scoreList)
    print('best score:')
    print(clf.best_score_)
    print('best estimator:')
    print(clf.best_estimator_)
    print('best_params:')
    print(clf.best_params_)
    end = time.clock()
    print("Time consuming: %f" % (end-start))
#German data best parameters: C=512, gamma=0.000244140625
#Australian data best parameters: C = 128, gamma = 0.00048828125
#function: SVM algorithm - training and testing data, parameter C and gamma are given, when the output is the predicted class
#def svm_algorithm(trainingFeature, traningLabel, testingFeature, testingLabel, kernelType, paraC, paraGamma):
#     clf = svm.SVC(C = paraC, kernel = kernelType, )

def svmclassifier(trainFeature, trainLabel, testFeature, para_C, para_gamma):
    clf = svm.SVC(C = para_C, gamma = para_gamma)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

if(__name__ == "__main__"):
    folderNum = 5
    para_C = 128
    para_gamma = 0.00048828125
    dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
    #dataFeature, dataLabel = read_GermanData('./Data/german/german.data-numeric')
    #svm_GridSearch_creditScore(dataFeature, dataLabel)
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, svmclassifier, para_C, para_gamma)
    print(accu1, accu2)
