#This is a series of classification algorithms implementation
#SVM for Classification
'''
f = open('./Data/german/german.data-numeric')
for item in f.readlines():
    print(item.split('\t')[0])
'''
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



'''
trainMatrix = dataFeature[:800]
testMatrix = dataFeature[800:]
trainLabel = dataLabel[:800]
testLabel = dataLabel[800:]
'''
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
    
#function: SVM algorithm - training and testing data, parameter C and gamma are given, when the output is the predicted class
#def svm_algorithm(trainingFeature, traningLabel, testingFeature, testingLabel, kernelType, paraC, paraGamma):
#     clf = svm.SVC(C = paraC, kernel = kernelType, )
'''
clf = svm.SVC()
clf.fit(trainMatrix, trainLabel)
print(clf)
test = clf.predict(testMatrix)
print((list(test == testLabel).count(True))*1.0/len(test))
print(clf.support_vectors_)
print(clf.n_support_)
print(clf.support_)
'''

folderNum = 5
dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
#dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
'''
clf = svm.SVC(C=512, gamma=0.000244140625, cache_size=600)
clf.fit(dataFeature[:800, :], dataLabel[:800])
predictedLabel = clf.predict(dataFeature[800:, :])
diff = list(dataLabel[800:] - predictedLabel)
type_one_error = diff.count(1)/float(list(dataLabel[800:]).count(1))
type_two_error = diff.count(-1)/float(list(dataLabel[800:]).count(0))
print(1-type_one_error, 1-type_two_error)
'''
def svmclassifier(trainFeature, trainLabel, testFeature, testLabel):
    clf = svm.SVC(C = 512, gamma = 0.000244140625)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    diff = list(testLabel - predictedLabel)
    type_one_error = diff.count(1)/float(list(testLabel).count(1))
    type_two_error = diff.count(-1)/float(list(testLabel).count(0))
    return(1-type_one_error, 1-type_two_error)

def crossValidation_svm(dataFeature, dataLabel, folderNum):
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
        cross_accuracy.append(svmclassifier(trainFeature, trainLabel, testFeature, testLabel))
    accu1 = 0
    accu2 = 0
    for j in range(len(cross_accuracy)):
        accu1 += cross_accuracy[j][0]
        accu2 += cross_accuracy[j][1]
    return(accu1/float(j+1), accu2/float(j+1))  
  
if(__name__ == "__main__"):
    #svm_creditScore()
    #svm_GridSearch_creditScore(dataFeature, dataLabel)
    
    crossValidation_svm(dataFeature, dataLabel, folderNum)
    accu1 = 0
    accu2 = 0
    for i in range(10):
        accu1 += crossValidation_svm(dataFeature, dataLabel, folderNum)[0]
        accu2 += crossValidation_svm(dataFeature, dataLabel, folderNum)[1]
    print(accu1/10.)
    print(accu2/10.)
    