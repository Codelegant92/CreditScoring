#This is for logistic regression
from commonFunction import *
from sklearn import linear_model

def logistic_regression(trainFeature, trainLabel, testFeature, testLabel):
    clf = linear_model.LogisticRegression(penalty='l2', dual=False)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    diff = list(testLabel - predictedLabel)
    type_one_error = diff.count(1)/float(list(testLabel).count(1))
    type_two_error = diff.count(-1)/float(list(testLabel).count(0))
    return(1-type_one_error, 1-type_two_error)

def crossValidation_logisticRegression(dataFeature, dataLabel, folderNum):
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
        cross_accuracy.append(logistic_regression(trainFeature, trainLabel, testFeature, testLabel))
    accu1 = 0
    accu2 = 0
    for j in range(len(cross_accuracy)):
        accu1 += cross_accuracy[j][0]
        accu2 += cross_accuracy[j][1]
    return(accu1/float(j+1), accu2/float(j+1))
folderNum = 5
dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
#dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
trainFeature = dataFeature[:800, :]
print(trainFeature[0, :])
trainLabel = dataLabel[:800]
testFeature = dataFeature[800:, :]
testLabel = dataLabel[800:]
for i in range(len(dataLabel)):
    if(dataLabel[i] == -1):
        dataLabel[i] = 0
#print(logistic_regression(trainFeature, trainLabel, testFeature, testLabel))
accu1 = 0
accu2 = 0
for i in range(10):
    accu1 += crossValidation_logisticRegression(dataFeature, dataLabel, folderNum)[0]
    accu2 += crossValidation_logisticRegression(dataFeature, dataLabel, folderNum)[1]
print(accu1/10.)
print(accu2/10.)