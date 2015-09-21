#This is for logistic regression
from commonFunction import *
from sklearn import linear_model

def logistic_regression(trainFeature, trainLabel, testFeature):
    clf = linear_model.LogisticRegression(penalty='l2', dual=False)
    clf.fit(trainFeature, trainLabel)
    predictedLabel = clf.predict(testFeature)
    return(predictedLabel)

if(__name__ == "__main__"):
    folderNum = 5
    dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
    #dataFeature, dataLabel = read_GermanData('./Data/german/german.data-numeric')
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    accu1, accu2 = crossValidationFunc(featureFolder, labelFolder, logistic_regression)
    print(accu1, accu2)