#This is for KNN
from sklearn.neighbors import KNeighborsClassifier
from commonFunction import *
import numpy as np
from numpy import linalg as LA
from math import exp

def knn(trainFeature, trainLabel, testFeature, k):
    predictedLabel = []
    for testSample in testFeature:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(trainFeature, trainLabel)
        predictedLabel.extend(list(neigh.predict(testSample)))
    return(predictedLabel)


'function: distance weighted KNN-the more distance, the less weight'
def distanceWeightedKNN(trainFeature, trainLabel, testFeature, k):
    trainSampleNum = trainFeature.shape[0]
    predictedLabel = []
    for testFeatureItem in testFeature:
        print('<============= a new test sample ===========>')
        distanceTupleList = []
        weight = np.ones(k)
        labels = np.zeros(k)
        i = 0
        while(i < trainSampleNum):
            #calculate the Euclidean distance between the testing item and the training items
            distance = LA.norm(abs(trainFeature[i] - testFeatureItem))
            distanceTupleList.append((trainLabel[i], distance))
            i += 1
        distanceTupleList = sorted(distanceTupleList, key = lambda x: x[1])
        if(distanceTupleList[0][1] != distanceTupleList[k-1][1]):
            nominator = distanceTupleList[k-1][1] - distanceTupleList[0][1]
            j = 0
            for distanceTuple in distanceTupleList[:k]:
                weight[j] = (distanceTupleList[k-1][1] - distanceTuple[1]) / float(nominator)
                labels[j] = distanceTuple[0]
                j += 1
        'this part will be used in binary classification'
        l = 0
        positiveWeight = 0
        negativeWeight = 0
        for label in labels[:k]:
            if(label == 1):
                positiveWeight += weight[l]
            else:
                negativeWeight += weight[l]
            l += 1
        if(positiveWeight > negativeWeight):
            predictedLabel.append(1)
        else:
            predictedLabel.append(0)
    predictedLabel = np.array(predictedLabel)
    return(predictedLabel)

'Euclidean distance weighted by gain ratio'
def gainRatioDistanceKNN(trainFeature, trainLabel, testFeature, k):
    infoGainRatio = gainRatio(trainFeature, trainLabel) #numpy.ndarray
    #print(infoGainRatio/sum(infoGainRatio))
    predictedLabel = []
    trainSampleNum = trainFeature.shape[0]
    for testFeatureItem in testFeature:
        distanceLabelTupleList = []
        i = 0
        while(i < trainSampleNum):
            distanceLabelTupleList.append((LA.norm(abs(testFeatureItem - trainFeature[i])*infoGainRatio/sum(infoGainRatio)), trainLabel[i]))
            i += 1
        distanceLabelTupleList = sorted(distanceLabelTupleList, key = lambda x: x[0])
        #print(distanceLabelTupleList)
        labels = [distanceLabelTuple[1] for distanceLabelTuple in distanceLabelTupleList]
        if(labels[:k].count(1) >= labels[:k].count(0)):
            predictedLabel.append(1)
        else:
            predictedLabel.append(0)
    predictedLabel = np.array(predictedLabel)
    return(predictedLabel)

'locally weighted averaging KNN'
def locallyWeightedAverageKNN(trainFeature, trainLabel, testFeature, times, k):
    'part one: find the minimum distance between any two points of the training set'
    minimumDistance = miniDis(trainFeature)
    kernelWidth = minimumDistance * float(times)
    trainSampleNum = trainFeature.shape[0]
    predictedLabel = []
    for testFeatureItem in testFeature:
        print('<============= a new test sample ===========>')
        distanceTupleList = []
        weight = np.ones(k)
        labels = np.zeros(k)
        i = 0
        while(i < trainSampleNum):
            #print('+++calculate the %dth Euclidean distance+++' % (i))
            distance = LA.norm(abs(trainFeature[i] - testFeatureItem))
            distanceTupleList.append((trainLabel[i], distance))
            i += 1
        distanceTupleList = sorted(distanceTupleList, key = lambda x: x[1])
        if(distanceTupleList[0][1] != distanceTupleList[k-1][1]):
            nominator = kernelWidth ** 2
            j = 0
            for distanceTuple in distanceTupleList[:k]:
                weight[j] = exp(-(distanceTuple[0] ** 2) / float(nominator))
                labels[j] = distanceTuple[0]
                j += 1
        'this part will be used in binary classification'
        l = 0
        positiveWeight = 0
        negativeWeight = 0
        for label in labels[:k]:
            if(label == 1):
                positiveWeight += weight[l]
            else:
                negativeWeight += weight[l]
            l += 1
        if(positiveWeight > negativeWeight):
            predictedLabel.append(1)
        else:
            predictedLabel.append(0)
    predictedLabel = np.array(predictedLabel)
    return(predictedLabel)
    

'find the minimum distance between any two points of the training set'
def miniDis(featureMatrix):
    sampleNum = featureMatrix.shape[0]
    i = 0
    j = 0
    minimumDistance = []
    while(i < (sampleNum-1)):
        j = i+1
        distance = []
        while(j < sampleNum):
            distance.append(LA.norm(abs(featureMatrix[i] - featureMatrix[j])))
            j += 1
        i += 1
        #print(sorted(distance))
        minimumDistance.append(min(distance))
    return(min(minimumDistance))

if __name__ == "__main__":
    folderNum = 5
    k = 10
    times = 2
    #dataFeature, dataLabel = read_GermanData('./Data/german/german.data-numeric')    #class=0 means good credit, class=1 means bad credit
    dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
    featureFolder, labelFolder = crossValidation(dataFeature, dataLabel, folderNum)
    (accu1, accu2) = crossValidationFunc(featureFolder, labelFolder, knn, k)
    print(accu1, accu2)
