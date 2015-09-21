# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:54:17 2015

@author: yangchen
"""

#These are common-used functions
import numpy as np
import random
import math

class dataset():
    def __init__(self, filename):
        self.file = filename
    
    def loadGermanData(self):
        f = open(self.file)
        userProfile = []
        singleProfile = []
        data = ''
        for line in f.readlines():
            for i in range(len(line)):
                if((i == len(line)-1)):
                    if(line[i] != '\n'):
                        data += line[i]                
                        singleProfile.append(data)
                elif(line[i] != ' '):
                    data += line[i]
                    if(line[i+1] == ' '):
                        singleProfile.append(int(data))
                        data = ''
            userProfile.append(singleProfile)    
            singleProfile = []
        dataMatrix = np.array(userProfile)
        n, m = dataMatrix.shape
        roughLabel = dataMatrix[:, m-1]
        label = []
        for item in roughLabel:
            if(item == 2):
                label.append(0)
            else:
                label.append(1)
        featureMatrix = dataMatrix[:, 0:m-1] #numpy.ndarray
        label = np.array(label) #numpy.ndarray
        return(featureMatrix, label)

    def loadAustralianData(self):
        f = open(self.file)
        userProfile = []
        singleProfile = []
        for line in f.readlines():
            singleProfile = line.split(' ')
            if(singleProfile[-1] == '0\n'):
                singleProfile[-1] = 0
            else:
                singleProfile[-1] = 1
            singleProfile = [float(singleProfileItem) for singleProfileItem in singleProfile]
            userProfile.append(singleProfile)
        featureMatrix = np.array(userProfile)[:, :-1]
        label = np.array(userProfile)[:, -1]
        return(featureMatrix, label)

    def dataForLIBSVM(self):
        m, n = self.featureMatrix.shape
        f = open('stalogGerman.txt', 'w')
        for i in range(m):
            string = str(self.label[i])
            for j in range(n):
                string = string + ' ' + str(j+1) + ':' + str(self.featureMatrix[i, j])
            string += '\n'
            f.write(string)
        f.close()
        
#function: read data from given files, parameter is the file path
#for the label, 1 represents + and 0 represents -
def read_GermanData(data_filePath):
    data = dataset(data_filePath)
    dataFeature, dataLabel = data.loadGermanData()
    return(dataFeature, dataLabel) #both are ndarray type

#function: read the file-australian.dat
#for the label, 1 represents + and 0 represents -
def readAustralianData(data_filePath):
    data = dataset(data_filePath)
    dataFeature, dataLabel = data.loadAustralianData()
    return(dataFeature, dataLabel) #both are ndarray type

#function: write a uniform program of cross validation-randomly divide the dataset into folders
def crossValidation(featureMatrix, featureLabel, folderNum):
    instanceNum = featureMatrix.shape[0]
    n = instanceNum / folderNum
    folderNum -= 1
    sequence = range(instanceNum)
    random.shuffle(sequence)
    randomFolders = [sequence[(n*i):(n*(i+1))] for i in xrange(folderNum)]
    randomFolders.append(sequence[(n*folderNum):])
    randomFeatureMatrix = [np.array([list(list(featureMatrix)[j]) for j in folderList]) for folderList in randomFolders]
    randomFeatureLabel = [np.array([list(featureLabel)[k] for k in folderList]) for folderList in randomFolders]
    return(randomFeatureMatrix, randomFeatureLabel)#randomFeatureMatrix:a list of ndarray matrix [array([[],[],...,[]]), array([[],...,[]]),...,array([[],...,[]])]
                                                   #randomFeatureLabel:a list of ndarray [array([]), ..., array([])]

def crossValidationFunc(featureFolder, labelFolder, func, *args):
    cross_accuracy = []
    folderNum = len(labelFolder)
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
        print("=====CV %d==========") % (i+1)
        print("Training samples:")
        print("Positive: %d, Negative: %d") % (list(trainLabel).count(1), list(trainLabel).count(0))
        print("Testing samples:")
        print("Positive: %d, Negative: %d") % (list(testLabel).count(1), list(testLabel).count(0))
        predictedLabel = func(trainFeature, trainLabel, testFeature, *args)
        diff = list(testLabel - predictedLabel)
        type_one_error = diff.count(1)/float(list(testLabel).count(1))
        type_two_error = diff.count(-1)/float(list(testLabel).count(0))
        cross_accuracy.append((1-type_one_error, 1-type_two_error))
    print(cross_accuracy)
    accu1 = 0
    accu2 = 0
    for j in range(len(cross_accuracy)):
        accu1 += cross_accuracy[j][0]
        accu2 += cross_accuracy[j][1]
    return(accu1/float(j+1), accu2/float(j+1))


def FScore(featureMatrix, featureLabel):
    posMatrix = []
    negMatrix = []
    i = 0
    for label in featureLabel:
        if(label == 1):
            posMatrix.append(featureMatrix[i])
        else:
            negMatrix.append(featureMatrix[i])
        i += 1
    featureNum = featureMatrix.shape[1]
    featureMean = featureMatrix.mean(0) #average value of each feature
    posMatrix = np.array(posMatrix)
    negMatrix = np.array(negMatrix)
    posNum = posMatrix.shape[0] #number of positive instances
    negNum = negMatrix.shape[0] #number of negative instances
    #print('pos:', posNum)
    #print('neg:', negNum)
    posFeatureMean = posMatrix.mean(0) #average value of positive instances
    negFeatureMean = negMatrix.mean(0) #average value of negative instances
    #print(posFeatureMean)
    #print(negFeatureMean)
    #print((posFeatureMean+negFeatureMean)/2-featureMean)
    posSquareSum = [] #variance of positive instances
    negSquareSum = [] #variance of negative instances
    for i in range(featureNum):
        sum = 0
        for j in range(posNum):
            sum = sum + (posMatrix[j][i] - posFeatureMean[i]) ** 2
        posSquareSum.append(sum)
    #print(posSquareSum) #because the data has not been scaled, some averages will be much larger than others
    for i in range(featureNum):
        sum = 0
        for j in range(negNum):
            sum = sum + (negMatrix[j][i] - negFeatureMean[i]) ** 2
        negSquareSum.append(sum)
    #print(negSquareSum)
    Fscore = []
    for i in range(featureNum):
        Fscore.append(((posFeatureMean[i]-featureMean[i])**2+(negFeatureMean[i]-featureMean[i])**2)/(posSquareSum[i]/posNum + negSquareSum[i]/negNum))
    print(Fscore)
    return(Fscore)

#function: calculate the gain ratio of each Feature
def gainRatio(trainFeature, trainLabel):
    sampleNum, featureNum = trainFeature.shape
    uniqueClassList = list(set(trainLabel))
    uniqueFeatureValueList = [list(set(trainFeature[:, i])) for i in range(featureNum)]
    'calculate the entropy'
    Ent = 0
    for item in uniqueClassList:
        prob = list(trainLabel).count(item) / float(sampleNum)
        Ent += -prob * math.log(prob, 2) #the base of logarithm is 2
    'calculate the conditional entropy'
    informationGainRatioList = []
    for i in range(featureNum):
        conditionalEnt, featureEnt = condiEntropy(trainFeature[:, i], trainLabel, uniqueClassList, uniqueFeatureValueList[i])
        informationGainRatioList.append((Ent-conditionalEnt) / featureEnt)
    return(np.array(informationGainRatioList))
   
#function: calculate the conditional entropy of each partition of the Feature
def condiEntropy(trainFeatureCol, trainLabel, uniqueClassList, uniqueFeatureValue):
    featureLabelTupleList = [(trainFeatureCol[i], trainLabel[i]) for i in range(trainFeatureCol.shape[0])]
    featureLabelTupleList = sorted(featureLabelTupleList, key = lambda x: x[0])
    sampleNum = len(featureLabelTupleList)
    uniqueClassList = sorted(uniqueClassList)
    uniqueFeatureValue = sorted(uniqueFeatureValue)
    i = 0
    featureLabelTuplePartList = []
    featureLabelTupleTemp = []
    for item in featureLabelTupleList:
        if(item[0] == uniqueFeatureValue[i]):
            featureLabelTupleTemp.append(item)
        else:
            featureLabelTuplePartList.append(featureLabelTupleTemp)
            featureLabelTupleTemp = []
            featureLabelTupleTemp.append(item)
            i += 1
    featureLabelTuplePartList.append(featureLabelTupleTemp)
    #print(featureLabelTuplePartList)
    Prob_EachFeatureValueList = []
    Ent_EachFeatureValueList = []
    labelDict = dict()
    i = 0
    for label in uniqueClassList: #the classList is sorted
        labelDict[label] = i
        i += 1
    for featureLabelTuplePart in featureLabelTuplePartList:
        Num_EachFeatureValue = len(featureLabelTuplePart)
        Prob_EachFeatureValueList.append(Num_EachFeatureValue / float(sampleNum))
        num_classEachFeatureValue = [0 for j in range(len(uniqueClassList))]
        for featureLabelTuple in featureLabelTuplePart:
            num_classEachFeatureValue[labelDict[featureLabelTuple[1]]] += 1
        Ent_EachFeatureValue = 0
        for num in num_classEachFeatureValue:
            prob_classEachFeatureValue = num / float(Num_EachFeatureValue)
            if(prob_classEachFeatureValue != 0):
                Ent_EachFeatureValue += (-prob_classEachFeatureValue * math.log(prob_classEachFeatureValue, 2))
        Ent_EachFeatureValueList.append(Ent_EachFeatureValue)
    #print(Prob_EachFeatureValueList)
    #print(Ent_EachFeatureValueList)
    conditionalEntropy = 0
    featureEntropy = 0
    for i in range(len(Prob_EachFeatureValueList)):
        conditionalEntropy += Prob_EachFeatureValueList[i] * Ent_EachFeatureValueList[i]
        featureEntropy += (-Prob_EachFeatureValueList[i] * math.log(Prob_EachFeatureValueList[i], 2))
    #print(conditionalEntropy, featureEntropy)
    return(conditionalEntropy, featureEntropy)

if __name__ == "__main__":
    '''
    dataFeature, dataLabel = read_Germandata('./Data/german/german.data-numeric')
    print(dataFeature[0,:])
    '''

    dataFeature1, dataLabel1 = readAustralianData('./Data/Australia/australian.dat')
    #print(dataFeature1[689, :])

    a, b = crossValidation(dataFeature1[0:5, :], dataLabel1[0:5], 2)
    print(a)
    print(b)