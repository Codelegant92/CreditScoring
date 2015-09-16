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
    
    def loadData(self):
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
        self.featureMatrix = dataMatrix[:, 0:m-1] #numpy.ndarray
        self.label = np.array(label) #numpy.ndarray

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
def read_data(data_filePath):
    data = dataset(data_filePath)
    data.loadData()
    dataFeature = data.featureMatrix
    dataLabel = data.label
    return dataFeature, dataLabel

#function: read the file-australian.dat
def readAustralianData(filePath):
    f = open(filePath)
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
    dataFeature = np.array(userProfile)[:, :-1]
    dataLabel = np.array(userProfile)[:, -1]
    return(dataFeature, dataLabel)

#function: write a uniform program of cross validation-randomly divide the dataset into folders
def crossValidation(featureMatrix, featureLabel, folder):
    instanceNum = featureMatrix.shape[0]
    n = instanceNum / folder
    sequence = range(instanceNum)
    random.shuffle(sequence)
    randomFolders = [sequence[(n*i):(n*(i+1))] for i in xrange(folder)]
    randomFeatureMatrix = [np.array([list(list(featureMatrix)[j]) for j in folderList]) for folderList in randomFolders]
    randomFeatureLabel = [np.array([list(featureLabel)[k] for k in folderList]) for folderList in randomFolders]
    return(randomFeatureMatrix, randomFeatureLabel)#randomFeatureMatrix:a list of ndarray matrix [array([[],[],...,[]]), array([[],...,[]]),...,array([[],...,[]])]
                                                   #randomFeatureLabel:a list of ndarray [array([]), ..., array([])]

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
        #print(i)
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


#0.47548875021634684, 2.4464393446710155          
#dataFeature, dataLabel = readAustralianData('./Data/Australia/australian.dat')
#dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
#print(dataFeature)
#print(dataLabel)