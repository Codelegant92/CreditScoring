from commonFunction import *
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np
from pybrain.structure import SoftmaxLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules import TanhLayer

'function: artificial neural network'
def ANN(trainFeature, trainLabel, testFeature, testLabel, netStructure, para_rate, para_momentum): #netStructure is a list [in, hidden, out], momentum is a parameter in SGD
    sampleNum = trainFeature.shape[0]
    featureNum = trainFeature.shape[1]
    Dataset = SupervisedDataSet(featureNum, 1)
    i = 0
    while(i < sampleNum):
        print(i)
        Dataset.addSample(list(trainFeature[i]), [trainLabel[i]])
        i += 1
    Network = buildNetwork(netStructure[0], netStructure[1], netStructure[2], netStructure[3], hiddenclass = SigmoidLayer, outclass=SigmoidLayer) 
    T = BackpropTrainer(Network, Dataset, learningrate = para_rate, momentum = para_momentum, verbose = True)
    #print(Dataset['input'])
    errorList = []
    errorList.append(T.testOnData(Dataset))
    T.trainOnDataset(Dataset)
    errorList.append(T.testOnData(Dataset))
    T.trainOnDataset(Dataset)
    while(abs(T.testOnData(Dataset)-errorList[-1]) > 0.0001):
        T.trainOnDataset(Dataset)
        errorList.append(T.testOnData(Dataset))
    pass #this step is for the output of predictedLabel
    print(np.array([Network.activate(x) for x in trainFeature])) 
    #print(testLabel)
    print(Network.activate([0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    return(errorList) 


dataFeature, dataLabel = read_GermanData('./Data/german/german.data-numeric')
trainFeature = dataFeature[:800, :]
print(trainFeature.shape)
trainLabel = dataLabel[:800]
testFeature = dataFeature[800:, :]
testLabel = dataLabel[800:]
errorLIST = ANN(trainFeature, trainLabel, testFeature, testLabel, [24, 12, 6, 1], 0.15, 0.5)  
print(errorLIST) 