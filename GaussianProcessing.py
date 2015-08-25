#Gaussian Processing for classification
from commonFunction import *
import numpy as np
from sklearn import gaussian_process

def gaussianProcessing(trainFeature, trainLabel, testFeature, testLabel):
	gp = gaussian_process.GaussianProcess(theta0 = 1e-2, thetaL = 1e-4, thetaU = 1e-1)
	gp.fit(trainFeature, trainLabel)
	predictLabel, sigma2_pred = gp.predict(testFeature, eval_MSE = True)
	return(list(predictLabel*testLabel > 0).count(True)*1.0/len(testLabel))

def crossValidation_gp(dataFeature, dataLabel, folderNum):
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
		cross_accuracy.append(gaussianProcessing(trainFeature, trainLabel, testFeature, testLabel))
	return(cross_accuracy)

    

folderNum = 5
dataFeature, dataLabel = read_data('./Data/german/german.data-numeric')
#gaussianProcessing(dataFeature[:500], np.array(dataLabel[:500]), dataFeature[500:], np.array(dataLabel[500:]))

if(__name__ == "__main__"):
	accu = crossValidation_gp(dataFeature, dataLabel, folderNum)
	print(np.array(accu).mean(0))
