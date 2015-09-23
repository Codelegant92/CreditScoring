__author__ = 'robin'
from PyQt4 import QtCore, QtGui
from creditScoring_UI import Ui_Dialog, _translate
from commonFunction import *
from decisionTree import decision_Tree

class Window(QtGui.QDialog):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.fileButton.clicked.connect(self.openFileDialog)
        self.ui.pushButton.clicked.connect(self.classifier)

    def openFileDialog(self):
        filename = QtGui.QFileDialog.getOpenFileNames(self, "Open File", "/home/robin/Documents")
        print(filename[0])
        for i in range(len(filename[0])):
            if(filename[0][-i-1] == '/'):
                break
        self.ui.filePathBrowser.setText('.../' + filename[0][-i:])
        self.ui.fileButton.setText(_translate("Dialog", filename[0], None))

    def classifier(self):
        testFeature = np.zeros(20)
        testFeature[0] = int(self.ui.comboBox_10.currentIndex())
        testFeature[1] = int(self.ui.spinBox.text())
        testFeature[2] = int(self.ui.comboBox_12.currentIndex())
        testFeature[3] = int(self.ui.comboBox_9.currentIndex())
        testFeature[4] = int(self.ui.spinBox_2.text())
        testFeature[5] = int(self.ui.comboBox_11.currentIndex())
        testFeature[6] = int(self.ui.comboBox_5.currentIndex())
        testFeature[7] = float(self.ui.doubleSpinBox.text())
        testFeature[8] = int(self.ui.comboBox_2.currentIndex())
        testFeature[9] = int(self.ui.comboBox_13.currentIndex())
        testFeature[10] = int(self.ui.spinBox_3.text())
        testFeature[11] = int(self.ui.comboBox_8.currentIndex())
        testFeature[12] = int(self.ui.spinBox_4.text())
        testFeature[13] = int(self.ui.comboBox_14.currentIndex())
        testFeature[14] = int(self.ui.comboBox_7.currentIndex())
        testFeature[15] = int(self.ui.spinBox_5.text())
        testFeature[16] = int(self.ui.comboBox_4.currentIndex())
        testFeature[17] = int(self.ui.spinBox_6.text())
        testFeature[18] = int(self.ui.comboBox_3.currentIndex())
        testFeature[19] = int(self.ui.comboBox_6.currentIndex())
        filePath = self.ui.fileButton.text()
        trainFeature, trainLabel = read_GermanData20(filePath)
        predictedLabel = decision_Tree(trainFeature, trainLabel, testFeature)
        self.ui.textBrowser.setText(str(predictedLabel))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

