from Params import *
from DataGenerator import *

class TrainerDevice:
    def __init__(self, type):
        self.type = type
        self.dataX, self.dataY = DataGenerator().generateData(0, type)
        
    def train(self, model, nSteps):
        modelP = np.copy(model)
        for i in range(nSteps):
            modelP = modelP - (0.01*mu/DataParams().nData) * np.dot(self.dataX.T, np.dot(self.dataX, modelP) - self.dataY)
            diff = np.dot(self.dataX, modelP) - self.dataY
            # print(np.dot(diff.T, diff))
        return modelP

TrainerDevice(8).train(np.zeros((DataParams().dataDimension, 1)), 10000)