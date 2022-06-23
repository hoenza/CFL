from Params import *
from DataGenerator import *

class Trainer:
    def __init__(self, type):
        self.type = type
        self.dataX, self.dataY = DataGenerator().generateData(0, type)
        
    def train(self, model):
        modelP = np.copy(model)
        for i in range(trainerLocalSteps[self.type-1]):
            modelP = modelP - (0.01*mu/DataParams().nData) * np.dot(self.dataX.T, np.dot(self.dataX, modelP) - self.dataY)
            diff = np.dot(self.dataX, modelP) - self.dataY
            # print(np.dot(diff.T, diff))
        return modelP

Trainer(8).train(np.zeros((DataParams().dataDimension, 1)))