from Params import *
from DataGenerator import *
import copy

class Trainer:
    def __init__(self, type):
        self.type = type
        self.dataX, self.dataY = DataGenerator().generateData(0, type)
        self.nData = self.dataX.shape[0]
        
    def train(self, model):
        # modelP = np.copy(model)
        modelP = copy.deepcopy(model)
        a = self.loss(modelP)
        for i in range(trainerLocalSteps[self.type-1]):
            modelP = modelP - (0.001*mu/DataParams().nData) * np.dot(self.dataX.T, np.dot(self.dataX, modelP) - self.dataY)
        return modelP
    
    def loss(self, model):
        diff = np.dot(self.dataX, model) - self.dataY
        return np.dot(diff.T, diff).squeeze()

# Trainer(8).train(np.zeros((DataParams().dataDimension, 1)))