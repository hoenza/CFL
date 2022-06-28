from Params import *
from DataGenerator import *
import copy

class Trainer:
    def __init__(self, type):
        self.type = type
        self.dataX, self.dataY = DataGenerator().generateData(0, type)
        self.nData = self.dataX.shape[0]
        
    def train(self, model):
        modelP = np.copy(model)
        a = self.loss(modelP)
        for i in range(trainerLocalSteps[self.type-1]):
            sampleX, sampleY = self.sampleData()
            modelP = modelP - (0.001*mu/nEdgeDeviceData) * np.dot(sampleX.T, np.dot(sampleX, modelP) - sampleY)
        return modelP
    
    def sampleData(self):
        sampleIndices = np.random.choice(self.dataX.shape[0], sn)
        sampleX = self.dataX[sampleIndices]
        sampleY = self.dataY[sampleIndices]
        return sampleX, sampleY

    def loss(self, model):
        diff = np.dot(self.dataX, model) - self.dataY
        return np.dot(diff.T, diff).squeeze()
    

# Trainer(8).train(np.zeros((modelSize, 1)))