from Params import *
from DataGenerator import *

class Coder:
    def __init__(self, type):
        self.type = type
        self.dataX, self.dataY = DataGenerator().generateData(1, type)

    def encode(self):
        nSamples = DataParams().nCodingData[self.type-1]
        matrixG = np.random.randn(sPrime, nSamples)
        sampleX, sampleY = self.sampleData()
        codedX = np.dot(matrixG, sampleX)
        codedY = np.dot(matrixG, sampleY)
        self.nData = codedX.shape[0]
        return codedX, codedY
    
    def sampleData(self):
        nSamples = DataParams().nCodingData[self.type-1]
        sampleIndices = np.random.choice(self.dataX.shape[0], nSamples)
        sampleX = self.dataX[sampleIndices]
        sampleY = self.dataY[sampleIndices]
        return sampleX, sampleY

    # a, b = Coder(5).encode()
    # print(a.shape, b.shape)