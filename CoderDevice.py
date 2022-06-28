from Params import *
from DataGenerator import *

class Coder:
    def __init__(self, type):
        self.type = type
        self.dataX, self.dataY = DataGenerator().generateData(1, type)

    def encode(self):
        matrixG = np.random.randn(sPrime, nCodingData[self.type-1])
        sampleX, sampleY = self.sampleData()
        codedX = np.dot(matrixG, sampleX)
        codedY = np.dot(matrixG, sampleY)
        self.nData = codedX.shape[0]
        return codedX, codedY
    
    def sampleData(self):
        sampleIndices = np.random.choice(self.dataX.shape[0], nCodingData[self.type-1])
        sampleX = self.dataX[sampleIndices]
        sampleY = self.dataY[sampleIndices]
        return sampleX, sampleY

    # a, b = Coder(5).encode()
    # print(a.shape, b.shape)