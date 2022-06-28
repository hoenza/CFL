from Params import *
class DataGenerator:
    def generateData(self, deviceType, dataType):
        X = np.random.randn(nEdgeDeviceData, modelSize)
        y = np.dot(X, targetBeta)
        y[y>=0] = 1
        y[y<False] = -1
        # print(X.shape, y.shape)
        y = self.noiseData(y, deviceType, dataType-1)
        # print(X.shape, y.shape)
        return X, y
    
    def noiseData(self, y, deviceType, dataType):
        noiseRatioN = noiseRatio[deviceType, dataType]
        nNoisyData = int(np.floor(noiseRatioN * nEdgeDeviceData))
        for i in range(nNoisyData):
            if(y[-i]==1):
                y[-i] = -1
            else:
                y[-i] = 1
        return y

# DataGenerator().generateData(0, 10)