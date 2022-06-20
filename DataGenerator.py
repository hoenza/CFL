from Params import *
class DataGenerator:
    def generateData(self, deviceType, dataType):
        X = np.random.randn(DataParams().nData, DataParams().dataDimension)
        y = np.dot(X, DataParams().beta)
        y[y>=0] = 1
        y[y<False] = -1
        print(X.shape, y.shape)
        y = self.noiseData(y, deviceType, dataType)
        print(X.shape, y.shape)
        return X, y
    
    def noiseData(self, y, deviceType, dataType):
        noiseRatio = DataParams().noiseRatio[deviceType, dataType]
        nNoisyData = int(np.floor(noiseRatio * DataParams().nData))
        for i in range(nNoisyData):
            if(y[-i]==1):
                y[-i] = -1
            else:
                y[-i] = 1
        return y

# DataGenerator().generateData(0, 10)