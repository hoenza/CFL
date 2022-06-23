from Params import *
from TrainerDevice import *
from CoderDevice import *

class TaskPublisher:
    def __init__(self, sysType, nDevices):
        self.testDataX, self.testDataY = DataGenerator().generateData(1, N)
        self.sysType = sysType
        self.nDevices = nDevices
        self.federatedDevices = []
        self.coderDevices = []
        self.create_devices()
        self.globalModel = np.zeros((DataParams().dataDimension, 1))
    
    def create_devices(self):
        for i in range(N):
            nn = np.floor(self.nDevices*pn[0, i]*qn[0, i]).astype(int)
            # print(nn)
            federatedDevicesN = []
            for j in range(nn):
                federatedDevicesN.append(Trainer(i+1))
            self.federatedDevices.append(federatedDevicesN)
            nn = np.floor(self.nDevices*pn[0, i]*(1-qn[0, i])).astype(int)
            coderDevicesN = []
            for j in range(nn):
                coderDevicesN.append(Coder(i+1))
            self.coderDevices.append(coderDevicesN)
        # print(self.federatedDevices)
        # print(self.coderDevices)

    def train(self, steps):
        for step in range(steps):
            models0 = []
            for dev in self.federatedDevices:
                modelsI = []
                for devI in dev:
                    modelsI.append(devI.train(self.globalModel))
                models0.append(modelsI)

            models1 = []
            for dev in self.coderDevices:
                modelsI = []
                for devI in dev:
                    eDataX, eDataY = devI.encode()
                    modelsI.append(self.trainLocal(self.globalModel, eDataX, eDataY))
                models1.append(modelsI)
            # print("models0", len(models0))
            # for i in models0:
            #     print(len(i))
            # print("models1", len(models1))
            # for i in models1:
            #     print(len(i))
            self.globalModel = self.joinModels(models0, models1)
            print('loss', self.loss())
    
    def joinModels(self, models0, models1):
        # print('join')
        countData = 0
        modelsSum = np.zeros_like(self.globalModel)
        for t, i in enumerate(models0):
            for tt, j in enumerate(i):
                # print(t, tt, self.federatedDevices[t][tt].nData)
                modelsSum = modelsSum + self.federatedDevices[t][tt].nData * j
                countData = countData + self.federatedDevices[t][tt].nData
        # print('hoi')
        # for t, i in enumerate(models1):
        #     for tt, j in enumerate(i):
        #         # print(t, tt, self.coderDevices[t][tt].nData)
        #         modelsSum = modelsSum + self.coderDevices[t][tt].nData * j
        #         countData = countData + self.coderDevices[t][tt].nData
        print(countData)
        return modelsSum / countData

    def trainLocal(self, model, dataX, dataY):
        return model - (0.01*mu/sPrime) * np.dot(dataX.T, np.dot(dataX, model) - dataY)
    
    def loss(self):
        diff = np.dot(self.testDataX, self.globalModel) - self.testDataY
        return np.dot(diff.T, diff).squeeze()

TaskPublisher(0, 100).train(1000)