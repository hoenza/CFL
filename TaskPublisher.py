from Params import *
from TrainerDevice import *
from CoderDevice import *

class TaskPublisher:
    def __init__(self, sysType, nDevices):
        self.sysType = sysType
        self.nDevices = nDevices
        self.federatedDevices = []
        self.coderDevices = []
        self.create_devices()
        self.globalModel = np.copy(DataParams().beta)
    
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
        print(self.federatedDevices)
        print(self.coderDevices)

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
            print("models0", len(models0))
            print("models1", len(models1))
    
    def trainLocal(self, model, dataX, dataY):
        return model - (0.01*mu/sPrime) * np.dot(dataX.T, np.dot(dataX, model) - dataY)

TaskPublisher(0, 100).train(1)