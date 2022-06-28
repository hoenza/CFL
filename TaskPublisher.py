from Params import *
from TrainerDevice import *
from CoderDevice import *


# sysType = 0 : Federated Mode Only
# sysType = 1 : Coded Mode Only
# sysType = 2 : Federated & Coded Joint model
class TaskPublisher:
    def __init__(self, sysType, nDevices):
        self.testDataX, self.testDataY = DataGenerator().generateData(-1, N)
        self.sysType = sysType
        self.nDevices = nDevices
        self.federatedDevices = []
        self.coderDevices = []
        self.create_devices()
        self.globalModel = np.zeros((modelSize, 1))
    
    def create_devices(self):
        for i in range(N):
            nn = np.floor(self.nDevices*pn[0, i]*qn[0, i]).astype(int)
            federatedDevicesN = []
            for j in range(nn):
                federatedDevicesN.append(Trainer(i+1))
            self.federatedDevices.append(federatedDevicesN)
            if self.sysType:
                nn = np.floor(self.nDevices*pn[0, i]*(1-qn[0, i])).astype(int)
                coderDevicesN = []
                for j in range(nn):
                    coderDevicesN.append(Coder(i+1))
                self.coderDevices.append(coderDevicesN)

    def train(self, steps):
        report = {'losses':[], 'accs':[]}
        for step in range(steps):
            models0 = []
            for dev in self.federatedDevices:
                modelsI = []
                for devI in dev:
                    modelsI.append(devI.train(self.globalModel))
                models0.append(modelsI)

            models1 = []
            if self.sysType:
                for dev in self.coderDevices:
                    modelsI = []
                    for devI in dev:
                        eDataX, eDataY = devI.encode()
                        modelsI.append(self.trainLocal(self.globalModel, eDataX, eDataY))
                    models1.append(modelsI)
            
            self.globalModel = self.joinModels(models0, models1)
            print('step:', step, 'loss:', self.loss(), 'acc:', self.accuracy())
            report['losses'].append(self.loss())
            report['accs'].append(self.accuracy())
        return report
    
    def joinModels(self, models0, models1):
        countData = 0
        modelsSum = np.zeros_like(self.globalModel)
        
        if self.sysType != 1:
            for t, i in enumerate(models0):
                for tt, j in enumerate(i):
                    modelsSum = modelsSum + self.federatedDevices[t][tt].nData * j
                    countData = countData + self.federatedDevices[t][tt].nData
        
        if self.sysType != 0:
            for t, i in enumerate(models1):
                for tt, j in enumerate(i):
                    modelsSum = modelsSum + self.coderDevices[t][tt].nData * j
                    countData = countData + self.coderDevices[t][tt].nData
        
        return modelsSum / countData

    def trainLocal(self, model, dataX, dataY):
        modelP = np.copy(model)
        for i in range(1):
            modelP = modelP - (0.001*mu/sPrime) * np.dot(dataX.T, np.dot(dataX, modelP) - dataY)
        return modelP
    
    def loss(self):
        diff = np.dot(self.testDataX, self.globalModel) - self.testDataY
        return np.log(np.dot(diff.T, diff).squeeze())
    
    def accuracy(self):
        labels = np.dot(self.testDataX, self.globalModel)
        labels[labels>=0] = 1
        labels[labels<0] = -1
        return np.sum(labels==self.testDataY)/self.testDataX.shape[0]

globalTrainingSteps = 100
report0 = TaskPublisher(0, 100).train(globalTrainingSteps)
report1 = TaskPublisher(1, 100).train(globalTrainingSteps)
report2 = TaskPublisher(2, 100).train(globalTrainingSteps)


plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(2, 1)
xAxis = list(range(1, globalTrainingSteps+1))

axs[0].plot(xAxis, report0['losses'], label='Federated')
axs[0].plot(xAxis, report1['losses'], label='Coded')
axs[0].plot(xAxis, report2['losses'], label='Joint')
axs[0].set_title('loss')
axs[0].legend()

axs[1].plot(xAxis, report0['accs'], label='Federated')
axs[1].plot(xAxis, report1['accs'], label='Coded')
axs[1].plot(xAxis, report2['accs'], label='Joint')
axs[1].set_title('accuracy')
axs[1].legend()

plt.savefig('plot2.jpg')
plt.savefig('plot2.eps')
