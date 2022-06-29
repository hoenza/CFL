from Params import *
from TrainerDevice import *
from CoderDevice import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


# sysType = 0 : Federated Mode Only
# sysType = 1 : Coded Mode Only
# sysType = 2 : Federated & Coded Joint model
class TaskPublisher:
    def __init__(self, sysType, nDevices, data=None):
        if data is None:
            self.testDataX, self.testDataY = DataGenerator().generateData(-1, N)
        else:
            self.testDataX = data[0]
            self.testDataY = data[1]
        
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
            report['losses'].append(self.loss())
            report['accs'].append(self.accuracy())
        
        print('loss:', self.loss(), 'acc:', self.accuracy())
        self.table()
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
    
    def label(self):
        labels = np.dot(self.testDataX, self.globalModel)
        labels[labels>=0] = 1
        labels[labels<0] = -1
        self.labels = labels
        return labels

    def accuracy(self):
        labels = self.label()
        return np.sum(labels==self.testDataY)/self.testDataX.shape[0]
    
    def table(self):
        pred_labels = self.label()
        truth_labels = self.testDataY
        conf_mat = confusion_matrix(truth_labels, pred_labels)
        return conf_mat

    def plot_table(self, curAx):
        conf_mat = self.table()
        pdframe = pd.DataFrame(conf_mat, range(2), range(2))
        sns.heatmap(pdframe, annot=True, ax=curAx, fmt='g')


globalTrainingSteps = 100
taskPublisher0 = TaskPublisher(0, 100)
data = [taskPublisher0.testDataX, taskPublisher0.testDataY]
report0 = taskPublisher0.train(globalTrainingSteps)
taskPublisher1 = TaskPublisher(1, 100, data)
report1 = taskPublisher1.train(globalTrainingSteps)
taskPublisher2 = TaskPublisher(2, 100, data)
report2 = taskPublisher2.train(globalTrainingSteps)


plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(2, 1)
xAxis = list(range(1, globalTrainingSteps+1))

axs[0].plot(xAxis, report0['losses'], label='Federated')
axs[0].plot(xAxis, report1['losses'], label='Coded')
axs[0].plot(xAxis, report2['losses'], label='Joint')
axs[0].set_title('log(loss)')
axs[0].legend()

axs[1].plot(xAxis, report0['accs'], label='Federated')
axs[1].plot(xAxis, report1['accs'], label='Coded')
axs[1].plot(xAxis, report2['accs'], label='Joint')
axs[1].set_title('accuracy')
axs[1].legend()

plt.savefig('plot2.jpg')
plt.savefig('plot2.eps')
plt.close()

plt.title('Confusion Matrix')
plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(1, 3)
fig.set_size_inches(14, 4)

taskPublisher0.plot_table(axs[0])
axs[0].set_title('Federated')
taskPublisher1.plot_table(axs[1])
axs[1].set_title('Coded')
taskPublisher2.plot_table(axs[2])
axs[2].set_title('Joint')


plt.savefig('plot3.jpg')
plt.savefig('plot3.eps')
plt.close()
