import numpy as np

#Constants
N = 10
np.random.seed(8)

pn = np.ones((1, N))/N
qn = np.ones((1, N))/2
omega = 800
omegaPrime = omega
tMax = 600
rMax = 10000
tComN = 10
eComN = 20
psi = 15
# epsilons = np.expand_dims(np.exp(np.arange(start=np.log(0.2), stop=np.log(0.92), step=(np.log(0.92)-np.log(0.2))/(N))), axis=0)
# epsilons = np.expand_dims(np.arange(start=0.2, stop=0.92, step=0.036), axis=0)
# thetas = psi/np.log(1/epsilons)
# plt.plot(thetas[0])
thetas = np.expand_dims(np.arange(0.6, 1, 0.02), axis=0) * 1000
epsilons =  np.exp(-psi/thetas)
trainerLocalSteps = np.squeeze((np.log(1/epsilons)*1000).astype(int))
print('lStes', trainerLocalSteps.shape,  trainerLocalSteps)
# plt.plot(epsilons[0])
# print(epsilons.shape, epsilons[0])
# plt.plot(thetas[0])
# print(thetas, thetas.shape)
cn = 5
sn = 20
capacitance = 2
mu =    1
l = 1
sPrime = 30
gamma = 100000
dataDimension = 500
targetBeta = np.random.randn(dataDimension, 1)

class DataParams:
    def __init__(self):
        self.nData = 1000
        self.dataDimension = 500
        self.beta = np.random.randn(self.dataDimension, 1)

        federatedNoiseRatio = np.expand_dims(np.arange(start=0.2, stop=0, step=-0.2/N), axis=0)
        codersNoiseRatio = np.expand_dims(np.arange(start=0.1, stop=0, step=-0.1/N), axis=0)
        self.noiseRatio = np.concatenate((federatedNoiseRatio, codersNoiseRatio), axis=0)

        self.nCodingData = np.squeeze((gamma/thetas).astype(int))
        # print(self.nCodingData)

DataParams()
# print(gamma/thetas)