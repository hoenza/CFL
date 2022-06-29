import numpy as np
import matplotlib.pyplot as plt
np.random.seed(8)

N = 10
pn = np.ones((1, N))/N
qn = np.ones((1, N))/2
psi = 15
gamma = 1000

epsilons = np.arange(start=0.2, stop=0.92, step=(0.92 - 0.2)/N)
# print('epsilons', epsilons)
thetas = psi/np.log(1/epsilons)
# print('thetas', thetas)
trainerLocalSteps = (10 * np.log(1/epsilons)).astype(int)
# print('lStes', trainerLocalSteps.shape,  trainerLocalSteps)
numOfCodingDatas = (gamma/thetas).astype(int)
# print('nCodingDatas', numOfCodingDatas)

plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(2, 2)
xAxis = list(range(1, N+1))
axs[0, 0].plot(xAxis, epsilons)
axs[0, 0].set_title('epsilons')
axs[0, 1].plot(xAxis, thetas)
axs[0, 1].set_title('thetas')
axs[1, 0].plot(xAxis, trainerLocalSteps)
axs[1, 0].set_title('num of local steps')
axs[1, 1].plot(xAxis, numOfCodingDatas)
axs[1, 1].set_title('num of coding data')
plt.savefig('plot1.jpg')
plt.savefig('plot1.eps')
plt.close()

sn = 20
mu = 1
sPrime = 20

nEdgeDeviceData = 1000
modelSize = 500
targetBeta = np.random.randn(modelSize, 1)
beta = np.random.randn(modelSize, 1)

federatedNoiseRatio = np.expand_dims(np.arange(start=0.3, stop=0, step=-0.3/N), axis=0)
codersNoiseRatio = np.expand_dims(np.arange(start=0.1, stop=0, step=-0.1/N), axis=0)
noiseRatio = np.concatenate((federatedNoiseRatio, codersNoiseRatio), axis=0)

nCodingData = np.squeeze((gamma/thetas).astype(int))