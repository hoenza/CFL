import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

def plot_epsilons_thetas(epsilons, thetas):
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)
    xAxis = list(range(1, N+1))
    axs[0].plot(xAxis, epsilons)
    axs[0].set_title('Epsilons')
    axs[0].set_xlabel('#Type')
    axs[0].set_ylabel('epsilon')
    axs[1].plot(xAxis, thetas)
    axs[1].set_title('Thetas')
    axs[1].set_xlabel('#Type')
    axs[1].set_ylabel('theta')

    plt.savefig('plot_epsilons_thetas.jpg')
    plt.savefig('plot_epsilons_thetas.eps')
    plt.close()

def plot_rewards_frequencies(rn, rpn, fn, fpn):
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)
    xAxis = list(range(1, N+1))

    axs[0].plot(xAxis, rn, label='Federated')
    axs[0].plot(xAxis, rpn, label='Coded')
    axs[0].set_title('Rewards')
    axs[0].set_xlabel('#Type')
    axs[0].set_ylabel('reward')
    axs[0].legend()
    axs[1].plot(xAxis, fn, label='Federated')
    axs[1].plot(xAxis, fpn, label='Coded')
    axs[1].set_title('Frequencies')
    axs[1].set_xlabel('#Type')
    axs[1].set_ylabel('#Frequency')
    axs[1].legend()

    plt.savefig('plot_rewards_frequencies.jpg')
    plt.savefig('plot_rewards_frequencies.eps')
    plt.close()

def plot_rsus(rsu, rsuP):
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 9)
    xAxis = list(range(1, N+1))

    maxRsus = []
    for j, i in enumerate(rsu):
        axs[0].plot(xAxis, i, label=str(j+1))
        maxRsus.append(i[j])
    axs[0].plot(xAxis, maxRsus, color='m', ls='', marker='o')
    axs[0].set_title('Federated Devices Profits')
    axs[0].set_xlabel('#Type')
    axs[0].set_ylabel('profit')
    axs[0].legend()

    maxRsuPs = []
    for j, i in enumerate(rsuP):
        axs[1].plot(xAxis, i, label=str(j+1))
        maxRsuPs.append(i[j])
    axs[1].plot(xAxis, maxRsuPs, color='m', ls='', marker='o')
    axs[1].set_title('Coded Devices Profits')
    axs[1].set_xlabel('#Type')
    axs[1].set_ylabel('profit')
    axs[1].legend()

    plt.savefig('plot_rsus.jpg')
    plt.savefig('plot_rsus.eps')
    plt.close()
#Constants
N = 10
pn = np.ones((1, N))/N
qn = np.ones((1, N))/2
omega = 800
omegaPrime = omega
tMax = 600
rMax = 10000
tComN = 10
eComN = 20
psi = 150
# epsilons = np.expand_dims(np.arange(start=0.1, stop=0.85, step=(0.85-0.1)/N), axis=0)
# thetas = psi/np.log(1/epsilons)
thetas = np.expand_dims(np.arange(200, 920, (920-200)/N), axis=0)
epsilons =  np.exp(-psi/thetas)
plot_epsilons_thetas(epsilons[0], thetas[0])
cn = 5
sn = 20
capacitance = 2
mu = 1
l = 1
gamma = 15
eComNp = eComN
Cm = 10
d = 2
tComNP = tComN
tMax = 600
rMax = 10000
omega = 80000
omegaPrime = omega
c = cn
s = sn
sPrime = s


gn = np.zeros((1, N))
for i in range(N):
    gn[0][i] = psi * pn[0][i] / thetas[0][i]
    for j in range(i+1, N):
        gn[0][i] = gn[0][i] + (psi/thetas[0][i] - psi/thetas[0][i+1]) * pn[0][j]

fn = cp.Variable((1, N))
siTmp = psi/thetas
tCmpN = cn * sn * cp.inv_pos(fn)
lnExpr = cp.log(tMax - tComN - cp.multiply(siTmp, tCmpN))
uTP = N * omega * cp.sum(cp.multiply(pn, lnExpr)) - N * l * eComN - N * l * capacitance * cp.sum(c * s * cp.multiply(gn, cp.square(fn)))
eCmpN = capacitance * sn * cp.square(fn)
constraints = [cn * sn / tMax <= fn,
               N * eComN + N * capacitance * cn * sn * (gn.T * cp.square(fn)) <= rMax]
            #    N * 20 + N * capacitance * cp.sum(c * s * cp.multiply(gn, cp.square(fn))) <= rMax

hn = np.zeros((1, N))
for i in range(N):
    hn[0][i] = (gamma**2 * pn[0][i] * (1-qn[0][i])) / (thetas[0][i]**2)
    for j in range(i+1, N):
        hn[0][i] = hn[0][i] + ((gamma/thetas[0][i])**2 - (gamma/thetas[0][i+1])**2) * pn[0][j] * (1-qn[0][j])

fpn = cp.Variable((1, N))
gammaTmp = cp.square(gamma/thetas)
tCmpNp = Cm * (d+1) * cp.inv_pos(fpn)
lnExprC = cp.log(tMax - tComNP - cp.multiply(gammaTmp, tCmpNp))
uTP = N * omega * cp.sum(cp.multiply(cp.multiply(pn, qn), lnExpr)) - N * l * eComN - N * l * capacitance * cp.sum(c * s * cp.multiply(gn, cp.square(fn)))
uTP = uTP + N * omega * cp.sum(cp.multiply(cp.multiply(pn, 1-qn), lnExprC)) - N * l * eComNp - N * l * capacitance * cp.sum(cp.multiply(hn, Cm * (d+1) * cp.square(fpn)))
constraints = [c * s / tMax <= fn,
               N * eComN + N * capacitance * c * s * (gn.T * cp.square(fn)) + N * eComNp + N * capacitance * cp.sum(cp.multiply(hn, Cm * (d+1) * cp.square(fpn))) <= rMax,
               fpn <= fn/2,
               fn[0, :-1] <= fn[0, 1:],
               fpn[0, :-1] <= fpn[0, 1:]]


prob = cp.Problem(cp.Maximize(uTP), constraints)
prob.solve()
print("The optimal value is", prob.value)
print("A solution X is")
print(fn.value)
print(fpn.value)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

eCmpN = np.squeeze(capacitance * c * s * fn.value**2)
rnTmp = []
rnTmp.append(eComN + (psi/thetas[0, 0]) * eCmpN[0])
for i in range(1, N):
    rnTmp.append(rnTmp[-1] + (psi/thetas[0, i]) * (eCmpN[i] - eCmpN[i-1]))
rnTmp = np.array(rnTmp)
# plt.plot(rnTmp)
rsu = []
for i in range(N):
    a = rnTmp - eComN - (psi/thetas[0, i]) * eCmpN
    rsu.append(a.tolist())
rsu = np.array(rsu)

epCmpN = np.squeeze(capacitance * 2 *sPrime * d * Cm * fpn.value**2)
rnPTmp = []
rnPTmp.append(eComNp + (gamma/thetas[0, 0]) * epCmpN[0])
for i in range(1, N):
    rnPTmp.append(rnPTmp[-1] + (gamma/thetas[0, i]) * (epCmpN[i] - epCmpN[i-1]))
rnPTmp = np.array(rnPTmp)
# plt.plot(rnPTmp)
rsuP = []
for i in range(N):
    a = rnPTmp - eComNp - (gamma/thetas[0, i]) * epCmpN
    rsuP.append(a.tolist())
rsuP = np.array(rsuP)

plot_rewards_frequencies(rnTmp, rnPTmp, fn.value[0], fpn.value[0])
plot_rsus(rsu, rsuP)