
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
def addActivityLegend(mode):
    if len(mode)==0:
        return
    s = scio.loadmat('./data/BufferedAccelerations.mat')['actnames']
    u=[0]
    for i in range(0,6):
        u.append(s[0][i][0])
    u=np.array(u)
    if mode=='all':
        plt.legend(u[1:-1])
    else :
        u=u[mode]
        plt.legend(u)


