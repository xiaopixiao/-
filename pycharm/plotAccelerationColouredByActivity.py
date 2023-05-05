import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import copy
import addActivityLegend
def plotAccelerationColouredByActivity(t, acc, actid, varargin):
    s=scio.loadmat('./data/BufferedAccelerations.mat')['actnames']
    acts = np.unique(actid)
    nacts = np.max(len(acts))
    if nacts<1:
        nacts=1
    nplots = np.size(acc,1)
    cmap=['b', 'g', 'r', 'c', 'm', 'y']
    for kp in range(nplots):
        plt.subplot(nplots,1,kp+1)
        for ka in range(nacts):
            aid,tsel,asel=getDataForActivityId(ka,acts,actid,acc,kp,t)
            plt.plot(tsel,asel,cmap[aid-1],linewidth=1.5)
        plt.xlabel('time (s)')
        plt.ylabel('a_z (m/s^2)')
        plt.xlim([t[0], t[-1]])
        if len(varargin)>=1:
            plt.title(varargin[kp])
        plt.grid()

    addActivityLegend.addActivityLegend(acts)
    plt.show()

def getDataForActivityId(ka,acts,actid,acc,kp,t):
    aid = 1
    try:
        aid=acts[ka]
    except:
        print('wrong')
    seel=np.where(actid!=aid)
    asel = copy.deepcopy(acc[:,kp])
    asel[seel[0]]=np.nan
    tsel=copy.deepcopy(t)
    tsel[seel]=np.nan
    return aid,tsel,asel


