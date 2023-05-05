import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import addActivityLegend

def plotCompareHistForActivities(acc, actid, actname1, actname2):
    plt.subplot(211)
    plotHistogram(acc, actid, actname1)
    plt.subplot(212)
    plotHistogram(acc, actid, actname2)
    plt.show()

def plotHistogram(data, id, actstr):
    s = scio.loadmat('./data/BufferedAccelerations.mat')['actnames']
    actlabels = s
    actid = np.where(actlabels == actstr)[1]
    sel = np.where(id == actid + 1)
    datasel = data[sel]
    d = np.arange(0, 20.5, 0.5)
    cmap = ['b', 'g', 'r', 'c', 'm', 'y']
    col = cmap[actid[0]]
    plt.hist(datasel, bins=d, edgecolor='black', color=col)
    plt.xlabel('Acceleration Values (m /s^2)')
    plt.ylabel('Occurencies')
    plt.xlim(np.min(datasel), np.max(datasel))
    addActivityLegend.addActivityLegend(np.array(actid+1))


