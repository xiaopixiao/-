import getRawAcceleration
import hpfilter
import scipy.signal as sg
import scipy.io as scio
import skimage.measure as sm
import numpy as np
import matplotlib.pyplot as plt
import addActivityLegend
def plotCorrActivityComparisonForSubject(subject, act1name, act2name):
    fs=50
    acc,actid,t=getRawAcceleration.getRawAcceleration(AccelerationType='total',SubjectID=subject,Component='x',fs=fs)
    b, a = hpfilter.hpfilter()
    ab = sg.filtfilt(b, a, acc, axis=0)
    s = scio.loadmat('./data/BufferedAccelerations.mat')['actnames']
    id = [np.where(s == act1name)[1] + 1, np.where(s == act2name)[1] + 1]
    cmap= ['b', 'g', 'r', 'c', 'm', 'y']
    for k in range(0,len(id)):
        sel = np.where(actid == id[k][0], 1, 0)
        reglabs = sm.label(sel)
        sel = np.where(reglabs == 1)
        d = ab[sel[0], :]
        d=np.array(d).flatten()
        c=np.correlate(d,d,mode='full')
        lags = np.arange(-(len(d) - 1), len(d))
        tc = (1 / fs) * lags
        plt.plot(tc, c, color=cmap[id[k][0]],linewidth = 1.5)
    plt.grid()
    plt.xlim(-5,5)
    plt.title('Autocorrrelation Comparison')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Correlation')
    addActivityLegend.addActivityLegend(np.array([id[0][0],id[1][0]]))
    plt.show()




