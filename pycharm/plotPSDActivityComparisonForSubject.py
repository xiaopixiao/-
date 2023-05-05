import hpfilter
import scipy.signal as sg
import scipy.io as scio
import numpy as np
import skimage.measure as sm
import pwelch
import matplotlib.pyplot as plt
import getRawAcceleration
import  addActivityLegend
def plotPSDActivityComparisonForSubject(subject, act1name, act2name):
    fs = 50
    acc,actid,t=getRawAcceleration.getRawAcceleration('total',subject,'x',fs)
    b, a = hpfilter.hpfilter()
    ab = sg.filtfilt(b, a, acc, axis=0)
    s = scio.loadmat('./data/BufferedAccelerations.mat')['actnames']
    id = [np.where(s == act1name)[1] + 1, np.where(s == act2name)[1] + 1]
    cmap = ['b', 'g', 'r', 'c', 'm', 'y']
    for k in range(0, len(id)):
        sel = np.where(actid == id[k][0], 1, 0)
        reglabs = sm.label(sel)
        sel=np.where(reglabs==1)
        d=ab[sel[0],:]
        [fp,psd] = pwelch.pwelch(d,fs=fs,w='boxcar',n=513)
        plt.plot(fp,20*np.log10(psd),cmap[id[k][0]],linewidth = 1.5)
    plt.xlim(0, 10)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/frequency (dB/Hz)')
    plt.title('Power Spectral Density Comparison')
    addActivityLegend.addActivityLegend(np.array([id[0][0],id[1][0]]))
    plt.show()


