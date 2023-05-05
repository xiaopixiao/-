import hpfilter
import getRawAcceleration
import numpy as np
import scipy.signal as sg
import pwelch
import skimage.measure as sm
import matplotlib.pyplot as plt
import scipy.io as scio
import addActivityLegend
def plotPSDForGivenActivity(activityid):
    cmap = ['b', 'g', 'r', 'c', 'm', 'y']
    s = scio.loadmat('./data/BufferedAccelerations.mat')['actnames']
    b, a = hpfilter.hpfilter()
    ax1 = plt.axes(projection='3d')
    for subject in range(1,31):
        fs = 50
        acc, actid,t=getRawAcceleration.getRawAcceleration(SubjectID=subject,Component='x',AccelerationType='total',fs=fs)

        ab = sg.filtfilt(b, a, acc, axis=0)
        sel = np.where(actid == activityid,1,0)
        reglabs=sm.label(sel)
        sel=np.where(reglabs==1)
        d=ab[sel[0],:]
        fp, psd = pwelch.pwelch(d,fs=fs,w='boxcar',n=513)
        ax1.plot3D(subject*np.ones(len(fp), dtype=int),fp,psd,color=cmap[activityid-1],linewidth = 1.5)
    plt.grid()
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Subject ID')
    addActivityLegend.addActivityLegend(np.array([activityid]))
    plt.show()
