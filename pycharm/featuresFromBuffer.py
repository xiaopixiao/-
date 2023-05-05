import hpfilter
import numpy as np
import scipy.signal as sg
import pwelch


def featuresFromBuffer(atx, aty, atz, fs):
    b, a = hpfilter.hpfilter()
    feat=np.zeros((1,66))
    abx = sg.filtfilt(b,a,atx)
    aby = sg.filtfilt(b, a, aty)
    abz = sg.filtfilt(b, a, atz)
    feat[0,0]=np.mean(atx)
    feat[0, 1] = np.mean(aty)
    feat[0, 2] = np.mean(atz)

    feat[0,3]=np.sqrt(np.mean(abx**2))
    feat[0, 4] = np.sqrt(np.mean(aby ** 2))
    feat[0, 5] = np.sqrt(np.mean(abz ** 2))

    feat[0,6:9]=covFeatures(abx, fs)
    feat[0, 9:12] = covFeatures(aby, fs)
    feat[0, 12:15] = covFeatures(abz, fs)

    feat[0,15: 27] = spectralPeaksFeatures(abx, fs)
    feat[0,27: 39] = spectralPeaksFeatures(abx, fs)
    feat[0,39: 51] = spectralPeaksFeatures(abx, fs)

    feat[0,51: 56] = spectralPowerFeatures(abx, fs)
    feat[0,56: 61] = spectralPowerFeatures(aby, fs)
    feat[0,61: 66]= spectralPowerFeatures(abz, fs)


    return feat
def covFeatures(x, fs):
    feats = np.zeros((1, 3))
    feats=np.array(feats).flatten()
    x=np.array(x).flatten()
    c=np.correlate(x, x, mode='full')
    lags = np.arange(-(len(x) - 1), len(x))
    minprom = 0.0005
    mindist_xunits = 0.3
    minpkdist = np.floor(mindist_xunits / (1 / fs))
    locs,_=sg.find_peaks(c, prominence=minprom, distance=minpkdist)
    pks=c[locs]
    tc = (1 / fs) * lags
    tcl = tc[locs]
    if np.size(tcl)!=0:
        feats[0] = pks[int(-1+(len(pks) + 1) / 2)]
    if len(tcl) >= 3:
        feats[1] = tcl[int((len(tcl) + 1) / 2 )]
        feats[2] = pks[int((len(pks) + 1) / 2 )]
    return feats



def spectralPeaksFeatures(x, fs):
    mindist_xunits = 0.3
    feats = np.zeros((1, 12))
    feats = np.array(feats).flatten()
    N = 4096
    minpkdist = np.floor(mindist_xunits / (fs / N))
    x=x.reshape(-1,1)
    f,p=pwelch.pwelch(x,w='boxcar', length=len(x), n=N, fs=fs)
    locs, _ = sg.find_peaks(p,distance=minpkdist)
    locs = locs[np.argpartition(f[locs], 20)[:20]]
    pks=p[locs]
    if np.size(locs)!=0:
        if len(locs)>6:
            mx=6
        else:
            mx=len(locs)
        spks = -np.sort(-pks)
        idx=np.argsort(-pks)

        slocs = locs[idx]

        pks = spks[0:mx]
        locs = slocs[0:mx]

        slocs = np.sort(locs)
        idx=np.argsort(locs)
        spks = pks[idx]
        pks = spks
        locs = slocs
    fpk = f[locs]
    feats[0: int(len(pks))]= fpk
    feats[6: int(7 + len(pks) - 1)] = pks
    return feats




def spectralPowerFeatures(x, fs):
    feats = np.zeros((1, 5))
    feats = np.array(feats).flatten()
    edges = [0.5, 1.5, 5, 10, 15, 20]

    f,p = sg.periodogram(x, window='boxcar', nfft=4096, fs=fs)

    for kband in range(1,len(edges) ):
        feats[kband-1] = np.sum(p[np.where((f >= edges[kband - 1]) & (f < edges[kband]))])
    return feats










