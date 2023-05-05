import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
def pwelch(x,length=None,w=None,no=None,n=None,fs=50):
    f,p=sg.welch(x[:,0],fs,nperseg=length,nfft=n,noverlap=no,window=w)
    return f,p
