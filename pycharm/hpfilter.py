import scipy.signal as sg

def hpfilter():
    Fs=50
    Fstop=0.016
    Fpass=0.032
    Astop=60
    Apass=1
    [n,wn]=sg.ellipord(wp=Fpass,ws=Fstop,gpass=Apass,gstop=Astop)
    [b,a]=sg.ellip(n,Apass,Astop,wn,'highpass')
    return b,a