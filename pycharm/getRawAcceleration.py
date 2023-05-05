import scipy.io as scio
import numpy as np
def getRawAcceleration(AccelerationType, SubjectID, Component, fs):
    a=scio.loadmat('./data/RecordedAccelerationsBySubject.mat')
    subjects=a['subjects']
    subid=SubjectID
    if AccelerationType =='total':
        type='totalacc'
    elif AccelerationType=='body':
        type='bodyacc'
    if Component=='x':
        component=1
    elif Component=='y':
        component=2
    elif Component=='z':
        component=3
    acc = subjects[0, subid - 1][type][:, component-1]
    acc = acc.reshape(-1, 1)
    actid = subjects[0, subid - 1]['actid']
    t = np.arange(0, len(acc), 1) * (1 / fs)
    t = t.reshape(-1, 1)
    return acc,actid,t



