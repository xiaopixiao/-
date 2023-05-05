
import plotCompareHistForActivities
import plotAccelerationColouredByActivity
import hpfilter
import pwelch
import plotPSDActivityComparisonForSubject
import getRawAcceleration
import plotCorrActivityComparisonForSubject
import featuresFromBuffer
import addActivityLegend
import plotPSDForGivenActivity
import plotAccelerationBufferAndPrediction
import myplotAccelerationBufferAndPrediction

import scipy.signal as sg
import scipy.io as scio
import numpy as np
import math

import matplotlib.pyplot as plt
import os
import neurolab as nl
import sklearn.model_selection as sd
import pretty_confusion_matrix.pretty_confusion_matrix as pp

# 读取数据
acc,actid,t=getRawAcceleration.getRawAcceleration('total',1,'x',50)
# 绘制不同状态加速度对比图
plotCompareHistForActivities.plotCompareHistForActivities(acc, actid,'Walking', 'WalkingUpstairs')

# 定义滤波器
b,a=hpfilter.hpfilter()
ab=sg.filtfilt(b,a,acc,axis=0)

# 绘制加速度对比图，以不同颜色区分数据
plotAccelerationColouredByActivity.plotAccelerationColouredByActivity(t,np.hstack((acc,ab)),actid,['Original','High-pass filtered'])
# # #以下为作业6

# 绘制一个前250个t数据的且是walking状态的加速度数据
walking=1
a=np.where(actid==walking,1,0)
b=np.where(t<250,1,0)
sel=np.multiply(a,b)
tw=t[np.flatnonzero(sel)]
abw=ab[np.flatnonzero(sel)]
plotAccelerationColouredByActivity.plotAccelerationColouredByActivity(tw,abw,[],['walking'])

# 绘制功率谱密度
f,p=pwelch.pwelch(abw,fs=50,w='boxcar',length=1024)
plt.plot(f,20*np.log10(p))
plt.title('Welch Power Spectral Density Estimate')
plt.ylabel('Power/frequency (dB/Hz)')
plt.xlabel('Frequency (Hz)')
plt.show()

# 绘制不同状态的功率谱图
plotPSDActivityComparisonForSubject.plotPSDActivityComparisonForSubject(1,'Walking','WalkingUpstairs')

# 绘制三维功率谱图，x坐标为subjectid，y坐标为频谱，z坐标为功率值
plotPSDForGivenActivity.plotPSDForGivenActivity(walking)
# 以下为作业7

# 标注功率谱的峰值点
locs,_=sg.find_peaks(p)
plt.plot(f,20*np.log10(p))
plt.plot(f[locs],20*np.log10(p[locs]),'rs')
plt.grid()
addActivityLegend.addActivityLegend(np.array([1]))
plt.title('Power Spectral Density with Peaks Estimates')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.show()

# 有要求的标注功率谱峰值点，取最高的8个点

fmindist = 0.25
N = 2*(len(f)-1)
fs=50
minpkdist = math.floor(fmindist/(fs/N))
locs,_=sg.find_peaks(p,distance=minpkdist,prominence=0.15)
locs=locs[np.argpartition(f[locs], 8)[:8]]
plt.plot(f,20*np.log10(p))
plt.plot(f[locs],20*np.log10(p[locs]),'rs')
plt.grid()
addActivityLegend.addActivityLegend(np.array([1]))
plt.title('Power Spectral Density with Peaks Estimates')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.ylim(-80,50)
plt.show()


# 有要求的标注求自相关后的峰值点
abw=np.array(abw).flatten()
c=np.correlate(abw,abw,mode='full')
lags=np.arange(-(len(abw)-1),len(abw))
tmindist = 0.3
minpkdist = np.floor(tmindist/(1/fs))
locs,_= sg.find_peaks(c,prominence=1e4,distance=minpkdist)
tc = (1/fs)*lags
plt.plot(tc,c)
plt.plot(tc[locs],c[locs],'rs')
plt.grid()
plt.xlim(-5,5)
addActivityLegend.addActivityLegend(np.array([1]))
plt.title('Autocorrrelation with Peaks Estimates')
plt.xlabel('Time lag (s)')
plt.ylabel('Correlation')
plt.show()

# 对比不同状态的自相关结果
plotCorrActivityComparisonForSubject.plotCorrActivityComparisonForSubject(1, 'WalkingUpstairs', 'WalkingDownstairs')


# 以下为小作业8
# 判断特征提取函数是否存在以及打印其行数
featureExtractionFcn = 'featuresFromBuffer.py'
dirs=os.path.join('./',featureExtractionFcn)
if not os.path.exists(dirs):
    os.makedirs(dirs)
file = open(dirs)
count = len(file.readlines())
print('\n%d lines of source code found in %s\n'% (count,featureExtractionFcn))
file.close()

# 加载训练数据
buf=scio.loadmat('./data/BufferFeatures.mat')
bufa=scio.loadmat('./data/BufferedAccelerations.mat')
X=buf['feat'].transpose()
actid=bufa['actid']
atx=bufa['atx']
aty=bufa['aty']
atz=bufa['atz']
actnames=bufa['actnames']
t=bufa['t']


# 预处理训练数据及标签
n = np.max(actid)
tgtall = (actid.T[:, None, :] == np.arange(1, n+1)[:, None]).astype(np.int)
tgtall=tgtall[0].transpose()
X=X.transpose()




# 分割训练集，测试集以及验证集
trainInd,uu,trainInd_la,uu_la=sd.train_test_split(X,tgtall,test_size=0.3)
valInd,testInd,valInd_la,testInd_la=sd.train_test_split(uu,uu_la,test_size=0.5)
# 读取网络，这里trained net为利用官方数据训练的lvq网络,trained net2为利用官方数据训练的ff网络
# net=nl.load('trained net')
net=nl.load('trained net2')
# 定义网络，此处可选择newlvq(学习向量量化神经网络)或者newff(多层神经网络)
# net=nl.net.newlvq(nl.tool.minmax(X),8448,[0.1,0.1,0.2,0.2,0.2,0.2])
# net=nl.net.newff(nl.tool.minmax(X),[18,32,32,64,64,32,6])

# 开始训练网络，这里epoch表示训练轮数，show表示显示间隔，goal=-1表示目标error为0
# net.train(trainInd,trainInd_la,epochs=100,show=1,goal=-1)
# 保存网络，名为trained net或trained net2
# net.save(fname='trained net')
# net.save(fname='trained net2')
# 动态显示模块，可用来动态显示加速度及预测标签和真实标签
# for k in range(0,np.size(atx,0)):
#     ax = atx[k,:]
#     ay = aty[k,:]
#     az = atz[k,:]
#     f = featuresFromBuffer.featuresFromBuffer(ax, ay, az, 50)
#     scores=net.sim(f)
#     maxidx = np.argmax(scores)
#     estimatedActivity = actnames[0,maxidx]
#     actualActivity = actnames[0,actid[k,0]-1]
#     plotAccelerationBufferAndPrediction.plotAccelerationBufferAndPrediction(ax, ay, az, t, actualActivity, estimatedActivity)
#     print(k)



# uk表示真实数据标签，scoreval最后得出预测数据标签


# scoreval = net.sim(testInd)
# scoreval = np.argmax(scoreval,1)+1
# uk=np.argmax(testInd_la,1)+1
#
#
#
# #用混淆矩阵显示结果
# pp.pp_matrix_from_data(uk,scoreval, columns=['walk','walkupstairs','walkdownstairs','sit','stand','lay'])


# 下面是用自己采集的数据进行训练以及预测部分

# 提取数据，buf为数据，可根据需要选择3个中的任意一个，前两个为测试数据，最后一个为训练数据
buf=scio.loadmat('./data/mytest.mat')
# buf=scio.loadmat('./data/mytest2.mat')
# buf=scio.loadmat('./data/myrecordedacc.mat')

actid=buf['actid']
atx=buf['atx']
aty=buf['aty']
atz=buf['atz']
t=buf['t']
# 提取网络，这里可提取多种网络，其中mytrained net为利用lvq网络训练的，mytrained net3为利用多层神经网络训练的
# net=nl.load('mytrained net')
net=nl.load('mytrained net3')

# 乱序部分，其实乱序与否都可以，因为这个网络并没有batch的说法，所以无论是否乱序，一次训练都是把所有数据拿来训练的
# np.random.seed(201)
# np.random.shuffle(actid)
# np.random.seed(201)
# np.random.shuffle(atx)
# np.random.seed(201)
# np.random.shuffle(aty)
# np.random.seed(201)
# np.random.shuffle(atz)

# 数据预处理部分，作用是将训练数据提取66个特征并聚集为训练集
# for k in range(0,np.size(atx,0)):
#     ax = atx[k, :]
#     ay = aty[k,:]
#     az = atz[k,:]
#     f = featuresFromBuffer.featuresFromBuffer(ax, ay, az, 50)
#     if k==0:
#         a=f
#         continue
#     a=np.vstack((a,f))


# 数据预处理部分，处理标签数据

# n = np.max(actid)
# tgtall = (actid.T[:, None, :] == np.arange(1, n+1)[:, None]).astype(np.int)
# tgtall=tgtall[0].transpose()


# 定义网络部分，可定义lvq网络或者ff网络

# net=nl.net.newlvq(nl.tool.minmax(a),318336,[0.1,0.1,0.2,0.2,0.2,0.2])
# net=nl.net.newff(nl.tool.minmax(a),[18,32,32,64,64,32,6])

# 训练网络部分

# net.train(a,tgtall,epochs=100,show=1,goal=-1)
# 保存网络
# net.save(fname='mytrained net3')
# net.save(fname='mytrained net')
#预测部分，uk为准确标签，scoreval为预测标签



# uk=actid
# scoreval=net.sim(a)
# scoreval = np.argmax(scoreval,1)+1
#
#
#
#混淆矩阵部分
#
#
# pp.pp_matrix_from_data(uk,scoreval, columns=['walk','walkupstairs','walkdownstairs','sit','stand','lay'])


# 动态显示

for k in range(0,np.size(atx,0)):
    ax = atx[k,:]
    ay = aty[k,:]
    az = atz[k,:]
    f = featuresFromBuffer.featuresFromBuffer(ax, ay, az, 50)
    scores=net.sim(f)
    maxidx = np.argmax(scores)
    estimatedActivity = actnames[0,maxidx]
    actualActivity = actnames[0,actid[k,0]-1]
    myplotAccelerationBufferAndPrediction.plotAccelerationBufferAndPrediction(ax, ay, az, t, actualActivity, estimatedActivity)
    print(k)