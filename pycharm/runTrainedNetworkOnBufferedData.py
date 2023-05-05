import scipy.io as scio



def runTrainedNetworkOnBufferedData():
    sn = scio.loadmat('./data/TrainedNetwork.mat', 'net')