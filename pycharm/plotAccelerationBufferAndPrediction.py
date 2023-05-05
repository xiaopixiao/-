
import matplotlib.pyplot as plt



def plotAccelerationBufferAndPrediction(x,y,z,t, actual, estimated):
    g = 9.81
    plt.figure(1)
    x2=x.transpose()
    y2 = y.transpose()
    z2 = z.transpose()
    plt.plot(t[0,:], g*x2, linewidth= 1.5)
    plt.plot(t[0,:], g * y2, linewidth=1.5)
    plt.plot(t[0,:], g * z2, linewidth=1.5)
    plt.xlim(0,t[0,-1])
    plt.ylim(-2 * g,2 * g)
    plt.xlabel('Time offset (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Acceleration (%s)\nactual(%s)'%(estimated[0],actual[0]))
    plt.legend(labels=['a_x', 'a_y', 'a_z'])
    plt.grid()
    plt.draw()
    plt.pause(0.1)
    plt.clf()