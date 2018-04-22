import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

T=[0.9294117647,0.9294117647,0.9294117647,0.9295624333,0.9294117647,0.9294117647,0.9294117647,0.9295624333,0.9254966887,0.9042979943,0.9006354708,0.8895382817,0.8800471559,0.8740740741,0.8666865494,0.8578298768,0.8481357987,0.8481357987,0.8481357987,0.8481357987,0.8331797235]
s=[0.6138107417,0.6138107417,0.6138107417,0.6138107417,0.6138107417,0.6138107417,0.6138107417,0.6138107417,0.4402985075,0.200913242,0.1674418605,0.1327014218,0.1148325359,0.1057692308,0.0966183575,0.0873786408,0.0780487805,0.0780487805,0.0780487805,0.0780487805,0.068627451]

T= np.array(T)
s=np.array(s)

xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max

print(s)

power_smooth = spline(T,s,xnew)


plt.plot(xnew,power_smooth)
plt.show()
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)

# plt.xlabel('time (s)')
# plt.ylabel('voltage (mV)')
# plt.title('About as simple as it gets, folks')
# plt.grid(True)
# plt.savefig("test.png")
# plt.show() 	