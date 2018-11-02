import matplotlib.pyplot as plt
import numpy as np

f = 50.0
g=9.81



t=np.linspace(0,1/200,100)
dh = 1000*(0.5*g*np.square(t) - (0.188E-3/(2*np.pi*f))*(1 - np.cos(2*np.pi*f*t)))

plt.figure()
plt.plot(t,dh,'rx')
plt.show()