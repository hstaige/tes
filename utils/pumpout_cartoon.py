import numpy as np
import matplotlib.pyplot as plt

def pump_curve(t,dt,N0,rate):
   return N0*np.exp(-rate*(t-dt))

init_t = np.arange(0,6,1)

# TES
plt.plot(init_t,np.ones_like(init_t),color='b',label='TES')
plt.plot(np.arange(5,81,1),pump_curve(np.arange(5,81,1),5,1,.01),color='b')
plt.plot(np.arange(80,120,1),pump_curve(np.arange(80,120,1),80,pump_curve(76,1,1,.01),.1),color='b')
plt.axvline(80,color='k')
plt.text(81,.5,'Open SuckMaster',rotation=90)

# Dead Volume
plt.plot(init_t,np.ones_like(init_t),color='r',label='Dead Volume')
plt.plot(np.arange(5,120,1),pump_curve(np.arange(5,120,1),5,1,.1),color='r')
plt.axvline(5,color='k')
plt.text(2,.5,'Start TES/Dead Pump',rotation=90)

# Cal
plt.plot(np.arange(0,91,1),np.ones_like(np.arange(0,91,1)),color='y',label='Cal Volume')
plt.plot(np.arange(90,120,1),pump_curve(np.arange(90,120,1),90,1,.1),color='y')
plt.axvline(90,color='k')
plt.text(91,.4,'Start Cal Pump',rotation=90)

plt.legend()
plt.ylabel('Pressure (arb, approx atm)')
plt.xlabel('Time [min]')
plt.show()