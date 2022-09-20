from swarmalator import RK4, Swarm, Alator, PickInitialValues
import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as colormap

#This file declares input variables for use with the swarmalator.py module, and runs them.
#Variable names are self-explanatory.

NSwarmalators = 20
NatFreq = np.array([0.0] * NSwarmalators)
InitVelocityX = np.array([0.0] * NSwarmalators)
InitVelocityY = np.array([0.0] * NSwarmalators)

MinXPosition = -1
MaxXPosition = 1
MinYPosition = -1
MaxYPosition = 1
MinPhase = m.pi*-1
MaxPhase = m.pi

stepSize = 0.1
Terminate = 100
totalLength = (int) (Terminate / stepSize)

InitialX = PickInitialValues(NSwarmalators, MinXPosition, MaxXPosition)
InitialY = PickInitialValues(NSwarmalators, MinYPosition, MaxYPosition)
InitialPhase = PickInitialValues(NSwarmalators, MinPhase, MaxPhase)

CouplingConstant = -.1
PhaseAttraction = 1.0
InitialPositionData = np.stack((InitialX, InitialY, InitVelocityX, InitVelocityY))
InitialPhaseData = np.stack((InitialPhase, NatFreq))

#The line below runs the RK4 function for both equations, and splits the data into separate tables.
swarmalatorData = RK4(Alator, Swarm, InitialPhaseData, InitialPositionData, stepSize, PhaseAttraction, CouplingConstant, Terminate)
thetaTable = np.array(swarmalatorData[0])
xTable = np.array(swarmalatorData[1])
yTable = np.array(swarmalatorData[2])

#here be dragons, and other nightmarish bodging.
#learn from my mistakes, DO NOT do these animations in matplotlib!

csize = 1.7
fig = plt.figure()
ax = plt.gca()
plt.gca().set_aspect('equal')
plt.xlim([-csize, csize])
plt.ylim([-csize, csize])
fig.colorbar(colormap.ScalarMappable(norm=None, cmap='hsv'))
cmap = colormap.hsv

chart_swarmalators = np.zeros(NSwarmalators, dtype=[('position', float, 2),
                                                   ('color',    float, 4)])

chart_swarmalators['position'] = np.transpose(np.stack((xTable[0,:], yTable[0,:])))
scat = ax.scatter(chart_swarmalators['position'][:,0], chart_swarmalators['position'][:,1],c=thetaTable[0,:],cmap='hsv', linewidth=0)

def update(frame_number):
    ax.clear()
    
    plt.xlim([-csize, csize])
    plt.ylim([-csize, csize])

    chart_swarmalators['color'] = cmap(thetaTable[frame_number,:])

    chart_swarmalators['position'] = np.transpose(np.stack((xTable[frame_number,:],yTable[frame_number,:])))
    
    ax.scatter(chart_swarmalators['position'][:,0], chart_swarmalators['position'][:,1],c=thetaTable[frame_number,:],cmap='hsv', linewidth=0)

animation = FuncAnimation(fig, update, frames=totalLength, interval=10, repeat=False)
animation.save('Animation-j={}-k={}-n={}.gif'.format(PhaseAttraction,CouplingConstant,NSwarmalators), writer='imagemagick', fps=30)

