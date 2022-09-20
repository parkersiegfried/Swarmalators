import math as m
import cmath as cm
import numpy as np
import matplotlib.pylab as lab
import random as rand
from copy import deepcopy

def Swarm(x, y, vx, vy, J, theta):
    A = 1 #A and B are arbitrary constants. Both are 1 for simplicity.
    B = 1
    N = len(x) #Defines the number of distinct oscillators N as the size of one input array, under the assumption every input array is the same size
    dPosx = np.array(vx) #Initialize two numpy arrays to be used for output.
    dPosy = np.array(vy) #Set their initial value to their natural velocity.
    for i in range(len(x)):
        for k in range(len(x)):
            if k == i:
                continue
            #swarming equation in component form
            dPosx[i] += (1/N) * (((x[k] - x[i])/m.sqrt(((x[k]-x[i])**2+(y[k]-y[i])**2))) * (A + (J*m.cos(theta[k]-theta[i]))) - ((B) * ((x[k]-x[i]) / (((x[k]-x[i])**2+(y[k]-y[i])**2))) ))
            dPosy[i] += (1/N) * (((y[k] - y[i])/m.sqrt(((x[k]-x[i])**2+(y[k]-y[i])**2))) * (A + (J*m.cos(theta[k]-theta[i]))) - ((B) * ((y[k]-y[i]) / (((x[k]-x[i])**2+(y[k]-y[i])**2))) ))
        
    dPos = np.stack((dPosx, dPosy))
    #this will return outputs of the form:
    #array[[x0, x1, x2, ..., xN]
    #      [y0, y1, y2, ..., yN]]
    return dPos

def Alator(theta, omega, K, x, y):
    N = len(theta) 
    dTheta = omega
    for i in range(N):
        dTheta[i]=0
        for j in range(N):
            if j == i:
                continue
            dTheta[i] += (K/N) * ((m.sin(theta[j]-theta[i]))/(m.sqrt((x[j]-x[i])**2+(y[j]-y[i])**2)))
            #eqn. 4
        
    
    #This function will return outputs of the form:
    #array[theta0, theta1, ..., thetaN]
    return dTheta

def RK4(f, g, FInit, GInit, h, J, K, terminate):
    #4th-order Runge-Kutta algorithm for 2 coupled equations.
    #FInit and GInit expect a 2-dimensional numpy array.
    theta = FInit[0,:] #first row of phase equation inputs: initial phase values
    omega = FInit[-1,:] #second row: initial angular speeds
    x = GInit[0,:] #first and second rows of position equation inputs: initial position data
    y = GInit[1,:]
    vx = GInit[2,:] #third and fourth rows of position equation inputs: initial 
    vy = GInit[3,:]
    print(vx)
    newX = [None] * len(x)
    newY = [None] * len(y)
    nextTheta = [None] * len(theta)
    print("theta: {}\nx:{}\ny:{}\nvx:{}\nvy:{}".format(theta,x,y,vx,vy))
    
    N = (int)(terminate/h)
    timeLine = [None] * N
    for t in range(N):
        timeLine[t] = h*t
        #Translated: the time value at step t is equal to the amount of time steps taken multiplied by the step size
    
    thetasTimeTable = [None] * N
    xTimeTable = [None] * N
    yTimeTable = [None] * N

    for m in range(N):        

        k1 = f(theta, omega, K, x, y)
        j1 = g(x, y, vx, vy, J, theta)

        for j in range(0,len(x)):
            #j1 outputs a 2d array of dx/dt and dy/dt - row 0 has x values, row 1 has y values
            newX[j] = x[j] + h * (j1[0,j] / 2)
            newY[j] = y[j] + h * (j1[1,j] / 2)
            
        for i in range(0,len(theta)):
            nextTheta[i] = theta[i]+h*(k1[i]/2)
            
        k2 = f(nextTheta, omega, K, newX, newY)
        j2 = g(newX, newY, vx, vy, J, nextTheta)
        
        for j in range(0,len(x)):
            newX[j] = newX[j] + h*(j2[0,j] / 2)
            newY[j] = newY[j] + h*(j2[1,j] / 2)
        
        for i in range(0,len(theta)):
            nextTheta[i] = nextTheta[i] + h*(k2[i] / 2)
        
        k3 = f(nextTheta, omega, K, newX, newY)
        j3 = g(newX, newY, vx, vy, J, nextTheta)
        
        for j in range(0,len(x)):
            newX[j] = newX[j] + h*(j3[0,j])
            newY[j] = newY[j] + h*(j3[1,j])
        
        for i in range(0,len(theta)):
            nextTheta[i] = nextTheta[i] + h*(k3[i])
        
        k4 = f(nextTheta, omega, K, newX, newY)
        j4 = g(newX, newY, vx, vy, J, nextTheta)
        
        for i in range(0,len(theta)):
            theta[i] += ((k1[i] + 2 * (k2[i] + k3[i]) + h*k4[i])/6)
            x[i] += ((j1[0,i] + 2 * (j2[0,i] + j3[0,i]) + h*j4[0,i])/6)
            y[i] += ((j1[1,i] + 2 * (j2[1,i] + j3[1,i] + h*j4[1,i]))/6)

        thetasTimeTable[m] = deepcopy(theta) #deepcopies here prevent any pointer confusion without much effect on performance
        xTimeTable[m] = deepcopy(x)
        yTimeTable[m] = deepcopy(y)
            
        #This should return an argument of the form:
    #tuple(array[theta values], array[x values], array[y values])
    return (thetasTimeTable, xTimeTable, yTimeTable)

def PickInitialValues(size, low, high):
    #Declare an array with [size] number of empty elements.
    outList = [None] * size
    #Initialize randomness. 
    rand.seed()
    for N in range(size):
        #Write a (pseudo)random float between the minimum and maximum allowed values to N.
        outList[N] = rand.uniform(low, high)
    #Return the full list of these random floats.
    outList = np.array(outList)
    return outList

