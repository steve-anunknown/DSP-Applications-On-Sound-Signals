# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:59:22 2022

@author: Anastasios Stefanos Anagnostou
         Spyridon Papadopoulos
"""

#1.1

import random
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
plt.style.use("ggplot")

L = 256
w = np.array([np.pi/9, np.pi/5])
A = np.array([1.0, 0.9])
phase = np.array([random.uniform(0,2*np.pi), random.uniform(0,2*np.pi)])
n = np.linspace(0,L-1,L)

hammwin = np.hamming(L)
x = [A[0]*np.exp(1j*(w[0]*n+phase[0])), A[1]*np.exp(1j*(w[1]*n+phase[1]))]
y = (x[0]+x[1])*hammwin

#number of DFT samples
N = 256

y_fft = np.fft.fft(y,N)

fig_num = 0 #counter to keep track of figures
plt.figure(fig_num)
plt.title("DFT of signal y with N = 256 samples, {}={}{}, {}={}{}".format(
        'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π'))
plt.plot(np.linspace(0,2*np.pi,np.size(y_fft)),abs((y_fft)))
plt.xlim([0,2*np.pi/5])

#define how small the increment should be
STEPS = 5
increment = (w[1]-w[0])/STEPS

#loop that generates new dft plots in order to compare the peaks
dw = [w[1]-w[0]]
w[1] = w[1]-increment
fig, axis = plt.subplots(STEPS)
fig.suptitle("DFT of signal with a stable frequency {}={}{} \n and a moving frequency starting from {}={}{}".format(
    'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π'))

while w[0]<=w[1]:
    fig_num += 1
    dw.append(w[1]-w[0])
    x[1] = A[1]*np.exp(1j*(w[1]*n+phase[1]))
    y = (x[0]+x[1])*hammwin
    y_fft = np.fft.fft(y,N)
    axis[fig_num-1].plot(np.linspace(0,2*np.pi,np.size(y_fft)),abs((y_fft)))
    axis[fig_num-1].set_xlim(0,2*np.pi/5)
    w[1] = w[1]-increment
# =============================================================================    

# =============================================================================

#1.2

L = 256
w = np.array([np.pi/9, np.pi/5])
A = np.array([1.0, 0.9])
phase = np.array([random.uniform(0,2*np.pi), random.uniform(0,2*np.pi)])
n = np.linspace(0,L-1,L)
hammwin = np.hamming(L)
x = [A[0]*np.exp(1j*(w[0]*n+phase[0])), A[1]*np.exp(1j*(w[1]*n+phase[1]))]
y = (x[0]+x[1])*hammwin

#number of DFT samples
N = np.array([512, 1024])
fftlinspace = np.linspace(0,N[1]-1,N[1])
zero_padding = np.zeros(N[1])

y_zero_padded = [np.append(y,zero_padding[0:N[0]]),np.append(y,zero_padding)]

y_zero_padded_fft = [np.fft.fft(y_zero_padded[0],N[0]), np.fft.fft(y_zero_padded[1],N[1])]

fig_num +=1 #counter to keep track of figures

fig, axs = plt.subplots(2)
fig.suptitle("DFT of zero padded Y signal {}={}{}, {}={}{} \n Samples = {},{}".format(
        'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π',N[0],N[1]))
for index, elem in enumerate(axs,start=0):
    elem.plot(np.linspace(0,2*np.pi,np.size(y_zero_padded_fft[index])),
              abs((y_zero_padded_fft[index])))
    elem.set_xlim(0,2*np.pi/5)
fig_num+=1

#define how small the increment should be
STEPS = 5
increment = (w[1]-w[0])/STEPS

#loop that generates new dft plots in order to compare the peaks
dw = [w[1]-w[0]]
w[1] = w[1]-increment
while w[0]<=w[1]:
    fig_num += 1
    dw.append(w[1]-w[0])
    x[1] = A[1]*np.exp(1j*(w[1]*n+phase[1]))    
    y = (x[0]+x[1])*hammwin
    y_zero_padded = [np.append(y,zero_padding[0:N[0]]),np.append(y,zero_padding)]
    y_zero_padded_fft = [np.fft.fft(y_zero_padded[0],N[0]), np.fft.fft(y_zero_padded[1],N[1])]
    fig, axs = plt.subplots(2)
    fig.suptitle("DFT of zero padded Y signal {}={}{}, {}={}{} \n Samples = {},{}".format(
        'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π',N[0],N[1]))
    for index, elem in enumerate(axs,start=0):
        elem.plot(np.linspace(0,2*np.pi,np.size(y_zero_padded_fft[index])),
                  abs((y_zero_padded_fft[index])))
        elem.set_xlim(0,2*np.pi/5)
    w[1] = w[1]-increment

# =============================================================================



# =============================================================================

#1.3


#frequency where the peaks start covering each other w ~= 0.140pi

L = np.array([512,1024])
w = np.array([np.pi/9, np.pi*0.7/5])
A = np.array([1.0, 0.9])
phase = np.array([random.uniform(0,2*np.pi), random.uniform(0,2*np.pi)])


n = [np.linspace(0,L[0]-1,L[0]),np.linspace(0,L[1]-1,L[1])]
hammwin = [np.hamming(L[0]),np.hamming(L[1])]
x = [[A[0]*np.exp(1j*(w[0]*n[0]+phase[0])), A[1]*np.exp(1j*(w[1]*n[0]+phase[1]))],
    [A[0]*np.exp(1j*(w[0]*n[1]+phase[0])), A[1]*np.exp(1j*(w[1]*n[1]+phase[1]))]]
y = [(x[0][0]+x[0][1])*hammwin[0],(x[1][0]+x[1][1])*hammwin[1]]

y_fft = [np.fft.fft(y[0],L[0]),np.fft.fft(y[1],L[1])]

fig, axs = plt.subplots(2)
fig.suptitle("DFT of Y signal {}={}{}, {}={}{} \n Samples = {},{}".format(
        'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π',L[0],L[1]))
xlim = L/5
for index, elem in enumerate(axs,start=0):
    elem.plot(np.linspace(0,2*np.pi,np.size(y_fft[index])),abs((y_fft[index])))
    elem.set_xlim(0,2*np.pi/5)
fig_num+=1

#define how small the increment should be
STEPS = 5
increment = (w[1]-w[0])/STEPS

#loop that generates new dft plots in order to compare the peaks
dw = [w[1]-w[0]]
w[1] = w[1]-increment
while w[0]<=w[1]:
    fig_num += 1
    dw.append(w[1]-w[0])
    for index, space in enumerate(n,start=0):
        x[index][1] = A[1]*np.exp(1j*(w[1]*space+phase[1]))
    y = [(x[0][0]+x[0][1])*hammwin[0],(x[1][0]+x[1][1])*hammwin[1]]
    y_fft = [np.fft.fft(y[0],L[0]),np.fft.fft(y[1],L[1])]    
    fig, axs = plt.subplots(2)
    fig.suptitle("DFT of Y signal {}={}{}, {}={}{} \n Samples = {},{}".format(
        'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π',L[0],L[1]))
    for index, elem in enumerate(axs,start=0):
        elem.plot(np.linspace(0,2*np.pi,np.size(y_fft[index])),abs((y_fft[index]))) #n[index]
        elem.set_xlim(0,2*np.pi/5)
    w[1] = w[1]-increment

# =============================================================================
# Inspecting the DFT plots we can infer that the spectral distinguishability
# of the sin signals increases as the number of samples increases. This comes
# as expected, since a larger number of signal samples (as opposed to zero padding)
# results in a more precise spectrum. However, performing the DFT onto the zero
# padded signal results in a more detailed spectrum.
# =============================================================================


# =============================================================================



# =============================================================================

#1.4

L = 256
w = np.array([0.35*np.pi, 0.4*np.pi])
phases = np.array([random.uniform(0,2*np.pi), random.uniform(0,2*np.pi)])
A = np.array([1, 0.05])
hammwin = np.hamming(L)
n = np.linspace(0,L-1,L)
x = [A[0]*np.exp(1j*(w[0]*n+phase[0])), A[1]*np.exp(1j*(w[1]*n+phase[1]))]
y = [x[0]+x[1],(x[0]+x[1])*hammwin]

#number of DFT samples
N = 2**10
fftlinspace = np.linspace(0,N-1,N)

y = [np.append(y[0],np.zeros(N-L)),np.append(y[1],np.zeros(N-L))]

y_fft = [np.fft.fft(y[0],N), np.fft.fft(y[1],N)]

#fig_num = 0 counter to keep track of figures

fig, axs = plt.subplots(2)
fig.suptitle("DFT Y signal {}={}{}, {}={}{} \n Samples = {}, w(n) = 1, hamming(n)".format(
        'ω1',round(w[0]/np.pi,3),'π','ω2',round(w[1]/np.pi,3),'π',N))
for index, elem in enumerate(axs,start=0):
    elem.plot(np.linspace(0,2*np.pi,np.size(y_fft[index])),abs(y_fft[index]))
    elem.set_xlim(0,2*np.pi/3)
fig_num+=1

# =============================================================================
# The difference between using an ordinary window instead of the
# hamming window is found in the resulting spectrum of the signal.
# In the first case, the DFT of the signal depicts some spectral
# content in every frequency, especially around the signal's fundamental
# frequency, whereas in the case of the hamming window, these
# extra frequencies are cut out. Most notably, the hamming window
# is effective enough so as to make the "weak" frequency component
# visible (weak because of its lower amplitude), in contrast to the
# case of the ordinary window, where the weak frequency component
# is almost completely overwhelmed by the surrounding frequencies.
# As a result, it is clear that the hamming window is the superior
# way to filter a signal, at least when there is interest in depicting
# weaker frequency components that are close to the fundamental
# frequencies of the signal.
# =============================================================================


# =============================================================================
