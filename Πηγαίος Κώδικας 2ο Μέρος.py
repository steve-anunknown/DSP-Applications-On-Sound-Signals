# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:00:45 2022

@author: Dell
"""

# =============================================================================

#2.1

import numpy as np
import matplotlib.pyplot as plt
import soundfile
from scipy.signal import find_peaks

plt.close('all')
plt.style.use("ggplot")

L = 1000
wrow = np.array([0.5346, 0.5906, 0.6535, 0.7217])
wcol = np.array([0.9237, 1.0247, 1.1328])
S_RATE = 8192
n = np.linspace(0,L-1,L)
d = [np.sin(wrow[3]*n)+np.sin(wcol[1]*n)]

for i in range(len(wrow)-1):
    for j in range(len(wcol)):
        d = np.append(d,[np.sin(wrow[i]*n)+np.sin(wcol[j]*n)],axis = 0)

# =============================================================================

# =============================================================================

#2.2

EXTRA_Z = 9000
d_fft5 = np.fft.fft(np.append(d[5],np.zeros(EXTRA_Z)))
d_fft8 = np.fft.fft(np.append(d[8],np.zeros(EXTRA_Z)))

fig_num=0 # added so that part 2 can be executed independetly
plt.figure(fig_num)
plt.plot(np.linspace(0,np.pi,int((L+EXTRA_Z)/2)),abs(d_fft5[0:int((L+EXTRA_Z)/2)]))
plt.title("DFT of tone d[5] with L = 1000 samples")
fig_num+=1
plt.figure(fig_num)
plt.plot(np.linspace(0,np.pi,int((L+EXTRA_Z)/2)),abs(d_fft8[0:int((L+EXTRA_Z)/2)]))
plt.title("DFT of tone d[8] with L = 1000 samples")
fig_num+=1

# =============================================================================

# =============================================================================

#2.3

Z = 100
AM1 = 3119051
AM2 = 3119058
SEQ_DIGITS = AM1+AM2
tone_sequence = np.array([d[0]])
for num_char in str(SEQ_DIGITS):
    tone_sequence=np.append(tone_sequence,np.zeros(Z))
    tone_sequence=np.append(tone_sequence,[d[int(num_char)]])
soundfile.write('tone_sequence.wav',tone_sequence/max(tone_sequence),S_RATE)

# =============================================================================

# =============================================================================

#2.4

#Square Window

tone_sequence_fft = np.array(
                        np.fft.fft(
                            np.append(tone_sequence[0:L],np.zeros(EXTRA_Z))
                        )
                    )
#define this function just to save space
def myplot(message,fig_numb):
    plt.figure(fig_numb)
    plt.plot(np.linspace(0,np.pi,int((L+EXTRA_Z)/2)),abs(tone_sequence_fft[0:int((L+EXTRA_Z)/2)]))
    plt.title(message)

myplot("DFT of tone 1 with L = 1000 samples and {} extra zeroes".format(EXTRA_Z),fig_num)
fig_num+=1

zeroes = 0
index = L
for index in range(L,len(tone_sequence)-2*L,L):
    zeroes = int(Z*(index/L))
    tone_sequence_fft =np.fft.fft(
                            np.append(tone_sequence[index+zeroes:index+zeroes+L],
                                      np.zeros(EXTRA_Z))
                                    )
    myplot("DFT of tone {} with L = 1000 samples and {} extra zeroes".format(
        1+int(index/L),EXTRA_Z),fig_num)
    fig_num+=1

index+=L
tone_sequence_fft = np.fft.fft(
                            np.append(tone_sequence[index+zeroes:index+zeroes+L],np.zeros(EXTRA_Z))
                            )

myplot("DFT of tone {} with L = 1000 samples and {} extra zeroes".format(
        1+int(index/L),EXTRA_Z),fig_num)
fig_num+=1

#Hamming window
tone_sequence_fft = np.array(
                                np.fft.fft(
                                    np.append(
                                        np.hamming(L)*tone_sequence[0:L],
                                        np.zeros(EXTRA_Z)
                                            )
                                )
                    )


myplot("DFT of hamming tone 1 with L = 1000 samples and {} extra zeroes".format(EXTRA_Z),fig_num)
fig_num+=1


for index in range(L,len(tone_sequence)-L,L):
    zeroes = int(Z*(index/L))
    tone_sequence_fft = np.fft.fft(
                                    np.append(
                                        np.hamming(L)*tone_sequence[index+zeroes:index+zeroes+L],
                                        np.zeros(EXTRA_Z)
                                        )
    )
    myplot("DFT of hamming tone {} with L = 1000 samples and {} extra zeroes".format(
            1+int(index/L),EXTRA_Z),fig_num)
    fig_num+=1

# =============================================================================
# It seems that the spikes that occur in the DFT of the signals showcase different
# heights. The reason why this happens is that not too many samples were used for
# the DFT and, as a result, there is a lower probability of hitting the exact spot
# where the maximum occurs. If more samples were to be used, we would indeed notice
# that the spikes are of the same height.
# =============================================================================

# =============================================================================    
    
# =============================================================================

#2.5
k_row = [int(x) for x in wrow*S_RATE/(2*np.pi)]
k_col = [int(x) for x in wcol*S_RATE/(2*np.pi)]

# =============================================================================


# =============================================================================

# #2.6

def find_digit(frequencies, row_freqs, col_freqs):
    row_ind = -1
    col_ind = -1
    error_margin = 0.02 #acceptable error
    for freq in frequencies:
        if freq < (row_freqs[0]+col_freqs[2])/2:
            for indx, row in enumerate(row_freqs, start=0):
                if abs(row-freq)/row<error_margin:
                    row_ind = indx
        else:
            for indx, col in enumerate(col_freqs, start=0):
                if abs(col-freq)/col<error_margin:
                    col_ind = indx
    if row_ind == -1 or col_ind ==-1:
        return -1
    if row_ind == 3 and col_ind == 1:
        return 0
    return row_ind*3+col_ind+1

def mean_energy(signal):
    return sum([x*x for x in signal])/np.size(signal)

def ttdecode(signIn, row_freqs = wrow , col_freqs = wcol):
    vector = '' #initialisation of the result
    num = 100   #number of windows that 'fit' into the signal
    length = np.size(signIn)    
    width = int(length/num) #width of the window, 1/100 of signal
    zero_pads = 20*width    #extra zeroes in order to ensure good DFT quality
    window = np.hamming(width)
    #indexes for the given frequencies
    k_index_row = [int(x) for x in row_freqs*(width+zero_pads)/(2*np.pi)]
    k_index_col = [int(x) for x in col_freqs*(width+zero_pads)/(2*np.pi)]
    step = max(int(width/10),1) #size of window step
    threshold = mean_energy(signIn)*1 #threshold to qualify as a tone
    indx = 0
    tone = []
    tone_fft = []
    peak_freqs = []
    window_energy = 0
    while indx + width <= length:
        window_energy = mean_energy(signIn[indx:indx+width])
        if window_energy < threshold:
            #not recognised as a tone
            while window_energy < threshold and indx + width + step <= length:
                #still not on a tone
                indx += step
                window_energy = mean_energy(signIn[indx:indx+width])
                #keep looping until a tone is found        
        if window_energy >= threshold:
            #recognised a tone
            tone = signIn[indx:indx+width]
            tone_fft = np.fft.fft(np.append(window*tone,np.zeros(zero_pads)))
            tone_fft = abs(tone_fft[0:int((width+zero_pads)/2)])
            peak_freqs, _ = find_peaks(tone_fft,
                                       distance = 10,
                                       height = 0.2*max(abs(tone_fft)))
            digit = find_digit(peak_freqs, k_index_row, k_index_col)
            vector += str(digit) + ' '
            while window_energy >= threshold and indx + width + step <= length:
                #still on the same tone
                indx += step
                window_energy = mean_energy(signIn[indx:indx+width])
                #keep looping until noise or gap is found
        indx+=step
    return vector

print("tone sequence = {}".format(ttdecode(tone_sequence)))

# =============================================================================
# tone sequence = 0 6 2 3 8 1 0 9 
# easy signal = 7 3 5 8 2 8 0 2 
# hard signal = 4 3 5 6 6 2 0 9 9 5 
# =============================================================================

# =============================================================================



# =============================================================================
# #2.7

easy_sig = np.load('easy_sig.npy')
hard_sig = np.load('hard_sig.npy')

print("easy signal = {}".format(ttdecode(easy_sig)))
print("hard signal = {}".format(ttdecode(hard_sig)))

# =============================================================================