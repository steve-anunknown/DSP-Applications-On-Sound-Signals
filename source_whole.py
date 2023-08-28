# -*- coding: utf-8 -*-
# =============================================================================
#1.1

import random
import numpy as np
import matplotlib.pyplot as plt
import soundfile
from scipy.signal import find_peaks

plt.close("all")

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

plt.style.use("ggplot")

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



# =============================================================================

#2.1

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
    step = max(int(width/20),1) #size of window step
    threshold = mean_energy(signIn)*0.85    #threshold to qualify as a tone
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
                                       height = 0.5*max(abs(tone_fft)))
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



# =============================================================================
# #2.7

easy_sig = np.load('easy_sig.npy')
hard_sig = np.load('hard_sig.npy')

print("easy signal = {}".format(ttdecode(easy_sig)))
print("hard signal = {}".format(ttdecode(hard_sig)))

# =============================================================================


# =============================================================================

#3.1

speech_utterance, speech_srate = soundfile.read('speech_utterance.wav')
times = np.array([20,25,30,35,40,45,50])/1000
sizes = [int(t*speech_srate) for t in times]
speech_length = np.size(speech_utterance)

def rates(signIn,sampling_rate,fig_numb, widths = sizes):
    help_signal = abs(np.sign(signIn[1:])-np.sign(signIn[:-1]))
    for width in widths:
        short_time_en = np.convolve([x*x for x in signIn], np.hamming(width))
        plt.figure(fig_numb)
        plt.plot(np.linspace(0,int(np.size(signIn)/sampling_rate),
                             np.size(short_time_en)),short_time_en)
    plt.title("short time energy using window of varying \n sizes = {}".format(widths))
    fig_numb+=1
    for width in widths:
        zero_cross_rate = np.convolve(help_signal, np.hamming(width))
        plt.figure(fig_numb)
        plt.plot(np.linspace(0,int(np.size(signIn)/sampling_rate),
                             np.size(zero_cross_rate)),zero_cross_rate)
    plt.title("zero crossing rate using windows of varying \n sizes = {}".format(widths))
    fig_numb+=1
    return fig_numb

plt.style.use("ggplot")
fig_num = rates(speech_utterance,speech_srate,fig_num,sizes)

short_time_energy = np.convolve([x*x for x in speech_utterance],np.hamming(sizes[0]))
short_time_energy = short_time_energy / max(short_time_energy)

help_sign = abs(np.sign(speech_utterance[1:])-np.sign(speech_utterance[:-1]))
zero_crossing_rate = np.convolve(help_sign,np.hamming(sizes[0]))
zero_crossing_rate = zero_crossing_rate / max(zero_crossing_rate)

plt.figure(fig_num)
plt.plot(np.linspace(0,int(speech_length/speech_srate),speech_length)
          ,speech_utterance/max(speech_utterance))
plt.plot(np.linspace(0,int(speech_length/speech_srate),np.size(short_time_energy))
          ,short_time_energy)
plt.title("speech utterance and short time energy normalised")
fig_num +=1

plt.figure(fig_num)
plt.plot(np.linspace(0,int(speech_length/speech_srate),speech_length)
          ,speech_utterance/max(speech_utterance))
plt.plot(np.linspace(0,int(speech_length/speech_srate),np.size(zero_crossing_rate))
          ,zero_crossing_rate)
plt.title("speech utterance and zero crossing rate normalised")
fig_num +=1

# =============================================================================
# fig, axis = plt.subplots(3, sharex = False)
# axis[0].plot(np.linspace(0,int(speech_length/speech_srate),speech_length)
#           ,speech_utterance/max(speech_utterance))
# axis[0].title.set_text("speech utterance")
# axis[1].plot(np.linspace(0,int(speech_length/speech_srate),np.size(short_time_energy))
#           ,short_time_energy)
# axis[1].title.set_text("short time energy")
# axis[2].specgram(speech_utterance/max(abs(speech_utterance)), Fs = speech_srate, mode = 'magnitude',
#              NFFT = 2**10, pad_to = 2**15, noverlap = (2**10)//1.05,
#              scale = 'dB',vmax=-30,vmin=-80,cmap = 'inferno' )
# axis[2].grid(False)
# axis[2].title.set_text("spectrogram")
# axis[2].set_xlabel("Time (sec)")
# axis[2].set_ylabel("Frequency (Hz logscale)")
# axis[2].set_yscale('log')
# axis[2].set_ylim(100,speech_srate//2)
# fig_num+=1
# 
# fig, axis = plt.subplots(3, sharex = False)
# axis[0].plot(np.linspace(0,int(speech_length/speech_srate),speech_length)
#           ,speech_utterance/max(speech_utterance))
# axis[0].title.set_text("speech utterance")
# axis[1].plot(np.linspace(0,int(speech_length/speech_srate),np.size(zero_crossing_rate))
#           ,zero_crossing_rate)
# axis[1].title.set_text("zero crossing rate")
# axis[2].specgram(speech_utterance/max(abs(speech_utterance)), Fs = speech_srate, mode = 'magnitude',
#              NFFT = 2**10, pad_to = 2**15, noverlap = (2**10)//1.05,
#              scale = 'dB',vmax=-30,vmin=-80,cmap = 'inferno' )
# axis[2].grid(False)
# axis[2].title.set_text("spectrogram")
# axis[2].set_xlabel("Time (sec)")
# axis[2].set_ylabel("Frequency (Hz logscale)")
# axis[2].set_yscale('log')
# axis[2].set_ylim(100,speech_srate//2)
# fig_num+=1
# =============================================================================


plt.figure(fig_num)
plt.style.use("default")
Pxx, freqs, bins, im = plt.specgram(speech_utterance/max(abs(speech_utterance)), Fs = speech_srate, mode = 'magnitude',
             NFFT = 2**10, pad_to = 2**15, noverlap = (2**10)//1.05,
             scale = 'dB',vmax=-30,vmin=-80,cmap = 'inferno' )
plt.title("speech utterance spectrogram")
plt.yticks(np.arange(100,speech_srate//2,step=500),np.arange(100,speech_srate//2,step=500))
plt.ylim(100,speech_srate//2)
plt.xlabel("Time (sec)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(im)
fig_num +=1

# =============================================================================
# It seems that the amplitude of the energy and of the zero crossing rate grows
# as the size of the window used grows. When the amplitude of the speech signal
# grows, the zero crossing rate seems to get smaller. As a result, voice can be
# distinguised from silence (or noise) by noticing where the zero crossing rate
# is small and where it's large respectively. Accordingly, voiced sounds can be
# distinguised from unvoiced sounds using the same thinking.
# =============================================================================

music_signal, music_srate = soundfile.read("music.wav")
music_signal = np.transpose(music_signal)
sizes = [int(t*music_srate) for t in times]
if music_signal.ndim == 2:    
    music_signal = (music_signal[1]+music_signal[0])/2
music_length = np.size(music_signal)

plt.style.use("ggplot")
fig_num = rates(music_signal,music_srate,fig_num,sizes)

short_time_energy = np.convolve([x*x for x in music_signal],np.hamming(sizes[0]))
short_time_energy = short_time_energy / max(short_time_energy)

help_sign = abs(np.sign(music_signal[1:])-np.sign(music_signal[:-1]))
zero_crossing_rate = np.convolve(help_sign,np.hamming(sizes[0]))
zero_crossing_rate = zero_crossing_rate / max(zero_crossing_rate)

plt.figure(fig_num)
plt.plot(np.linspace(0,int(music_length/music_srate),music_length)
          ,music_signal/max(music_signal))
plt.plot(np.linspace(0,int(music_length/music_srate),np.size(short_time_energy))
          ,short_time_energy)
plt.title("riders on the storm and short time energy normalised")
fig_num +=1

plt.figure(fig_num)
plt.plot(np.linspace(0,int(music_length/music_srate),music_length)
          ,music_signal/max(music_signal))
plt.plot(np.linspace(0,int(music_length/music_srate),np.size(zero_crossing_rate))
          ,zero_crossing_rate)
plt.title("riders on the storm and zero crossing rate normalised")
fig_num +=1

fig, axis = plt.subplots(3, sharex = False)
axis[0].plot(np.linspace(0,int(music_length/music_srate),music_length)
          ,music_signal/max(music_signal))
axis[0].title.set_text("riders on the storm")
axis[1].plot(np.linspace(0,int(music_length/music_srate),np.size(short_time_energy))
          ,short_time_energy)
axis[1].title.set_text("short time energy")
axis[2].specgram(music_signal, Fs = music_srate,
             NFFT = 2**10, pad_to = 2**13, cmap="inferno",
             scale = 'dB', noverlap = (2**10)//1.5,
             vmax=-30, vmin=-80, mode = 'magnitude')
axis[2].grid(False)
axis[2].title.set_text("spectrogram")
axis[2].set_xlabel("Time (sec)")
axis[2].set_ylabel("Frequency (Hz logscale)")
axis[2].set_yscale('log')
axis[2].set_ylim(100,music_srate//2)
fig_num+=1

fig, axis = plt.subplots(3, sharex = False)
axis[0].plot(np.linspace(0,int(music_length/music_srate),music_length)
          ,music_signal/max(music_signal))
axis[0].title.set_text("riders on the storm")
axis[1].plot(np.linspace(0,int(music_length/music_srate),np.size(zero_crossing_rate))
          ,zero_crossing_rate)
axis[1].title.set_text("zero crossing rate")
axis[2].specgram(music_signal, Fs = music_srate,
             NFFT = 2**10, pad_to = 2**13, cmap="inferno",
             scale = 'dB', noverlap = (2**10)//1.5,
             vmax=-30, vmin=-80, mode = 'magnitude')
axis[2].grid(False)
axis[2].title.set_text("spectrogram")
axis[2].set_xlabel("Time (sec)")
axis[2].set_ylabel("Frequency (Hz logscale)")
axis[2].set_yscale('log')
axis[2].set_ylim(100,music_srate//2)
fig_num+=1

# =============================================================================
