# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:01:25 2022

@author: Dell
"""

# =============================================================================

#3.1

import numpy as np
import matplotlib.pyplot as plt
import soundfile

plt.close('all')
plt.style.use("ggplot")

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

fig_num = 0 # added so that part 3 can be executed independetly
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
# axis[2].specgram(speech_utterance, Fs = speech_srate, mode = 'magnitude',
#              NFFT = 2**10, pad_to = 2**13, noverlap = (2**10)//1.5,
#              scale = 'dB',vmax=-30,vmin=-80,cmap = 'inferno' )
# axis[2].title.set_text("spectrogram")
# axis[2].grid(False)
# axis[2].set_xlabel("Time (sec)")
# axis[2].set_ylabel("Frequency (Hz dB)")
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
# axis[2].specgram(speech_utterance, Fs = speech_srate, mode = 'magnitude',
#              NFFT = 2**10, pad_to = 2**13, noverlap = (2**10)//1.5,
#              scale = 'dB',vmax=-30,vmin=-80,cmap = 'inferno' )
# axis[2].title.set_text("spectrogram")
# axis[2].grid(False)
# axis[2].set_xlabel("Time (sec)")
# axis[2].set_ylabel("Frequency (Hz dB)")
# axis[2].set_yscale('log')
# axis[2].set_ylim(100,speech_srate//2)
# fig_num+=1
# =============================================================================

plt.figure(fig_num)
Pxx, freqs, bins, im = plt.specgram(speech_utterance/max(abs(speech_utterance)), Fs = speech_srate, mode = 'magnitude',
             NFFT = 2**10, pad_to = 2**15, noverlap = (2**10)//1.05,
             scale = 'dB',vmax=-30,vmin=-80,cmap = 'inferno' )
plt.grid(False)
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
time = np.linspace(0,int(music_length/music_srate),music_length)

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
axis[2].set_ylabel("Frequency (Hz dB)")
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
axis[2].set_ylabel("Frequency (Hz dB)")
axis[2].set_yscale('log')
axis[2].set_ylim(100,music_srate//2)
fig_num+=1


# =============================================================================