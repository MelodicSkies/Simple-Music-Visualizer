
import pyaudio
import sounddevice

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import struct
from scipy.fftpack import fft
from scipy import signal

import matplotlib.pyplot as plt

def main():

    CHUNK = 1024
    Fs = 44100

    audio_input = pyaudio.PyAudio()
    audio_stream = audio_input.open(format = pyaudio.paFloat32, channels = 2, rate = Fs, input = True, output = True, frames_per_buffer = CHUNK)

    fig, (ax, ax2) = plt.subplots(2, figsize = (15, 8))
    x = np.arange(0, 2 * CHUNK, 2)
    #x_fft = np.arange(CHUNK)
    #x_fft = np.linspace(0, 44100, CHUNK)
    #x_fft = Fs//2*np.linspace(0,1,CHUNK//2+1)
    frequency_vector = Fs*np.arange(CHUNK)/CHUNK
    
    ax2.set_xlim(0, 2000)
    line, = ax.plot(x, np.random.rand(CHUNK), '-', lw = 2)
    
    line_fft, = ax2.plot(frequency_vector, np.random.rand(CHUNK), '-', lw = 2)
    
    plt.show(block = False)
    ax2.set_ylim(0, 50)

    while True:

        live_data = np.frombuffer(audio_stream.read(CHUNK, exception_on_overflow = False), dtype = np.float32)
       
        audio_left = live_data[0::2]
        audio_right = live_data[1::2]

        line.set_ydata(audio_left)

        y_fft = abs(fft(audio_left))
        
        #fft_data = (np.abs(fft(audio_left))[0:int(np.floor(CHUNK/2))])/CHUNK
        #fft_data[1:] = 2 * fft_data[1:]
        #line_fft.set_ydata(np.abs(y_fft[0:CHUNK]))
        #line_fft.set_ydata(10 * np.log10(np.abs(y_fft) ** 2))
        
        #welch = signal.welch(audio_left, Fs, nperseg = CHUNK)
        #print(welch)

        i = 0
        fftSmooth = []
        smoothing = 0
        
        while i < len(y_fft):
            if i == 0:
                fftSmooth.append(((1- smoothing) * y_fft[i]))
            else:
                fftSmooth.append(smoothing * fftSmooth[i - 1] + ((1- smoothing) * y_fft[i]))
            i += 1
        
        line_fft.set_ydata(10 * np.log10(np.abs(fftSmooth) ** 2))
        #print(fft_data)

        fig.canvas.draw()
        fig.canvas.flush_events()
        

if __name__ == '__main__':
    main()