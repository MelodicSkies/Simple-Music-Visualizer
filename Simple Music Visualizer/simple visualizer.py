import pyaudio
import sys
import time

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from scipy.fftpack import fft

class FFT():

    def calculate_fft(self, live_data, alpha):
        self.live_data = live_data
        self.data_splice()

        left_fft = abs(fft(self.audio_left))
        right_fft = abs(fft(self.audio_right))

        final_fft = [self.convert_to_dB(self.exponential_smooth(left_fft, alpha)), self.convert_to_dB(self.exponential_smooth(right_fft, alpha))]

        return final_fft

    #splits stereo data into left and right channels
    def data_splice(self):
        self.audio_left = self.live_data[0::2]
        self.audio_right = self.live_data[1::2]

    def convert_to_dB(self, data):
        for item in data:
            if item != 0:
                item = (10 * np.log10(item) ** 2) #accentuates peaks by converting to logarithmic scale

        return data

    #exponential smoothing to help reduce noise, smoothing factor is denoted as alpha (0 - 1)
    def exponential_smooth(self, fft_array, alpha):
        i = 0
        adjusted_fft = []

        while i < len(fft_array):
            if i == 0:
                adjusted_fft.append(abs(((1- alpha) * fft_array[i])))
            else:
                adjusted_fft.append(abs(alpha * adjusted_fft[i - 1] + ((1- alpha) * fft_array[i])))

            i += 1
    
        return adjusted_fft
        
class Music_Visualizer(object):
    def __init__(self):
        pg.setConfigOptions(antialias=True)
        self.app = QtGui.QApplication(sys.argv)

        #self.visualizer_plot = pg.plot()
        self.visualizer_plot = pg.PlotWidget() #use this to remove axis or use the line above to show axis
        self.CHUNK = 1024 #data chunking size for audio stream, use power of 2 for optimal fft speed, higher = better peak resolution, lower = better time resolution
        self.FS = 44100 #sampling rate, denoted by audio input/hardware (Hz)
        self.ALPHA = 0.3 #smoothing factor

        self.frequencies = (self.FS/self.CHUNK)*np.arange(0, self.CHUNK) #same for left and right channel data

        self.amplitudes_left = np.full((1, self.CHUNK), 0)[0] #sets to 0 when no audio input is detected
        self.amplitudes_right = np.full((1, self.CHUNK), 0)[0]

        self.visualizer_left = pg.BarGraphItem(x = self.frequencies, height = self.amplitudes_left, width = 1, brush = 'w')
        self.visualizer_right = pg.BarGraphItem(x = self.frequencies, height = self.amplitudes_right, width = 1, brush = 'w')

        self.visualizer_plot.addItem(self.visualizer_left)
        self.visualizer_plot.addItem(self.visualizer_right)

        #hide the x and y axis for more streamlined look
        self.visualizer_plot.getPlotItem().hideAxis('bottom')
        self.visualizer_plot.getPlotItem().hideAxis('left')
        self.visualizer_plot.show()

        self.visualizer_plot.setXRange(0, 5000) #can increase range to 17K Hz+, but most peaks of interest are < 10000 Hz in music, higher end mostly comprised of noise and harmonics
        self.visualizer_plot.setYRange(-50, 50) #adjust based on amplitude height

        audio_input = pyaudio.PyAudio()
        self.audio_stream = audio_input.open(format = pyaudio.paFloat32, channels = 2, rate = self.FS, input = True, output = True, frames_per_buffer = self.CHUNK) #initiates audio stream

        self.fft_analysis = FFT()
    
    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    #updates amplitude values for visualizer as audio data is streamed in real time
    def update(self):
        live_data = np.frombuffer(self.audio_stream.read(self.CHUNK, exception_on_overflow = False), dtype = np.float32)
        fft_data = self.fft_analysis.calculate_fft(live_data, self.ALPHA)
        self.visualizer_left.setOpts(height = fft_data[0])
        self.visualizer_right.setOpts(height = np.multiply(-1, fft_data[1])) #inverts right channel data on plot to create a "symmetrical" effect

    #keeps visualizer running in loop
    def real_time(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()


if __name__ == '__main__':

    visualizer = Music_Visualizer()
    visualizer.real_time()