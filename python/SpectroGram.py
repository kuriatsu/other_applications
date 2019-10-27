import wave
import sys
import numpy as np
# from scipy.fftpack import fft
import matplotlib.pyplot as plt

def getdata(filename):
    """ get binary data from .wave file
    args
    filename: the name of wave file

    return
    bin_data: binary data
    channel: channel number
    sample_bites_width: bites length of each points
    framerate: pointing(sampling) Hz, 44.1kHz etc.
    frame_num: the number of data points(samples)
    """

    # get data
    wave_file = wave.open(filename, 'r')
    bin_data = wave_file.readframes(-1) # binary data

    # basic data information
    nchannels = wave_file.getnchannels() # channel number
    sampwidth = wave_file.getsampwidth() # bites length of each points
    framerate = wave_file.getframerate() # pointing(sampling) Hz, 44.1kHz etc.
    nframes = wave_file.getnframes() # the number of data points(samples)

    wave_file.close()

    data = convert_data_to_ndarray(bin_data, sampwidth)

    return data, nchannels, sampwidth, framerate, nframes


def convert_data_to_ndarray(bin_data, sampwidth):
    """
    args
    bin_data: binary sound data
    sample_bites_width: 1 point(sample) is represented in ... 2->16bit 4->32bit

    return
    data: bites is converted to array data
    """

    if sampwidth == 2:
        data = np.frombuffer(bin_data, dtype='int16')

    elif sampwidth == 4:
        data = np.frombuffer(bin_data, dtype='int32')

    return data


# def plot_wav_graph(ndarray_data):


class kuri_spectrogram():

    def __init__(self, data, window_type) : #, nchannels, sampwidth, framerate, nframes):

        self.data_array = data[0]
        self.channel = data[1]
        self.sample_length = data[2]
        self.framerate = data[3]
        self.frame_length = data[4]

        self.window_length = 512
        self.shift_step = 2
        self.shift_length = self.window_length / self.shift_step

        # cut data and store it in matrix
        self.data_matrix= self.align_data()
        print(self.data_matrix.shape, len(self.data_matrix))

        # calcurate sound spectrogram
        self.ffted_data = self.fft(self.multiply_data_by_window(window_type))

        # plot spectrogram
        self.plot_spectrogram()


    def align_data(self):
        '''cut sound data and align into matlix

        '''
        # ----------------------------------------------------------------------
        # create matrix of data index
        # matrix column : whole one window is stored
        # matrix row : all windows are aligned
        # ----------------------------------------------------------------------

        # align_col_num = (the number of times we can shift, covering all data) - (surplus of window)
        align_col_num = int((1 + self.frame_length // self.shift_length) - (self.shift_step - 1))
        align_row_num = self.window_length

        # list shift number, then reshape and expand horizontary
        align_mat = np.arange(align_col_num).reshape(align_col_num, 1) * np.ones(align_row_num)
        # start index of window on data is listed in each columns
        align_mat = align_mat * self.shift_length
        # data index is applied to all element
        align_mat = align_mat + np.arange(self.window_length)
        # data type must be int64
        align_mat = align_mat.astype('int64')

        # ----------------------------------------------------------------------
        # insert matrix
        # ----------------------------------------------------------------------

        # padding data
        padding_data = np.pad(self.data_array, [0, align_col_num * align_row_num - self.frame_length], 'constant')

        # insert data to align matrix using "fancy index"
        return padding_data[align_mat]

        # print((1+self.frame_length//self.shift_length)-(self.shift_step-1))
        # print(self.frame_length)
        # print(np.arange(self.window_length).reshape(self.window_length, 1))


    def multiply_data_by_window(self, window_type):
        '''multiply sound data by window

        return: windowed data matrix
        '''
        # window array
        window = np.arange(self.window_length, dtype='float64')

        # select window type
        if window_type == 'hamming':
            window = 0.54 - 0.46 * np.cos(2 * np.pi * window / self.window_length)

        elif window_type == 'hanning':
            window = 0.5 - 0.5 * np.cos(2 * np.pi * window / self.window_length)

        elif window_type == 'blackman':
            window = 0.42 - 0.5 * np.cos(2 * np.pi * window / self.window_length) + 0.08 * np.cos(4 * np.pi * window / self.window_length)

        else: print('I do not know ',format(window_type))

        return self.data_matrix * window


    def fft(self, input):
        '''calcurate fft

        return: ffted data matrix
        '''

        # fft and normalization
        ffted_data = np.abs(np.fft.rfft(input)) * 2 / self.shift_length

        # Do not need to double DC component
        ffted_data[:, 0] = ffted_data[:, 0] / 2

        return ffted_data


    def plot_spectrogram(self):
        '''plot spectrogram


        '''
        sound_time = float(self.frame_length / self.framerate)

        plot_data = np.rot90(np.log(self.ffted_data), k=1)

        plt.subplot(211)
        plt.imshow(plot_data, extent=[0, sound_time, 0, self.framerate/2], aspect='auto')
        # plt.specgram(self.data_array, Fs=self.framerate)
        plt.colorbar()


    def ifft(self):

        iffted_data = np.fft.ifft(self.ffted_data) * self.window_length



if __name__ == "__main__":

    args = sys.argv
    filename = args[1]
    window_type = args[2]

    spectrogram = kuri_spectrogram(getdata(filename), window_type)
    # spectrogram.align_data()
    plt.show()
