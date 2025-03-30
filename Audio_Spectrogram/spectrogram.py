
### Spectrogram: Take a .wav file and print out a spectrogram image of the file

from cmath import cos, log10
import cmath
import math
import nltk
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt

# image library given to us
import image

## Get raw data out of .wav file, using pythons wave library
# .wav formats are 1 channel, 16-bit samples, and a 16,000 Hz sample rate.
# get data using the wave library
def get_wave_data(fname):
    f = wave.open(fname, 'rb')
    ret_data = []
    i = 0
    while i<f.getnframes():
        byte = f.readframes(1)
        byte = int.from_bytes(byte, byteorder='little', signed=True)
        ret_data.append(byte)
        i+=1
        # concatenate two bytes together into 1
    return ret_data

# int16 min: -32768, max: 32767
# turns bytes to floats, needs to read to bytes at once as samples are 16bit values
def scale_raw_data(raw_data):
    scaled_data = []
    for sample in raw_data:
        if sample >= 0:
            scaled_sample = sample/32767
        else:
            scaled_sample = sample/32768
        scaled_data.append(scaled_sample)
    return scaled_data

#used to check results w/ audacity
def print_data(data):
    i = 0
    start_time = 0.75 #ms
    end_time = 0.76 #ms
    # 16,000 samples/s, 1 second/1,000 ms, 
    start_smpl_n = start_time*16000 # start time ms, 1s/1,000ms, 16,000 smpls/s
    end_smpl_n = end_time*16000
    print('start: %i, end: %i'%(start_smpl_n,end_smpl_n))
    i = int(start_smpl_n)
    while(i<end_smpl_n):
        sample = data[i]
        print(sample)
        #print('sample #%i: %f'%(i,sample))
        i+=1

# X-Axis: 
def plot_wave_data(data):
    # get every 1,000th data point
    y = []
    i = 1
    while(i<len(data)):
        y.append(data[i])
        i+=500
    x = [x for x in range(len(y))]
    plt.title("Plotting 1-D array")
    plt.xlabel("Range")
    plt.ylabel("Wave Data")
    plt.plot(x, y, color = "red", marker = "o", label = "Array elements")
    plt.legend()
    plt.show()



## Create windows, (list of lists of samples)
# use a rectangular window with size s = 25 ms, creating a new data window each 10 ms
# Generate a window x[n], ..., x[n+s] based on this window size by taking a slice of the array of samples.
def create_windows(samples):
    sample_rate = 16000
    # 16,000 samples per second, 1 second per 1,000 milliseconds, 25 ms per window
    sample_size = sample_rate/1000*25           #400
    step_back_size = int(sample_rate/1000*10)   #160
    windows = []
    window = []
    i = 0
    while(i < len(samples)):
        window.append(samples[i])
        i+=1
        if (i%sample_size == 0):
            windows.append(window)
            window = []
            i = i-step_back_size
            sample_size = i+400
    return windows

def hamm_windows(windows):
    for window in windows:
        l = len(window)
        count = 1
        for i in range(len(window)):
            hamm_val = 0.54-0.46*cos(2*math.pi*count/l)
            window[i] = window[i] * hamm_val
            count+=1
    #return windows

## Fourier transformation
# Input: for each 10 ms time step, you will have a window x[n], ..., x[n+s]
# Output: an array of complex numbers
def transform(windows):
    output = []
    for window in windows:
        transformed_data = np.fft.fft(window)
        for data in transformed_data:
            # Convert complex numbers to magnitudes
            real = abs(data.real)
            imag = abs(data.imag)
            magnitude = math.sqrt(real) + math.sqrt(imag)
            # Convert from that to log scale with the formula 10 * log10
            log_scale = 10 * log10(magnitude)
            log_scale = log_scale.real
            output.append(log_scale)
    return output


## Plot the data points :)
# Each 10ms window is 1 pixel of width - each freq is 1 pixel
# frequency along the y-axis and intensity indicated by the darkness of the pixel
# white means 0 intensity and black means maximum
def create_spectrogram(data):
    ## TODO: finish this final part
    return 1


# load an all white image that is x pixels long, and y pixels high. x? y?
# loop through log_scale data and manipulate pixel by pixel

## to test, Write a visual output that will plot the sample value at each frame and create a graph of the sound over time. 
def test_wave_read():
    data = get_wave_data('/dr1/sa1.wav') # may need to scale this data -1 to 1
    windows = create_windows(data)
    hamm_windows(windows)
    ft_windows = transform(windows)


def main():
   test_wave_read()

if __name__ == '__main__':
   main()