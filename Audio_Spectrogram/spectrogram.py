
### Spectrogram: Take a .wav file and print out a spectrogram image of the file

from cmath import cos
import argparse
import cmath
import math
import os
import sys
import nltk
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image

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



## Create windows (list of lists of samples)
# Cut the samples into overlapping windows of WINDOW_MS (25 ms), starting a
# new window every STEP_MS (10 ms). Window n is the slice
# samples[n*step : n*step + size], so windows start at samples
# 0, 160, 320, ... and each holds 400 samples. A last piece shorter than a
# full window is dropped.
WINDOW_MS = 25
STEP_MS = 10

def create_windows(samples, sample_rate=16000):
    # 16,000 samples per second, 1 second per 1,000 milliseconds
    size = int(sample_rate / 1000 * WINDOW_MS)   # 400 samples
    step = int(sample_rate / 1000 * STEP_MS)     # 160 samples
    windows = []
    for start in range(0, len(samples) - size + 1, step):
        windows.append(list(samples[start:start + size]))
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
# Input: a list of windows x[n], ..., x[n+s], one for each 10 ms time step
# Output: a 2-D numpy array, one row per window and one column per
# frequency bin, holding the loudness of that bin in decibels.
#
# For a window of N samples, bin k is the frequency k * sample_rate / N
# (40 * k Hz for 400 samples at 16,000 Hz). Only bins 0 to N/2 are kept
# (0-200, so 0 to 8,000 Hz): for real audio the bins above N/2 are a mirror
# image of the ones below.
#
# Adding a tiny number before the log keeps a window of complete silence
# (magnitude 0) from crashing it, since log10(0) is undefined.
SILENCE_FLOOR = 1e-10

def transform(windows):
    output = []
    for window in windows:
        transformed_data = np.fft.fft(window)
        n_bins = len(window) // 2 + 1                    # 201 for 400
        # Magnitude of each complex number: sqrt(re^2 + im^2)
        magnitudes = np.abs(transformed_data[:n_bins])
        # Decibels of a magnitude are 20 * log10(|X|)
        # (the same as 10 * log10 of the power |X|^2)
        log_scale = 20 * np.log10(magnitudes + SILENCE_FLOOR)
        output.append(log_scale)
    return np.array(output)


## Draw the spectrogram image
# Each window (column) is 1 pixel of width and each frequency bin is 1 pixel
# of height. Frequency runs up the y-axis and intensity is shown by the
# darkness of the pixel: white means the lowest value and black the highest.
#
# data comes from transform(): one row per window, one column per bin.
def create_spectrogram(data, output_fname):
    # Turn it on its side: rows = frequency bins, columns = windows (time)
    values = np.array(data, dtype=float).T
    # Image row 0 is the top of the picture, but low frequencies belong at
    # the bottom, so flip it: bin k goes to row (height - 1 - k)
    values = np.flipud(values)

    # Scale the decibel values to shades: lowest -> 255 (white),
    # highest -> 0 (black)
    low, high = values.min(), values.max()
    if high > low:
        shades = 255 * (high - values) / (high - low)
    else:
        shades = np.full(values.shape, 255.0)   # every value is the same
    pixels = np.round(shades).astype(np.uint8)

    img = Image.fromarray(pixels)   # 2-D array of 0-255 -> greyscale image
    try:
        img.save(output_fname)
    except (ValueError, OSError) as e:
        # e.g. no permission to write there
        sys.exit("Error: could not save '%s' (%s)" % (output_fname, e))
    return img


# Folder this script is in, so the default input is found no matter which
# folder the script is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join('dr1', 'sa1.wav')
OUTPUT_SUFFIX = '_spectrogram'
OUTPUT_EXT = '.png'

def default_output_name(input_fname):
    # dr1/sa1.wav -> dr1/sa1_spectrogram.png (saved next to the input)
    base, _ = os.path.splitext(input_fname)
    return base + OUTPUT_SUFFIX + OUTPUT_EXT

def run(input_fname, output_fname):
    data = get_wave_data(input_fname) # may need to scale this data -1 to 1
    windows = create_windows(data)
    hamm_windows(windows)
    if len(windows) == 0:
        sys.exit("Error: '%s' is too short to draw (%i samples; one window "
                 "needs 400)" % (input_fname, len(data)))
    ft_windows = transform(windows)
    img = create_spectrogram(ft_windows, output_fname)
    print('Read %i samples from %s.' % (len(data), input_fname))
    print('Saved a %i x %i pixel spectrogram to %s.'
          % (img.width, img.height, output_fname))

def interface():
    parser = argparse.ArgumentParser(
        prog='spectrogram.py',
        description='Draw a spectrogram image of a .wav file of speech\n'
                    '(16-bit mono audio, like the files in dr1/).',
        epilog='examples:\n'
               '  python3 spectrogram.py                        '
               'dr1/sa1.wav -> dr1/sa1_spectrogram.png\n'
               '  python3 spectrogram.py dr1/si836.wav          '
               'dr1/si836.wav -> dr1/si836_spectrogram.png\n'
               '  python3 spectrogram.py dr1/si836.wav -o a.png '
               'dr1/si836.wav -> a.png',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', metavar='WAVFILE',
                        help='.wav file to read (default: %s in this '
                             'script\'s folder)' % DEFAULT_INPUT)
    parser.add_argument('-o', '--output', metavar='IMAGEFILE',
                        help='where to save the image (default: the input '
                             'file\'s name with %s%s on the end, in the '
                             'same folder as the input)'
                             % (OUTPUT_SUFFIX, OUTPUT_EXT))
    args = parser.parse_args()

    input_fname = args.input
    if input_fname is None:
        # Show a short path if run from inside this folder or the repo,
        # and the full path otherwise
        input_fname = os.path.join(SCRIPT_DIR, DEFAULT_INPUT)
        cwd = os.getcwd()
        if os.path.commonpath([cwd, input_fname]) == cwd:
            input_fname = os.path.relpath(input_fname)
    output_fname = args.output
    if output_fname is None:
        output_fname = default_output_name(input_fname)

    # Check the files before doing any work, so mistakes fail fast
    if not os.path.isfile(input_fname):
        parser.error("input file '%s' not found" % input_fname)
    if os.path.abspath(output_fname) == os.path.abspath(input_fname):
        parser.error("output file '%s' is the same as the input file"
                     % output_fname)
    ext = os.path.splitext(output_fname)[1].lower()
    if ext not in Image.registered_extensions():
        parser.error("output file '%s' needs an image extension such as "
                     ".png or .jpg" % output_fname)
    output_dir = os.path.dirname(output_fname)
    if output_dir and not os.path.isdir(output_dir):
        parser.error("folder '%s' for the output file does not exist"
                     % output_dir)

    try:
        run(input_fname, output_fname)
    except (wave.Error, EOFError) as e:
        sys.exit("Error: '%s' is not a .wav file (%s)" % (input_fname, e))



if __name__ == '__main__':
    interface()
