
### Spectrogram: Take a .wav file and print out a spectrogram image of the file

import argparse
import importlib.util
import os
import sys
import wave
import numpy as np
from PIL import Image

## Get raw data out of .wav file, using pythons wave library
# The sample files are 1 channel, 16-bit samples, at 16,000 samples per
# second. Other 16-bit files work too: the sample rate is read from the
# file, and a stereo (or more) file is mixed down to one channel by
# averaging the channels. Anything that isn't 16-bit is rejected.
#
# Returns (samples, sample_rate), where samples is a numpy array of 16-bit
# integers (-32768 to 32767).
class WavFormatError(Exception):
    pass

def get_wave_data(fname):
    with wave.open(fname, 'rb') as f:
        n_channels = f.getnchannels()
        sample_width = f.getsampwidth()   # bytes per sample
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())
    if sample_width != 2:
        raise WavFormatError('%i-bit samples; only 16-bit is supported'
                             % (sample_width * 8))
    # Each sample is 2 bytes, little-endian, signed ('<i2')
    samples = np.frombuffer(frames, dtype='<i2')
    if n_channels > 1:
        # Samples are interleaved: L R L R ... -> one row per frame,
        # then average the channels of each frame
        samples = samples[:len(samples) // n_channels * n_channels]
        samples = samples.reshape(-1, n_channels).astype(np.int32)
        samples = np.round(samples.mean(axis=1))
    return samples.astype(np.int16), sample_rate

# Scale 16-bit samples to the range -1 to 1, the way audio programs such as
# Audacity show them: divide by 32768, so -32768 -> -1.0 and
# 32767 -> 0.99997. (Dividing positives by 32767 and negatives by 32768
# would bend the wave slightly where it crosses 0.)
def scale_raw_data(raw_data):
    return np.asarray(raw_data, dtype=float) / 32768


## Debugging helpers: look at the raw audio
# Used to check results against Audacity. Prints every sample between
# start_time and end_time (in seconds), e.g. 0.75 to 0.76 prints samples
# 12,000 to 12,159 of a 16,000 Hz file.
def print_data(samples, sample_rate, start_time, end_time):
    start_n = int(round(start_time * sample_rate))
    end_n = min(int(round(end_time * sample_rate)), len(samples))
    print('Samples %i to %i (%.4f s to %.4f s):'
          % (start_n, end_n - 1, start_time, end_time))
    print('%8s %10s %7s %9s' % ('sample', 'time (s)', 'raw', 'scaled'))
    for i in range(start_n, end_n):
        print('%8i %10.5f %7i %9.5f'
              % (i, i / sample_rate, samples[i], samples[i] / 32768))

# Draw the waveform: amplitude (-1 to 1) against time in seconds.
# matplotlib is only imported here, so the rest of the script doesn't need
# it. Set show=False to get the figure back without opening a window.
def plot_wave_data(samples, sample_rate, title='Waveform', show=True):
    import matplotlib.pyplot as plt
    y = scale_raw_data(samples)
    x = np.arange(len(y)) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, y, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, len(y) / sample_rate)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


## Create windows (2-D array: one row of samples per window)
# Cut the samples into overlapping windows of WINDOW_MS (25 ms), starting a
# new window every STEP_MS (10 ms). Window n is the slice
# samples[n*step : n*step + size], so at 16,000 Hz windows start at samples
# 0, 160, 320, ... and each holds 400 samples. A last piece shorter than a
# full window is dropped.
WINDOW_MS = 25
STEP_MS = 10

def window_size(sample_rate):
    return int(sample_rate / 1000 * WINDOW_MS)   # 400 at 16,000 Hz

def create_windows(samples, sample_rate):
    size = window_size(sample_rate)
    step = int(sample_rate / 1000 * STEP_MS)     # 160 at 16,000 Hz
    starts = range(0, len(samples) - size + 1, step)
    windows = np.empty((len(starts), size))
    for row, start in enumerate(starts):
        windows[row] = samples[start:start + size]
    return windows

## Taper each window with a Hamming window
# Cutting a sound off sharply at the edges of a window adds frequencies that
# aren't really there, so each window is faded in and out with
#     w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1)),  n = 0 .. N-1
# which is 0.08 at both ends and 1 in the middle (the same as
# numpy.hamming(N)). Returns new windows; the input isn't changed.
def hamm_windows(windows):
    N = windows.shape[1]
    n = np.arange(N)
    hamm = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    return windows * hamm

## Fourier transformation
# Input: the windows, one row for each 10 ms time step
# Output: a 2-D numpy array, one row per window and one column per
# frequency bin, holding the loudness of that bin in decibels.
#
# For a window of N samples, bin k is the frequency k * sample_rate / N
# (40 * k Hz for 400 samples at 16,000 Hz). Only bins 0 to N/2 are kept
# (0-200, so 0 to 8,000 Hz): for real audio the bins above N/2 are a mirror
# image of the ones below. numpy.fft.rfft() ("real FFT") returns exactly
# those bins.
#
# Adding a tiny number before the log keeps a window of complete silence
# (magnitude 0) from crashing it, since log10(0) is undefined. Silence
# comes out as 20 * log10(1e-10) = -200 dB.
SILENCE_FLOOR = 1e-10

def transform(windows):
    transformed_data = np.fft.rfft(windows, axis=1)
    # Magnitude of each complex number: sqrt(re^2 + im^2)
    magnitudes = np.abs(transformed_data)
    # Decibels of a magnitude are 20 * log10(|X|)
    # (the same as 10 * log10 of the power |X|^2)
    return 20 * np.log10(magnitudes + SILENCE_FLOOR)


## Draw the spectrogram image
# Each window (column) is 1 pixel of width and each frequency bin is 1 pixel
# of height. Frequency runs up the y-axis and intensity is shown by the
# darkness of the pixel: white means quiet and black the loudest.
#
# Only the top DYNAMIC_RANGE_DB decibels are drawn: the loudest value in
# the file is black, anything DYNAMIC_RANGE_DB or more below it is white,
# and the values in between are shades of grey. Without the limit a few
# very quiet values (or digital silence, at -200 dB) would stretch the
# scale and squeeze the speech into dark greys.
#
# data comes from transform(): one row per window, one column per bin.
DYNAMIC_RANGE_DB = 80

def create_spectrogram(data, output_fname):
    # Turn it on its side: rows = frequency bins, columns = windows (time)
    values = np.array(data, dtype=float).T
    # Image row 0 is the top of the picture, but low frequencies belong at
    # the bottom, so flip it: bin k goes to row (height - 1 - k)
    values = np.flipud(values)

    # Scale the decibel values to shades: high -> 0 (black),
    # low and below -> 255 (white)
    high = values.max()
    low = max(high - DYNAMIC_RANGE_DB, values.min())
    if high > low:
        values = np.clip(values, low, high)
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

def run(input_fname, output_fname, print_range=None, plot=False):
    raw, sample_rate = get_wave_data(input_fname)
    windows = create_windows(scale_raw_data(raw), sample_rate)
    if len(windows) == 0:
        sys.exit("Error: '%s' is too short to draw (%i samples; one window "
                 "needs %i)" % (input_fname, len(raw),
                                window_size(sample_rate)))
    windows = hamm_windows(windows)
    ft_windows = transform(windows)
    img = create_spectrogram(ft_windows, output_fname)
    print('Read %i samples (%.2f s at %i Hz) from %s.'
          % (len(raw), len(raw) / sample_rate, sample_rate, input_fname))
    print('Saved a %i x %i pixel spectrogram to %s.'
          % (img.width, img.height, output_fname))

    if print_range is not None:
        print()
        print_data(raw, sample_rate, *print_range)
    if plot:
        plot_wave_data(raw, sample_rate, title=os.path.basename(input_fname))

def interface():
    parser = argparse.ArgumentParser(
        prog='spectrogram.py',
        description='Draw a spectrogram image of a .wav file of speech\n'
                    '(16-bit audio, like the files in dr1/).',
        epilog='examples:\n'
               '  python3 spectrogram.py                        '
               'dr1/sa1.wav -> dr1/sa1_spectrogram.png\n'
               '  python3 spectrogram.py dr1/si836.wav          '
               'dr1/si836.wav -> dr1/si836_spectrogram.png\n'
               '  python3 spectrogram.py dr1/si836.wav -o a.png '
               'dr1/si836.wav -> a.png\n'
               '  python3 spectrogram.py --print 0.75 0.76      '
               'also print the samples from 0.75 s to 0.76 s\n'
               '  python3 spectrogram.py --plot                 '
               'also show the waveform in a window',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', metavar='WAVFILE',
                        help='.wav file to read (default: %s in this '
                             'script\'s folder)' % DEFAULT_INPUT)
    parser.add_argument('-o', '--output', metavar='IMAGEFILE',
                        help='where to save the image (default: the input '
                             'file\'s name with %s%s on the end, in the '
                             'same folder as the input)'
                             % (OUTPUT_SUFFIX, OUTPUT_EXT))
    parser.add_argument('--print', nargs=2, type=float, dest='print_range',
                        metavar=('START', 'END'),
                        help='also print every sample from START to END '
                             'seconds, to check against Audacity')
    parser.add_argument('--plot', action='store_true',
                        help='also show the waveform against time in a '
                             'window (needs matplotlib)')
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

    # Check the arguments before doing any work, so mistakes fail fast
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
    if args.print_range is not None:
        start, end = args.print_range
        if start < 0 or end <= start:
            parser.error('--print needs 0 <= START < END (got %g %g)'
                         % (start, end))
    if args.plot and importlib.util.find_spec('matplotlib') is None:
        parser.error('--plot needs matplotlib '
                     '(pip3 install -r requirements.txt)')

    try:
        run(input_fname, output_fname, args.print_range, args.plot)
    except (wave.Error, EOFError) as e:
        sys.exit("Error: '%s' is not a .wav file (%s)" % (input_fname, e))
    except WavFormatError as e:
        sys.exit("Error: can't read '%s' (%s)" % (input_fname, e))


if __name__ == '__main__':
    interface()
