Audio Spectrogram
=================

What it does
------------
spectrogram.py reads a .wav file of speech and draws it as a
spectrogram: an image where time runs left to right, frequency runs bottom
to top, and the darkness of each point shows how much energy the sound has
at that time and frequency. White means quiet and black means the loudest.

The image has a time axis in seconds along the bottom, a frequency axis in
Hz up the side, and a scale bar on the right that shows how many decibels
below the loudest point each shade of grey is. Each 10 ms window of audio
is one column and each 40 Hz band from 0 to 8,000 Hz is one row: 340 x 201
cells for sa1.wav. It can also
print the raw samples between two times, or plot the waveform, to check
the audio against a program like Audacity.

How it works
------------
1. Read the audio (get_wave_data)
   The .wav files in dr1/ are mono (1 channel), 16-bit samples, 16,000
   samples per second. Python's wave module reads the whole file and
   numpy turns each 2-byte sample into an integer from -32768 to 32767.
   sa1.wav has 54,682 samples, about 3.4 seconds of audio.

   The sample rate is read from the file, so other rates work too. A
   stereo file is mixed down to one channel by averaging the channels.
   Files that aren't 16-bit (8-bit, 24-bit, ...) are rejected with an
   error.

   scale_raw_data() then divides every sample by 32768 so they run from
   -1 to 1, the way Audacity shows them.

2. Cut the audio into windows (create_windows)
   A Fourier transform needs a short stretch of sound in which the
   frequencies don't change much, so the audio is cut into overlapping
   windows of 25 ms (400 samples at 16,000 Hz). A new window starts every
   10 ms (160 samples), at samples 0, 160, 320, ..., so each column of the
   image is 10 ms of audio. A last piece shorter than a full window is
   dropped. sa1.wav gives 340 windows, si1466.wav 419 and si836.wav 428.
   The window and step lengths are WINDOW_MS and STEP_MS above
   create_windows().

3. Taper each window (hamm_windows)
   Cutting a sound off sharply at the edges of a window adds frequencies
   that aren't really there. A Hamming window fades the start and end of
   each window down to 8% of full volume:
       w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1)),  n = 0 .. N-1
   Each sample is multiplied by w[n]. This is the same as
   numpy.hamming(N).

4. Fourier transform (transform)
   numpy.fft.rfft() turns each window of 400 samples into 201 complex
   numbers, one per frequency bin. Bin k is the frequency k * 16000 / 400
   = 40 * k Hz, so bins 0-200 cover 0 to 8,000 Hz. (A full FFT gives 400
   bins, but for real audio bins 201-399 are a mirror image of bins 1-199,
   and rfft() leaves them out.)

   Each complex number X is turned into a magnitude, |X| =
   sqrt(re^2 + im^2) (numpy.abs), and the magnitude is put on a log
   (decibel) scale, 20 * log10(|X|), because the ear hears loudness
   roughly logarithmically. A tiny number (1e-10) is added before the log
   so a window of complete silence, magnitude 0, doesn't crash it; it
   comes out as -200 dB instead.

   The result is a 2-D numpy array with one row per window and one column
   per bin: 340 x 201 for sa1.wav, with values from about -115 to 24 dB.

5. Shade levels (shade_levels)
   The decibel values themselves depend on how the samples were scaled,
   so on their own they don't mean much (sa1.wav runs from about -115 to
   24 dB). What the image shows instead is how far each point is below
   the loudest point in the file: the loudest point is 0 dB (black) and
   anything 80 dB or more below it is -80 dB (white), with shades of grey
   in between. For sa1.wav, 18% of the points are at -80 dB.

   Capping the scale at 80 dB (DYNAMIC_RANGE_DB) keeps a few very quiet
   values, or digital silence at -200 dB, from stretching the scale and
   squeezing the speech into dark greys. A file where every value is the
   same (e.g. all silence) comes out all white.

6. Draw the image (create_spectrogram)
   matplotlib draws the levels with imshow(): column i is the window that
   starts at 10 ms * i, drawn centred on the middle of that window, and
   row k is the band around k * 40 Hz. Image rows normally count from the
   top, but origin='lower' puts bin 0 (0 Hz) at the bottom, where low
   frequencies belong. The x-axis runs from 0 to the end of the last
   window (3.415 s for sa1.wav) and the y-axis from 0 to half the sample
   rate (8,000 Hz). The title is the input file's name, and the scale bar
   on the right goes from 0 dB (black) at the top to -80 dB (white) at
   the bottom.

   The picture is 10 x 4.5 inches at 150 dots per inch, so a .png is
   1500 x 675 pixels (FIGURE_SIZE and FIGURE_DPI). The format comes from
   the output file's extension: .png, .jpg, .pdf, .svg and the other
   formats matplotlib can save. It's drawn with matplotlib's Figure
   directly rather than pyplot, so no window opens and no display is
   needed.

Files in this folder
--------------------
  spectrogram.py         the script
  test_spectrogram.py    tests for the script (see "Running the tests")
  dr1/                   three sample recordings with their transcripts.
                         They appear to be from the TIMIT speech corpus
                         (dialect region 1).
                           sa1.wav     3.4 s  "She had your dark suit in
                                               greasy wash water all
                                               year."
                           si1466.wav  4.2 s  "In simpler terms, it
                                               amounts to pointing the
                                               platform in the proper
                                               direction."
                           si836.wav   4.3 s  "No manufacturer has taken
                                               the initiative in pointing
                                               out the cost involved."
                         Each .txt holds the first and last sample number
                         and the sentence. The script doesn't use them.

Setup
-----
Follow the setup in the top-level README.md (install requirements.txt).
The script needs numpy and matplotlib (which installs Pillow, used by the
tests to read the saved images).

How to run
----------
      python3 spectrogram.py [WAVFILE] [-o IMAGEFILE] [--print START END]
                             [--plot]

  WAVFILE            the .wav file to read. If left out, dr1/sa1.wav in
                     the script's folder is used, so the script can be run
                     from any folder.
  -o IMAGEFILE       where to save the image. If left out, it's the input
                     file's name with _spectrogram.png on the end, in the
                     same folder as the input. The extension picks the
                     format: .png, .jpg, .pdf, .svg, ...
  --print START END  also print every sample from START to END seconds,
                     to check against Audacity.
  --plot             also show the waveform (amplitude against time in
                     seconds) in a window. Close the window to finish.
  -h                 show the help message.

Examples:

      python3 spectrogram.py                      dr1/sa1.wav
                                                  -> dr1/sa1_spectrogram.png
      python3 spectrogram.py dr1/si836.wav        dr1/si836.wav
                                                  -> dr1/si836_spectrogram.png
      python3 spectrogram.py dr1/si836.wav -o a.png
                                                  dr1/si836.wav -> a.png

It takes well under a second for sa1.wav:

      $ python3 spectrogram.py
      Read 54682 samples (3.42 s at 16000 Hz) from dr1/sa1.wav.
      Saved a spectrogram (340 windows x 201 bins) to dr1/sa1_spectrogram.png.

The other two recordings have 419 (si1466.wav) and 428 (si836.wav)
windows; their images are the same 1500 x 675 pixels, with the time axis
running to 4.2 and 4.3 s.

With --print, the samples are listed after the image is saved, by sample
number, time, raw value and scaled value:

      $ python3 spectrogram.py --print 0.75 0.7505
      Read 54682 samples (3.42 s at 16000 Hz) from dr1/sa1.wav.
      Saved a spectrogram (340 windows x 201 bins) to dr1/sa1_spectrogram.png.

      Samples 12000 to 12007 (0.7500 s to 0.7505 s):
        sample   time (s)     raw    scaled
         12000    0.75000    4246   0.12958
         12001    0.75006    2425   0.07401
         12002    0.75013     470   0.01434
         12003    0.75019   -1971  -0.06015
         12004    0.75025   -3850  -0.11749
         12005    0.75031   -5150  -0.15717
         12006    0.75038   -5535  -0.16891
         12007    0.75044   -5102  -0.15570

A missing input file, an output folder that doesn't exist, an output
file without an image extension, an output file that is the same as the
input, or a --print range with END before START stops the script with a
short error before any work is done:

      $ python3 spectrogram.py nope.wav
      usage: spectrogram.py [-h] [-o IMAGEFILE] [--print START END] [--plot]
                            [WAVFILE]
      spectrogram.py: error: input file 'nope.wav' not found

A file that isn't a .wav file gives:

      $ python3 spectrogram.py dr1/sa1.txt
      Error: 'dr1/sa1.txt' is not a .wav file (file does not start with RIFF id)

an 8-bit .wav file gives "can't read ... (8-bit samples; only 16-bit is
supported)", and a file shorter than one 25 ms window stops with "is too
short to draw".

Running the tests
-----------------
From inside this folder:

      python3 -m unittest test_spectrogram

  (or run the file with python3). test_spectrogram.py has 49 tests and
  takes about 5 seconds.

  Most use small made-up signals whose right answer can be worked out by
  hand: a pure tone must land in the right frequency bin, a cosine and a
  sine of the same size must give the same magnitude, silence must give
  -200 dB, and the loudest point must be 0 dB with nothing below -80. The
  drawing tests check the axis labels and ranges, that each cell lines up
  with its window and frequency band, that the scale bar runs from white
  at -80 dB to black at 0, and, by reading the pixels of a saved image,
  that a loud low band really is dark at the bottom of the picture. The
  rest read the three recordings and run the script from the command
  line, including every error above. Files the tests write go in a
  temporary folder.

  To check that the tests catch real mistakes, bugs that were fixed in
  this script were put back one at a time: windows starting every 15 ms
  instead of 10, the magnitude computed as sqrt(|re|) + sqrt(|im|), a
  Hamming window shifted by one sample, samples scaled by 32767 instead
  of 32768, 8-bit files read without an error, print_data() assuming
  16,000 Hz, the image drawn upside down, the time axis shifted by half
  a window, the shade scale reversed, and no 80 dB limit on the shades.
  Each one made at least one test fail.
