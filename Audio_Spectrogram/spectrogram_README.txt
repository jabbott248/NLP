Audio Spectrogram
=================

What it does
------------
spectrogram.py is meant to read a .wav file of speech and draw it as a
spectrogram: an image where time runs left to right, frequency runs bottom
to top, and the darkness of each pixel shows how much energy the sound has
at that time and frequency. White means no energy and black means the most.

It isn't finished. The script reads the audio, cuts it into windows and
runs a Fourier transform on each one, but the last step, turning those
numbers into an image (create_spectrogram), was never written. Right now
it doesn't produce any image or output. See "Known issues".

How it works
------------
The steps below describe what the code is meant to do, and where it
does something different (see "Known issues" for details).

1. Read the audio (get_wave_data)
   The .wav files are mono (1 channel), 16-bit samples, 16,000 samples
   per second. Python's wave module reads them one sample at a time and
   each sample becomes an integer from -32768 to 32767. sa1.wav has 54,682
   samples, about 3.4 seconds of audio.

   scale_raw_data() can scale the samples to the range -1 to 1, but it is
   never called.

2. Cut the audio into windows (create_windows)
   A Fourier transform needs a short stretch of sound in which the
   frequencies don't change much, so the audio is cut into overlapping
   windows of 25 ms (400 samples). A new window is meant to start every
   10 ms (160 samples), so each column of the image is 10 ms of audio
   and sa1.wav should give 340 windows. The code actually starts one every
   15 ms (240 samples) and gets 227 (Known issue 3).

3. Taper each window (hamm_windows)
   Cutting a sound off sharply at the edges of a window adds frequencies
   that aren't really there. A Hamming window fades the start and end of
   each window down to about 8% of full volume:
       w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
   Each sample is multiplied by w[n]. The windows are changed in place.

4. Fourier transform (transform)
   numpy.fft.fft() turns each window of 400 samples into 400 complex
   numbers, one per frequency bin. Bin k is the frequency k * 16000 / 400
   = 40 * k Hz. Only bins 0-200 (0 to 8,000 Hz) are useful: for real
   audio, bins 201-399 are a mirror image of bins 1-199.

   Each complex number X is turned into a magnitude, |X| =
   sqrt(re^2 + im^2), and the magnitude is put on a log (decibel) scale,
   because the ear hears loudness roughly logarithmically. The code
   computes the magnitude a different way (Known issue 4) and keeps all
   400 bins in one long list (Known issue 5).

5. Draw the image (create_spectrogram)
   Not written yet. The plan in the comments: one pixel column per window
   (10 ms), one pixel row per frequency bin, and each pixel's darkness set
   from its decibel value, with white for the lowest and black for the
   highest. For sa1.wav that would be an image about 340 pixels wide and
   201 pixels high.

Files in this folder
--------------------
  spectrogram.py   the script
  image.py         a small image library given with the assignment (Brad
                   Miller, Luther College). It wraps Pillow and tkinter
                   so pixels can be set one at a time and the image
                   shown in a window or saved to a file.
  dr1/             three sample recordings with their transcripts. They
                   appear to be from the TIMIT speech corpus (dialect
                   region 1).
                     sa1.wav     3.4 s  "She had your dark suit in greasy
                                         wash water all year."
                     si1466.wav  4.2 s  "In simpler terms, it amounts to
                                         pointing the platform in the
                                         proper direction."
                     si836.wav   4.3 s  "No manufacturer has taken the
                                         initiative in pointing out the
                                         cost involved."
                   Each .txt holds the first and last sample number and
                   the sentence. The script doesn't use them.

Setup
-----
Follow the setup in the top-level README.md (install requirements.txt).
The script imports numpy, matplotlib, nltk and image.py, and image.py
uses Pillow and tkinter. (nltk is imported but never used; see Known issue
10.)

How to run
----------
Run from inside this folder:

      python3 spectrogram.py

It takes no arguments. It is supposed to open dr1/sa1.wav, but it looks
for /dr1/sa1.wav (at the root of the disk) and crashes:

      FileNotFoundError: [Errno 2] No such file or directory: '/dr1/sa1.wav'

With the path fixed, steps 1-4 run on sa1.wav in well under a second,
and then the results are thrown away. Nothing is printed, shown or
saved.

Known issues
------------
Found by running the script and each of its functions on the three
recordings in dr1/.

 1. No spectrogram is produced. create_spectrogram() is a stub that
    returns 1, and test_wave_read() (the function main() calls) never
    calls it. It computes the Fourier transform and then discards it.

    A note for when it's written: image rows count from the top (row 0 is
    the top of the picture), but a spectrogram has low frequencies at the
    bottom. If frequency bin k goes to row k, the image comes out upside
    down. Bin k should go to row (height - 1 - k).

 2. The input file is hard-coded as '/dr1/sa1.wav'. The leading slash
    makes it an absolute path, so the file is never found and the script
    crashes (see "How to run"). There is also no way to choose the input
    file or where the image is saved; there are no command-line arguments
    and no output file.

 3. Windows start every 15 ms instead of every 10 ms. create_windows()
    collects 400 samples and then steps back 160, so each window starts
    240 samples (15 ms) after the one before it. The first windows start
    at samples 0, 240, 480, 720, 960. A 10 ms step means stepping back 240
    so they start at 0, 160, 320, ... It gives 227 windows instead of 340
    for sa1.wav (280 instead of 419 for si1466.wav and 286 instead of 428
    for si836.wav), so the image would be about 2/3 as wide as it should
    be. The loop's test (i % sample_size == 0, with sample_size reset to
    i + 400) is also hard to follow and hard-codes the 400.

 4. The magnitude is computed wrongly. transform() uses
        sqrt(|re|) + sqrt(|im|)
    instead of
        sqrt(re^2 + im^2)
    That isn't the size of the number, and it changes with the phase:
    two values of the same size come out differently.

        X                 code gives   real |X|
        100 + 0j          10.0         100.0
        70.71 + 70.71j    16.8         100.0
        3 + 4j            3.7          5.0
        0.01 + 0j         0.1          0.01

    abs(X) or numpy.abs() gives the right answer.

 5. transform() keeps all 400 frequency bins and puts every window's bins
    into one flat list: 90,800 numbers for sa1.wav (227 windows x 400).
    Bins 201-399 are a mirror image of bins 1-199 and only make the image
    twice as tall with a flipped copy on top. The flat list also loses
    track of which window each value came from, so create_spectrogram()
    would have to cut it back up into windows. A list of windows (or a 2-D
    numpy array, windows x 201 bins) would be easier to draw.

 6. A window of complete silence would crash the script. log10(0) is
    undefined, and cmath.log10(0) raises "ValueError: math domain error".
    The three sample files don't hit it, but they do contain 983, 1,340
    and 1,243 samples that are exactly 0, so a recording with a stretch of
    digital silence would. Adding a tiny number before the log (e.g.
    log10(magnitude + 1e-10)) avoids it.

    Also, 10 * log10 is the decibel formula for power (|X|^2). For a
    magnitude it is 20 * log10(|X|), or equivalently 10 * log10(|X|^2).
    This only stretches every value by 2, so it won't change the picture
    once the values are scaled to pixel shades, but the numbers aren't
    really decibels.

 7. hamm_windows() has two small problems:
      - It imports cos from cmath (the complex-number module) instead of
        math, so every windowed sample becomes a complex number
        (-0.08005674905843613-0j). numpy handles that, but it's not what
        was meant.
      - The formula counts n from 1 to N and divides by N, instead of n
        from 0 to N-1 and dividing by N-1, so the window is shifted by one
        sample. That is a tiny effect. numpy.hamming(400) gives the
        standard window.
    The comment above create_windows() says "use a rectangular window",
    but a Hamming window is what's applied.

 8. scale_raw_data() is never called, so the Fourier transform runs on
    the raw integers (-7,789 to 9,626 for sa1.wav) instead of values from
    -1 to 1. That only shifts every decibel value by the same amount, so
    the picture would look the same, but either the function should be
    used or it should be removed.

 9. get_wave_data() assumes 16-bit mono audio and doesn't check. It reads
    one frame at a time and converts the whole frame to one integer, so a
    stereo or 8-bit file would give garbage with no error. It never closes
    the file. The sample rate (16,000) is also hard-coded in
    create_windows() and print_data() instead of being read from the file
    (wave's getframerate()). The three sample files are all 16-bit mono
    16 kHz, so this works for them.

10. Unused and odd imports: nltk (never used, and it means the script
    needs nltk installed for nothing), struct (never used), and cmath
    imported twice, once as "import cmath" and once for cos and log10
    (see 6 and 7).

11. The debugging helpers have mistakes and are never called:
      - print_data(): the comments say start and end times are in ms, but
        0.75 and 0.76 are in seconds (it prints samples 12,000-12,160).
      - plot_wave_data(): the comment says it takes every 1,000th sample,
        but it takes every 500th, starting from sample 1 instead of 0.
        The x-axis is the point number, not time, and the title is
        "Plotting 1-D array". It's the only thing matplotlib is used for.

12. image.py, the library given with the assignment:
      - It opens a hidden tkinter window as soon as it's imported, so the
        script needs a display even just to save an image. It works on a
        normal desktop but would fail over SSH or on a server.
      - ListImage (build an image from a list of lists) is broken:
        it calls Pixel() with one argument instead of three and raises
        "TypeError: Pixel.__init__() missing 2 required positional
        arguments: 'green' and 'blue'".
      - Its own demo (python3 image.py) needs lcastle.ppm, which isn't
        here.
    EmptyImage, setPixel and save all work, which is enough to draw a
    spectrogram. Pillow on its own (Image.fromarray) could do the same
    thing faster without image.py.

13. There are no tests.
