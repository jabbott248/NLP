Audio Spectrogram
=================

What it does
------------
spectrogram.py reads a .wav file of speech and draws it as a
spectrogram: an image where time runs left to right, frequency runs bottom
to top, and the darkness of each pixel shows how much energy the sound has
at that time and frequency. White means no energy and black means the most.

For sa1.wav it saves a 340 x 201 pixel image: one column for every 10 ms
of audio and one row for every 40 Hz from 0 to 8,000 Hz.

How it works
------------
The steps below describe what the code does, and where it still does
something different from what was meant (see "Known issues").

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
   windows of 25 ms (400 samples). A new window starts every 10 ms (160
   samples), at samples 0, 160, 320, ..., so each column of the image is
   10 ms of audio. A last piece shorter than a full window is dropped.
   sa1.wav gives 340 windows, si1466.wav 419 and si836.wav 428. The
   window and step lengths are WINDOW_MS and STEP_MS at the top of
   create_windows().

3. Taper each window (hamm_windows)
   Cutting a sound off sharply at the edges of a window adds frequencies
   that aren't really there. A Hamming window fades the start and end of
   each window down to about 8% of full volume:
       w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
   Each sample is multiplied by w[n]. The windows are changed in place.

4. Fourier transform (transform)
   numpy.fft.fft() turns each window of 400 samples into 400 complex
   numbers, one per frequency bin. Bin k is the frequency k * 16000 / 400
   = 40 * k Hz. Only bins 0-200 (0 to 8,000 Hz) are kept: for real audio,
   bins 201-399 are a mirror image of bins 1-199.

   Each complex number X is turned into a magnitude, |X| =
   sqrt(re^2 + im^2) (numpy.abs), and the magnitude is put on a log
   (decibel) scale, 20 * log10(|X|), because the ear hears loudness
   roughly logarithmically. A tiny number (1e-10) is added before the log
   so a window of complete silence, magnitude 0, doesn't crash it; it
   comes out as -200 dB instead.

   The result is a 2-D numpy array with one row per window and one column
   per bin: 340 x 201 for sa1.wav. For sa1.wav the values run from about
   -19 to 114 dB.

5. Draw the image (create_spectrogram)
   The array from transform() is turned on its side, so each window is
   one pixel column (10 ms) and each
   frequency bin is one pixel row. Image rows count from the top (row 0
   is the top of the picture) but low frequencies belong at the bottom,
   so the rows are flipped: bin k goes to row (height - 1 - k).

   The decibel values are scaled to shades of grey, with the lowest value
   in the file white (255) and the highest black (0), and Pillow saves the
   result. The image format comes from the output file's extension (.png,
   .jpg, ...). Because the shades are scaled to the lowest and highest
   values in the file, a recording with digital silence in it comes out
   darker than it should (Known issue 14).

Files in this folder
--------------------
  spectrogram.py   the script
  image.py         a small image library given with the assignment (Brad
                   Miller, Luther College). It wraps Pillow and tkinter
                   so pixels can be set one at a time and the image
                   shown in a window or saved to a file. spectrogram.py
                   no longer uses it; it draws with Pillow directly.
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
The script imports numpy, matplotlib, nltk and Pillow. (nltk is
imported but never used; see Known issue 10.)

How to run
----------
      python3 spectrogram.py [WAVFILE] [-o IMAGEFILE]

  WAVFILE        the .wav file to read. If left out, dr1/sa1.wav in the
                 script's folder is used, so the script can be run from
                 any folder.
  -o IMAGEFILE   where to save the image. If left out, it's the input
                 file's name with _spectrogram.png on the end, in the
                 same folder as the input.
  -h             show the help message.

Examples:

      python3 spectrogram.py                      dr1/sa1.wav
                                                  -> dr1/sa1_spectrogram.png
      python3 spectrogram.py dr1/si836.wav        dr1/si836.wav
                                                  -> dr1/si836_spectrogram.png
      python3 spectrogram.py dr1/si836.wav -o a.png
                                                  dr1/si836.wav -> a.png

It takes about half a second for sa1.wav:

      $ python3 spectrogram.py
      Read 54682 samples from dr1/sa1.wav.
      Saved a 340 x 201 pixel spectrogram to dr1/sa1_spectrogram.png.

The other two recordings give 419 x 201 (si1466.wav) and 428 x 201
(si836.wav) images.

A missing input file, an output folder that doesn't exist, an output
file without an image extension, or an output file that is the same as
the input stops the script with a short error before any work is done:

      $ python3 spectrogram.py nope.wav
      usage: spectrogram.py [-h] [-o IMAGEFILE] [WAVFILE]
      spectrogram.py: error: input file 'nope.wav' not found

A file that isn't a .wav file gives:

      $ python3 spectrogram.py dr1/sa1.txt
      Error: 'dr1/sa1.txt' is not a .wav file (file does not start with RIFF id)

and a file shorter than one 25 ms window (400 samples) stops with
"is too short to draw".

Known issues
------------
Found by running the script and each of its functions on the three
recordings in dr1/.

 1. [FIXED] No spectrogram was produced. create_spectrogram() was a stub
    that returned 1, and nothing called it. It now draws the image with
    Pillow and saves it to the output file, with low frequencies at the
    bottom (bin k goes to row height - 1 - k).

 2. [FIXED] The input file was hard-coded as '/dr1/sa1.wav'. The leading
    slash made it an absolute path, so the file was never found and the
    script crashed. There was also no way to choose the input file or
    where the image is saved. The script now takes the input file and an
    -o output file on the command line (see "How to run"), defaults to
    dr1/sa1.wav found relative to the script, and checks the files
    before doing any work.

 3. [FIXED] Windows started every 15 ms instead of every 10 ms.
    create_windows() collected 400 samples and then stepped back 160, so
    each window started 240 samples after the one before it (0, 240, 480,
    ...). It gave 227 windows instead of 340 for sa1.wav, so the image was
    about 2/3 as wide as it should be. It now takes slices that start
    every 160 samples (0, 160, 320, ...) and gets 340, 419 and 428 windows
    for the three recordings.

 4. [FIXED] The magnitude was computed wrongly, as
    sqrt(|re|) + sqrt(|im|) instead of sqrt(re^2 + im^2). That isn't the
    size of the number, and it changed with the phase: 100 + 0j gave 10.0
    and 70.71 + 70.71j gave 16.8, though both have size 100. transform()
    now uses numpy.abs(), which gives 100.0 for both.

 5. [FIXED] transform() kept all 400 frequency bins and put every
    window's bins into one flat list. Bins 201-399 are a mirror image of
    bins 1-199, so the image was twice as tall with a flipped copy on
    top. It now keeps bins 0-200 and returns a 2-D numpy array (windows x
    201 bins), which create_spectrogram() draws directly.

 6. [FIXED] A window of complete silence crashed the script, because
    log10(0) is undefined and cmath.log10(0) raised "ValueError: math
    domain error". transform() now adds 1e-10 to each magnitude before the
    log. A test file with 1 second of digital silence before sa1.wav
    crashed the old code and now draws (but see issue 14). It also used
    10 * log10, the decibel formula for power (|X|^2), on a magnitude; it
    now uses 20 * log10(|X|), so the values are real decibels.

 7. hamm_windows() has two small problems:
      - It imports cos from cmath (the complex-number module) instead of
        math, so every windowed sample becomes a complex number
        (-0.08005674905843613-0j). numpy handles that, but it's not what
        was meant.
      - The formula counts n from 1 to N and divides by N, instead of n
        from 0 to N-1 and dividing by N-1, so the window is shifted by one
        sample. That is a tiny effect. numpy.hamming(400) gives the
        standard window.

 8. scale_raw_data() is never called, so the Fourier transform runs on
    the raw integers (-7,789 to 9,626 for sa1.wav) instead of values from
    -1 to 1. That only shifts every decibel value by the same amount, so
    the picture would look the same, but either the function should be
    used or it should be removed.

 9. get_wave_data() assumes 16-bit mono audio and doesn't check. It reads
    one frame at a time and converts the whole frame to one integer, so a
    stereo or 8-bit file would give garbage with no error. It never closes
    the file. The sample rate (16,000) is also assumed, as the default in
    create_windows() and hard-coded in print_data(), instead of being read
    from the file (wave's getframerate()). The three sample files are all
    16-bit mono 16 kHz, so this works for them.

10. Unused and odd imports: nltk (never used, and it means the script
    needs nltk installed for nothing), struct (never used), and cmath
    imported twice, once as "import cmath" (never used) and once for cos
    (see 7).

11. The debugging helpers have mistakes and are never called:
      - print_data(): the comments say start and end times are in ms, but
        0.75 and 0.76 are in seconds (it prints samples 12,000-12,160).
      - plot_wave_data(): the comment says it takes every 1,000th sample,
        but it takes every 500th, starting from sample 1 instead of 0.
        The x-axis is the point number, not time, and the title is
        "Plotting 1-D array". It's the only thing matplotlib is used for.

12. image.py, the library given with the assignment. spectrogram.py no
    longer imports it (it draws with Pillow, see issue 1), so these only
    matter if it is used again:
      - It opens a hidden tkinter window as soon as it's imported, so a
        script using it needs a display even just to save an image. It
        works on a normal desktop but would fail over SSH or on a server.
      - ListImage (build an image from a list of lists) is broken:
        it calls Pixel() with one argument instead of three and raises
        "TypeError: Pixel.__init__() missing 2 required positional
        arguments: 'green' and 'blue'".
      - Its own demo (python3 image.py) needs lcastle.ppm, which isn't
        here.
    EmptyImage, setPixel and save all work. It could be deleted from the
    folder now, or kept as part of the original assignment.

13. There are no tests.

14. Digital silence makes the whole image too dark. The shades are scaled
    from the lowest value in the file (white) to the highest (black).
    A silent window comes out at -200 dB (from the 1e-10 in issue 6), so
    with 1 second of silence added before sa1.wav the values span 314 dB
    instead of 133, and the speech is squeezed into the dark end: its
    average shade drops from 120 to 51 (0 = black, 255 = white). The three
    sample files have no silent windows, so they aren't affected. A common
    fix is to draw only the top 80 dB or so: anything quieter than
    (loudest - 80 dB) is drawn white. That would also give the sample
    files a bit more contrast, since their quietest 1% of values sit
    between about -19 and 15 dB and spread out the grey scale.
