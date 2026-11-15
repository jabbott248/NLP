# Tests for spectrogram.py
#
# Run from this folder with either of:
#     python3 -m unittest test_spectrogram
#     python3 test_spectrogram.py
#
# The unit tests mostly use small made-up signals (silence, a pure tone,
# windows of ones) so the right answer can be worked out by hand. The tests
# at the bottom run on the real recordings in dr1/ and run the script from
# the command line. Every file the tests write goes in a temporary folder.

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
import wave

# Draw plots off-screen, so the tests don't open windows
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from PIL import Image

import spectrogram as S

HERE = os.path.dirname(os.path.abspath(__file__))
SA1 = os.path.join(HERE, 'dr1', 'sa1.wav')
RECORDINGS = {        # file: (samples, windows)
    'sa1.wav': (54682, 340),
    'si1466.wav': (67380, 419),
    'si836.wav': (68813, 428),
}


def write_wav(fname, samples, sample_rate=16000, n_channels=1, width=2):
    """Write a .wav file from a list or array of integer samples."""
    dtype = {1: np.uint8, 2: '<i2'}[width]
    with wave.open(fname, 'wb') as f:
        f.setnchannels(n_channels)
        f.setsampwidth(width)
        f.setframerate(sample_rate)
        f.writeframes(np.asarray(samples).astype(dtype).tobytes())


def tone(freq, n_samples, sample_rate=16000, amplitude=0.5):
    """A pure sine wave, scaled to -1..1."""
    t = np.arange(n_samples) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestGetWaveData(TempDirTestCase):
    def test_reads_sample_file(self):
        samples, rate = S.get_wave_data(SA1)
        self.assertEqual(rate, 16000)
        self.assertEqual(len(samples), 54682)
        self.assertEqual(samples.dtype, np.int16)
        # Values checked against the --print output / Audacity
        self.assertEqual(list(samples[12000:12003]), [4246, 2425, 470])

    def test_reads_negative_samples(self):
        self.assertEqual(
            list(S.get_wave_data(self._wav([-32768, -1, 0, 1, 32767]))[0]),
            [-32768, -1, 0, 1, 32767])

    def test_reads_sample_rate_from_file(self):
        fname = self._wav([0] * 10, sample_rate=8000)
        self.assertEqual(S.get_wave_data(fname)[1], 8000)

    def test_stereo_is_mixed_down(self):
        # Frames (L, R): (100, 300) (-50, -150) (7, 8)
        fname = self.path('stereo.wav')
        write_wav(fname, [100, 300, -50, -150, 7, 8], n_channels=2)
        samples, _ = S.get_wave_data(fname)
        self.assertEqual(list(samples), [200, -100, 8])   # 7.5 rounds to 8

    def test_8_bit_is_rejected(self):
        fname = self.path('8bit.wav')
        write_wav(fname, [128] * 10, width=1)
        with self.assertRaisesRegex(S.WavFormatError, '8-bit'):
            S.get_wave_data(fname)

    def test_not_a_wav_file(self):
        fname = self.path('notes.txt')
        with open(fname, 'w') as f:
            f.write('not audio')
        with self.assertRaises(wave.Error):
            S.get_wave_data(fname)

    def _wav(self, samples, **kwargs):
        fname = self.path('test.wav')
        write_wav(fname, samples, **kwargs)
        return fname


class TestScaleRawData(unittest.TestCase):
    def test_scale(self):
        scaled = S.scale_raw_data([-32768, -16384, 0, 16384, 32767])
        np.testing.assert_allclose(
            scaled, [-1.0, -0.5, 0.0, 0.5, 32767 / 32768])

    def test_stays_in_range(self):
        scaled = S.scale_raw_data(np.array([-32768, 32767], dtype=np.int16))
        self.assertGreaterEqual(scaled.min(), -1.0)
        self.assertLess(scaled.max(), 1.0)


class TestCreateWindows(unittest.TestCase):
    def test_windows_start_every_10_ms(self):
        samples = np.arange(1000)
        windows = S.create_windows(samples, 16000)
        # 25 ms = 400 samples per window, a new one every 160 samples:
        # starts 0, 160, 320, 480 (640 + 400 > 1000, so it stops there)
        self.assertEqual(windows.shape, (4, 400))
        self.assertEqual(list(windows[:, 0]), [0, 160, 320, 480])
        np.testing.assert_array_equal(windows[1], samples[160:560])

    def test_exactly_one_window(self):
        self.assertEqual(S.create_windows(np.zeros(400), 16000).shape,
                         (1, 400))

    def test_too_short_for_a_window(self):
        self.assertEqual(S.create_windows(np.zeros(399), 16000).shape,
                         (0, 400))

    def test_other_sample_rate(self):
        # 8,000 Hz: 25 ms = 200 samples, 10 ms = 80 samples
        windows = S.create_windows(np.arange(400), 8000)
        self.assertEqual(windows.shape, (3, 200))
        self.assertEqual(list(windows[:, 0]), [0, 80, 160])

    def test_window_count_formula(self):
        for n in [400, 559, 560, 561, 54682]:
            self.assertEqual(len(S.create_windows(np.zeros(n), 16000)),
                             (n - 400) // 160 + 1)


class TestHammWindows(unittest.TestCase):
    def test_matches_numpy_hamming(self):
        windowed = S.hamm_windows(np.ones((2, 400)))
        np.testing.assert_allclose(windowed[0], np.hamming(400))
        np.testing.assert_allclose(windowed[1], np.hamming(400))

    def test_shape_of_the_taper(self):
        w = S.hamm_windows(np.ones((1, 401)))[0]
        self.assertAlmostEqual(w[0], 0.08)
        self.assertAlmostEqual(w[-1], 0.08)
        self.assertAlmostEqual(w[200], 1.0)   # middle of 401 samples

    def test_result_is_real(self):
        self.assertFalse(np.iscomplexobj(S.hamm_windows(np.ones((1, 400)))))

    def test_input_not_changed(self):
        windows = np.ones((1, 400))
        S.hamm_windows(windows)
        np.testing.assert_array_equal(windows, np.ones((1, 400)))


class TestTransform(unittest.TestCase):
    def test_keeps_bins_0_to_n_over_2(self):
        self.assertEqual(S.transform(np.ones((3, 400))).shape, (3, 201))

    def test_constant_window(self):
        # The FFT of 400 ones is 400 in bin 0 and 0 everywhere else
        db = S.transform(np.ones((1, 400)))[0]
        self.assertAlmostEqual(db[0], 20 * np.log10(400))
        self.assertTrue(np.all(db[1:] < -100))

    def test_magnitude_ignores_phase(self):
        # A cosine and a sine of the same size give the same magnitude
        t = np.arange(400) / 16000
        cos_db = S.transform(np.array([np.cos(2 * np.pi * 1000 * t)]))
        sin_db = S.transform(np.array([np.sin(2 * np.pi * 1000 * t)]))
        # 1,000 Hz is bin 1000 / 40 = 25, with magnitude N/2 = 200
        self.assertAlmostEqual(cos_db[0, 25], 20 * np.log10(200))
        self.assertAlmostEqual(sin_db[0, 25], 20 * np.log10(200))

    def test_tone_lands_in_the_right_bin(self):
        for freq in [440, 1000, 3000, 7000]:
            windows = S.hamm_windows(
                S.create_windows(tone(freq, 4000), 16000))
            db = S.transform(windows)
            self.assertEqual(np.argmax(db[0]), round(freq / 40), freq)

    def test_silence_does_not_crash(self):
        db = S.transform(np.zeros((2, 400)))
        np.testing.assert_allclose(db, -200.0)   # 20 * log10(1e-10)

    def test_matches_numpy_reference(self):
        windows = np.random.default_rng(0).normal(size=(5, 400))
        expected = 20 * np.log10(np.abs(np.fft.fft(windows))[:, :201]
                                 + 1e-10)
        np.testing.assert_allclose(S.transform(windows), expected)


class TestShadeLevels(unittest.TestCase):
    def test_loudest_is_0_db(self):
        levels = S.shade_levels(np.array([[100.0, 60.0, 30.0]]))
        np.testing.assert_allclose(levels, [[0.0, -40.0, -70.0]])

    def test_quieter_than_80_db_is_capped(self):
        levels = S.shade_levels(np.array([[100.0, 10.0, -200.0]]))
        np.testing.assert_allclose(levels, [[0.0, -80.0, -80.0]])

    def test_shape_unchanged(self):
        self.assertEqual(S.shade_levels(np.zeros((4, 201)) + np.arange(201))
                         .shape, (4, 201))

    def test_all_the_same_is_white(self):
        np.testing.assert_allclose(S.shade_levels(np.full((3, 5), -200.0)),
                                   -80.0)

    def test_silence_does_not_change_other_levels(self):
        rng = np.random.default_rng(1)
        speech = rng.uniform(20, 110, size=(10, 201))
        with_silence = np.vstack([np.full((5, 201), -200.0), speech])
        a = S.shade_levels(speech)
        b = S.shade_levels(with_silence)
        np.testing.assert_allclose(b[5:], a)
        np.testing.assert_allclose(b[:5], -80.0)


class TestCreateSpectrogram(TempDirTestCase):
    def draw(self, data, sample_rate=16000, name='out.png', title=None):
        fig = S.create_spectrogram(data, sample_rate, self.path(name),
                                   title=title)
        main_ax, bar_ax = fig.axes
        return fig, main_ax, bar_ax

    def test_saves_a_png(self):
        self.draw(np.zeros((5, 201)))
        with Image.open(self.path('out.png')) as saved:
            self.assertEqual(saved.format, 'PNG')
            self.assertEqual(saved.size, (1500, 675))   # 10 x 4.5 in, 150 dpi

    def test_other_formats(self):
        for name in ['out.pdf', 'out.svg', 'out.jpg']:
            self.draw(np.zeros((5, 201)), name=name)
            self.assertGreater(os.path.getsize(self.path(name)), 0, name)

    def test_axes_labels_and_title(self):
        _, ax, bar = self.draw(np.zeros((5, 201)), title='sa1.wav')
        self.assertEqual(ax.get_xlabel(), 'Time (s)')
        self.assertEqual(ax.get_ylabel(), 'Frequency (Hz)')
        self.assertEqual(ax.get_title(), 'sa1.wav')
        self.assertEqual(bar.get_ylabel(), 'Level (dB below the loudest point)')

    def test_axes_ranges(self):
        # 340 windows: the last one starts at 3.39 s and ends at 3.415 s
        _, ax, _ = self.draw(np.zeros((340, 201)))
        self.assertAlmostEqual(ax.get_xlim()[0], 0)
        self.assertAlmostEqual(ax.get_xlim()[1], 3.415)
        self.assertEqual(ax.get_ylim(), (0, 8000))

    def test_frequency_axis_follows_sample_rate(self):
        # 8,000 Hz: 200-sample windows -> 101 bins, 0 to 4,000 Hz
        _, ax, _ = self.draw(np.zeros((5, 101)), sample_rate=8000)
        self.assertEqual(ax.get_ylim(), (0, 4000))

    def test_cells_line_up_with_windows_and_bins(self):
        _, ax, _ = self.draw(np.zeros((340, 201)))
        left, right, bottom, top = ax.images[0].get_extent()
        # Window 0 is centred at 12.5 ms and each column is 10 ms wide
        self.assertAlmostEqual((left + 0.005), 0.0125)
        self.assertAlmostEqual(right - left, 3.40)
        # Bin 0 is centred at 0 Hz and each row is 40 Hz tall
        self.assertAlmostEqual(bottom, -20)
        self.assertAlmostEqual(top, 8020)

    def test_low_frequencies_at_the_bottom(self):
        data = np.full((3, 10), -200.0)
        data[:, 0] = 50        # bin 0 loud in every window, silence elsewhere
        _, ax, _ = self.draw(data)
        image = ax.images[0]
        self.assertEqual(image.origin, 'lower')   # array row 0 at the bottom
        shown = image.get_array()
        self.assertEqual(shown.shape, (10, 3))    # rows = bins, cols = time
        np.testing.assert_allclose(shown[0], 0.0)      # bin 0: loudest
        np.testing.assert_allclose(shown[1:], -80.0)

    def test_time_runs_left_to_right(self):
        data = np.full((4, 10), -200.0)
        data[0, :] = 50        # first window loud, silence elsewhere
        _, ax, _ = self.draw(data)
        shown = ax.images[0].get_array()
        np.testing.assert_allclose(shown[:, 0], 0.0)
        np.testing.assert_allclose(shown[:, 1:], -80.0)

    def test_white_is_quiet_and_black_is_loud(self):
        _, ax, bar = self.draw(np.zeros((5, 201)))
        image = ax.images[0]
        self.assertEqual(image.get_clim(), (-80, 0))
        white = image.cmap(image.norm(-80))
        black = image.cmap(image.norm(0))
        np.testing.assert_allclose(white[:3], [1, 1, 1])
        np.testing.assert_allclose(black[:3], [0, 0, 0])
        self.assertEqual(bar.get_ylim(), (-80, 0))

    def test_picture_has_dark_band_at_the_bottom(self):
        # Look at the saved pixels: a loud 0-400 Hz band and silence above
        data = np.full((50, 201), -200.0)
        data[:, :10] = 50
        _, ax, _ = self.draw(data)
        with Image.open(self.path('out.png')) as saved:
            pixels = np.array(saved.convert('L'))
        # Middle column of the plot area, near the bottom and near the top
        x0, y0, x1, y1 = ax.get_window_extent().extents
        height = pixels.shape[0]
        col = int((x0 + x1) / 2)
        near_bottom = pixels[height - int(y0) - 5, col]
        near_top = pixels[height - int(y1) + 5, col]
        self.assertLess(near_bottom, 50)       # dark
        self.assertGreater(near_top, 200)      # white


class TestDebugHelpers(unittest.TestCase):
    def test_print_data_range(self):
        samples, rate = S.get_wave_data(SA1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            S.print_data(samples, rate, 0.75, 0.76)
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], 'Samples 12000 to 12159 '
                                   '(0.7500 s to 0.7600 s):')
        rows = lines[2:]
        self.assertEqual(len(rows), 160)
        self.assertEqual(rows[0].split(), ['12000', '0.75000', '4246',
                                           '0.12958'])
        self.assertEqual(rows[-1].split()[0], '12159')

    def test_print_data_stops_at_end_of_file(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            S.print_data(np.arange(10), 10, 0.5, 5.0)
        self.assertEqual(len(out.getvalue().splitlines()), 2 + 5)

    def test_plot_wave_data(self):
        import matplotlib.pyplot as plt
        samples, rate = S.get_wave_data(SA1)
        fig = S.plot_wave_data(samples, rate, title='sa1.wav', show=False)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'Time (s)')
        self.assertEqual(ax.get_title(), 'sa1.wav')
        x, y = ax.lines[0].get_data()
        self.assertEqual(len(x), len(samples))         # every sample
        self.assertAlmostEqual(x[-1], (len(samples) - 1) / rate)
        self.assertLessEqual(np.abs(y).max(), 1.0)     # scaled to -1..1
        plt.close(fig)


class TestOutputName(unittest.TestCase):
    def test_default_output_name(self):
        self.assertEqual(S.default_output_name(os.path.join('dr1', 'sa1.wav')),
                         os.path.join('dr1', 'sa1_spectrogram.png'))
        self.assertEqual(S.default_output_name('talk.WAV'),
                         'talk_spectrogram.png')


# --- End to end: the real recordings and the command line ---

class TestRecordings(unittest.TestCase):
    def test_sample_and_window_counts(self):
        for name, (n_samples, n_windows) in RECORDINGS.items():
            samples, rate = S.get_wave_data(os.path.join(HERE, 'dr1', name))
            self.assertEqual(len(samples), n_samples, name)
            self.assertEqual(len(S.create_windows(samples, rate)),
                             n_windows, name)


class TestCommandLine(TempDirTestCase):
    def run_script(self, *args, cwd=HERE):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, 'spectrogram.py'), *args],
            capture_output=True, text=True, cwd=cwd,
            env={**os.environ, 'MPLBACKEND': 'Agg'})

    def test_draws_each_recording(self):
        for name, (n_samples, n_windows) in RECORDINGS.items():
            out = self.path(name + '.png')
            result = self.run_script(os.path.join('dr1', name), '-o', out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Read %i samples' % n_samples, result.stdout)
            self.assertIn('(%i windows x 201 bins)' % n_windows,
                          result.stdout)
            with Image.open(out) as img:
                self.assertEqual(img.format, 'PNG')

    def test_default_input_from_another_folder(self):
        # Run from the temp folder: the default dr1/sa1.wav is still found
        out = self.path('sa1.png')
        result = self.run_script('-o', out, cwd=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('sa1.wav', result.stdout)
        self.assertTrue(os.path.isfile(out))

    def test_default_output_next_to_input(self):
        fname = self.path('tone.wav')
        write_wav(fname, tone(1000, 8000) * 32767)
        result = self.run_script(fname)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.isfile(self.path('tone_spectrogram.png')))

    def test_print_flag(self):
        result = self.run_script('-o', self.path('a.png'),
                                 '--print', '0.75', '0.7505')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Samples 12000 to 12007', result.stdout)

    def test_plot_flag(self):
        result = self.run_script('-o', self.path('a.png'), '--plot')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_errors(self):
        notes = self.path('notes.txt')
        with open(notes, 'w') as f:
            f.write('not audio')
        eight = self.path('8bit.wav')
        write_wav(eight, [128] * 1000, width=1)
        short = self.path('short.wav')
        write_wav(short, [0] * 100)
        cases = [
            (['nope.wav'], 2, "input file 'nope.wav' not found"),
            (['-o', 'out'], 2, 'needs an image extension'),
            (['-o', os.path.join('nodir', 'a.png')], 2, 'does not exist'),
            (['dr1/sa1.wav', '-o', 'dr1/sa1.wav'], 2, 'same as the input'),
            (['--print', '0.8', '0.7', '-o', self.path('a.png')], 2,
             '--print needs'),
            ([notes, '-o', self.path('a.png')], 1, 'is not a .wav file'),
            ([eight, '-o', self.path('a.png')], 1, '8-bit samples'),
            ([short, '-o', self.path('a.png')], 1, 'too short to draw'),
        ]
        for args, code, message in cases:
            result = self.run_script(*args)
            self.assertEqual(result.returncode, code, args)
            self.assertIn(message, result.stderr, args)
        self.assertFalse(os.path.exists(os.path.join(HERE, 'out')))


if __name__ == '__main__':
    unittest.main()
