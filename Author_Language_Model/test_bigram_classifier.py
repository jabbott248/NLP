# Tests for bigram_classifier.py
#
# Run from this folder with either of:
#     python3 -m unittest test_bigram_classifier
#     python3 test_bigram_classifier.py
#
# The unit tests use tiny made-up corpora so the expected counts and
# probabilities can be worked out by hand. The end-to-end tests at the
# bottom train on the real author files and take a few seconds.

import contextlib
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from unittest import mock

import bigram_classifier as C

HERE = os.path.dirname(os.path.abspath(__file__))

# A tiny corpus: 3 sentences, 9 tokens.
#   unigrams: the 2, cat 2, sat 2, dog 1, a 1, ran 1
#   bigrams:  (<s>, the) 2, and 7 others seen once
#   so N_1 = 7, N_2 = 1
TOY = [['the', 'cat', 'sat'], ['the', 'dog', 'sat'], ['a', 'cat', 'ran']]


def trained(sentences, author='toy'):
    lm = C.LanguageModel(author=author)
    lm.train(sentences)
    return lm


def smoothed(*corpora):
    """Train one model per corpus and smooth them together."""
    models = [trained(s, author=f'author{i}') for i, s in enumerate(corpora)]
    C.smooth_models(models)
    return models


def shared_vocab(models):
    vocab = set()
    for lm in models:
        vocab.update(lm.unigrams)
    return vocab


class TestReadSentences(unittest.TestCase):
    def test_joins_wrapped_lines_and_splits_sentences(self):
        text = ("The cat sat\non the mat. It was\nhappy.\n"
                "\n\n"
                "A new paragraph\nstarts here\n")
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write(text)
        try:
            sentences = C.read_sentences(f.name)
        finally:
            os.remove(f.name)
        self.assertEqual([s for s, _ in sentences],
                         ['The cat sat on the mat.', 'It was happy.',
                          'A new paragraph starts here'])
        self.assertEqual(sentences[0][1],
                         ['The', 'cat', 'sat', 'on', 'the', 'mat', '.'])

    def test_blank_file_has_no_sentences(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write('\n\n   \n')
        try:
            self.assertEqual(C.read_sentences(f.name), [])
        finally:
            os.remove(f.name)


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.lm = trained(TOY)

    def test_unigram_counts(self):
        self.assertEqual(self.lm.unigrams,
                         Counter({'the': 2, 'cat': 2, 'sat': 2,
                                  'dog': 1, 'a': 1, 'ran': 1}))
        self.assertEqual(self.lm.total_words, 9)

    def test_start_token_is_a_context_not_a_word(self):
        self.assertNotIn(C.START_TOKEN, self.lm.unigrams)
        self.assertEqual(self.lm.bigrams[(C.START_TOKEN, 'the')], 2)
        self.assertEqual(self.lm.bigrams[(C.START_TOKEN, 'a')], 1)

    def test_bigrams_do_not_cross_sentences(self):
        self.assertNotIn(('sat', 'the'), self.lm.bigrams)
        self.assertNotIn(('sat', 'a'), self.lm.bigrams)
        self.assertEqual(sum(self.lm.bigrams.values()), 9)  # one per token

    def test_frequency_of_frequencies(self):
        # N_c counts bigrams seen exactly c times, under their final count
        self.assertEqual(self.lm.frequencies, Counter({1: 7, 2: 1}))


class TestGoodTuring(unittest.TestCase):
    def test_adjusted_count_formula(self):
        lm = trained(TOY)
        # c* = (c + 1) * N_(c+1) / N_c = 2 * 1 / 7
        self.assertAlmostEqual(lm.gt_adjusted_count(1), 2 / 7)

    def test_raw_count_when_next_frequency_is_missing(self):
        lm = trained(TOY)
        self.assertEqual(lm.gt_adjusted_count(2), 2)   # N_3 = 0

    def test_raw_count_at_or_above_max(self):
        lm = trained(TOY)
        lm.frequencies = Counter({C.GT_MAX_COUNT: 2, C.GT_MAX_COUNT + 1: 1})
        self.assertEqual(lm.gt_adjusted_count(C.GT_MAX_COUNT), C.GT_MAX_COUNT)


class TestSmoothing(unittest.TestCase):
    def setUp(self):
        other = [['a', 'dog', 'barked'], ['the', 'bird', 'sang']]
        self.models = smoothed(TOY, other)
        self.vocab = shared_vocab(self.models)

    def test_unigram_probabilities_add_up_to_1(self):
        # every word in the shared vocabulary, plus one slot for words no
        # author used
        for lm in self.models:
            total = sum(lm.unigram_prob(w) for w in self.vocab) + lm.unknown_unigram
            self.assertAlmostEqual(total, 1)

    def test_next_word_probabilities_add_up_to_1(self):
        for lm in self.models:
            for prev in lm.context_counts:
                total = sum(C.gt_get_prob_of_w(lm, w, prev) for w in self.vocab)
                total += C.gt_get_prob_of_w(lm, 'never-seen-word', prev)
                self.assertAlmostEqual(total, 1, msg=f'{lm.author}, after {prev!r}')

    def test_probabilities_are_valid(self):
        for lm in self.models:
            for prev, word in lm.bigrams:
                self.assertTrue(0 < C.gt_get_prob_of_w(lm, word, prev) <= 1)

    def test_unseen_words_split_the_unseen_share(self):
        # The words an author never used share N_1 / N between them, rather
        # than each one getting all of it.
        for lm in self.models:
            unseen = len(self.vocab) + 1 - len(lm.unigrams)
            once = sum(1 for c in lm.unigrams.values() if c == 1)
            self.assertAlmostEqual(lm.unknown_unigram * unseen, once / lm.total_words)

    def test_unseen_bigram_backs_off_to_unigram(self):
        lm = self.models[0]
        # 'dog' was never seen after 'cat', so it gets alpha(cat) * P(dog)
        self.assertNotIn(('cat', 'dog'), lm.bigrams)
        self.assertAlmostEqual(C.gt_get_prob_of_w(lm, 'dog', 'cat'),
                               lm.backoff_weight['cat'] * lm.unigram_prob('dog'))

    def test_unseen_previous_word_uses_unigram(self):
        lm = self.models[0]
        self.assertEqual(C.gt_get_prob_of_w(lm, 'cat', 'zebra'),
                         lm.unigram_prob('cat'))

    def test_fallback_when_nothing_is_discounted(self):
        # 'x' is always followed by 'y', 5 times, so Good-Turing frees up
        # nothing after 'x'. Unseen words after 'x' must still be possible,
        # and the probabilities after 'x' must still add up to 1.
        models = smoothed([['x', 'y']] * C.GT_MAX_COUNT + TOY, [['z']])
        lm = models[0]
        self.assertIn('x', lm.seen_scale)
        self.assertGreater(C.gt_get_prob_of_w(lm, 'cat', 'x'), 0)
        total = sum(C.gt_get_prob_of_w(lm, w, 'x') for w in shared_vocab(models))
        total += C.gt_get_prob_of_w(lm, 'never-seen-word', 'x')
        self.assertAlmostEqual(total, 1)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.austen_like = [['the', 'lady', 'smiled'], ['the', 'lady', 'danced']]
        self.dickens_like = [['a', 'boy', 'ran'], ['a', 'boy', 'cried']]
        self.models = smoothed(self.austen_like, self.dickens_like)

    def test_cost_is_sum_of_minus_log10_probabilities(self):
        lm = self.models[0]
        tokens = ['the', 'lady', 'ran']
        expected = (-math.log10(C.gt_get_prob_of_w(lm, 'the', C.START_TOKEN))
                    - math.log10(C.gt_get_prob_of_w(lm, 'lady', 'the'))
                    - math.log10(C.gt_get_prob_of_w(lm, 'ran', 'lady')))
        self.assertAlmostEqual(C.get_prob_of_sentence(lm, tokens), expected)

    def test_empty_sentence_costs_nothing(self):
        self.assertEqual(C.get_prob_of_sentence(self.models[0], []), 0)

    def test_seen_sentence_costs_less_than_shuffled(self):
        lm = self.models[0]
        self.assertLess(C.get_prob_of_sentence(lm, ['the', 'lady', 'smiled']),
                        C.get_prob_of_sentence(lm, ['smiled', 'lady', 'the']))

    def test_classify_picks_lowest_cost_author(self):
        self.assertIs(C.classify(self.models, ['the', 'lady', 'danced']), self.models[0])
        self.assertIs(C.classify(self.models, ['a', 'boy', 'smiled']), self.models[1])


class TestAuthorList(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        for name in ('austen.txt', 'dickens.txt'):
            with open(os.path.join(self.dir.name, name), 'w') as f:
                f.write('Some text.\n')

    def tearDown(self):
        self.dir.cleanup()

    def write_list(self, text):
        path = os.path.join(self.dir.name, 'authors')
        with open(path, 'w') as f:
            f.write(text)
        return path

    def test_author_name(self):
        self.assertEqual(C.author_name('texts/austen.txt'), 'austen')
        self.assertEqual(C.author_name('wilde'), 'wilde')

    def test_paths_are_relative_to_the_list_and_blank_lines_ignored(self):
        path = self.write_list('austen.txt\n\ndickens.txt\n')
        self.assertEqual(C.read_authorlist(path),
                         [os.path.join(self.dir.name, 'austen.txt'),
                          os.path.join(self.dir.name, 'dickens.txt')])

    def test_missing_author_file_exits(self):
        path = self.write_list('austen.txt\nmissing.txt\n')
        with self.assertRaises(SystemExit) as cm:
            C.read_authorlist(path)
        self.assertIn('missing.txt', str(cm.exception.code))

    def test_needs_two_authors(self):
        path = self.write_list('austen.txt\n')
        with self.assertRaises(SystemExit) as cm:
            C.read_authorlist(path)
        self.assertIn('at least two', str(cm.exception.code))


class TestNltkDownload(unittest.TestCase):
    def test_nothing_downloaded_when_data_is_present(self):
        with mock.patch.object(C.nltk.data, 'find'), \
             mock.patch.object(C.nltk, 'download') as download:
            C.download_nltk_data()
        download.assert_not_called()

    def test_failed_download_exits_with_package_name(self):
        with mock.patch.object(C.nltk.data, 'find', side_effect=LookupError), \
             mock.patch.object(C.nltk, 'download', return_value=False):
            with self.assertRaises(SystemExit) as cm:
                C.download_nltk_data()
        self.assertIn("'punkt'", str(cm.exception.code))


def run_cli(*args, cwd=HERE):
    return subprocess.run([sys.executable, os.path.join(HERE, 'bigram_classifier.py'), *args],
                          cwd=cwd, capture_output=True, text=True)


class TestCommandLine(unittest.TestCase):
    def test_no_arguments_prints_help(self):
        result = run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage: bigram_classifier.py', result.stdout)

    def test_missing_author_list_is_an_error(self):
        result = run_cli('nope.txt')
        self.assertEqual(result.returncode, 2)
        self.assertIn("author list 'nope.txt' not found", result.stderr)

    def test_missing_test_file_is_an_error(self):
        result = run_cli('authorlist1', '-test', 'nope.txt')
        self.assertEqual(result.returncode, 2)
        self.assertIn("test file 'nope.txt' not found", result.stderr)


class TestRealModels(unittest.TestCase):
    """Smoothing checks on models trained on the real author files."""

    @classmethod
    def setUpClass(cls):
        cls.models = []
        for fname in C.read_authorlist(os.path.join(HERE, 'authorlist1')):
            lm = C.LanguageModel(author=C.author_name(fname))
            lm.create_lm(fname)
            cls.models.append(lm)
        C.smooth_models(cls.models)
        cls.vocab = shared_vocab(cls.models)

    def test_unseen_word_less_likely_than_word_seen_once(self):
        # an unseen word must not score higher than a word seen once
        for lm in self.models:
            seen_once = next(w for w, c in lm.unigrams.items() if c == 1)
            self.assertLess(lm.unigram_prob('never-seen-word'),
                            lm.unigram_prob(seen_once), lm.author)

    def test_next_word_probabilities_add_up_to_1(self):
        # the 20 most common previous words, plus every one that needed the
        # fallback in smooth(), for one author
        lm = self.models[0]
        common = [w for w, _ in Counter(lm.context_counts).most_common(20)]
        for prev in common + list(lm.seen_scale):
            total = sum(C.gt_get_prob_of_w(lm, w, prev) for w in self.vocab)
            total += C.gt_get_prob_of_w(lm, 'never-seen-word', prev)
            self.assertAlmostEqual(total, 1, msg=f'after {prev!r}')


class TestEndToEnd(unittest.TestCase):
    """Runs both modes on the real author files (takes a few seconds)."""

    def test_default_mode_accuracy(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            C.run_no_test_flag(os.path.join(HERE, 'authorlist1'))
        right = total = 0
        for line in out.getvalue().splitlines():
            name, score, _ = line.split()
            self.assertIn(name, ('austen', 'dickens', 'wilde', 'tolstoy'))
            r, n = map(int, score.split('/'))
            right, total = right + r, total + n
        self.assertGreater(right / total, 0.75)   # 80.9% when written

    def test_test_mode_from_another_folder(self):
        # -test before the author list, run from the parent folder
        parent = os.path.dirname(HERE)
        folder = os.path.basename(HERE)
        result = run_cli('-test', os.path.join(folder, 'test.txt'),
                         os.path.join(folder, 'authorlist1'), cwd=parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        guesses = [line.split()[2] for line in result.stdout.splitlines()]
        self.assertEqual(len(guesses), 33)
        # test.txt has one held-out passage per author (see the README):
        # each author should get most of their own passage right
        passages = {'austen': range(0, 7), 'dickens': range(7, 13),
                    'wilde': range(13, 26), 'tolstoy': range(26, 33)}
        for author, sentences in passages.items():
            right = sum(guesses[i] == author for i in sentences)
            self.assertGreaterEqual(right, len(sentences) / 2, author)


if __name__ == '__main__':
    unittest.main()
