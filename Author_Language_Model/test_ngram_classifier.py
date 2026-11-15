# Tests for ngram_classifier.py
#
# Run from this folder with either of:
#     python3 -m unittest test_ngram_classifier
#     python3 test_ngram_classifier.py
#
# The unit tests use tiny made-up corpora so the expected counts and
# probabilities can be worked out by hand, and check the smoothing at every
# n-gram order the script accepts (1-5). The tests at the bottom train on
# the real author files, check that n = 2 gives exactly the same results as
# bigram_classifier.py, and take several seconds.

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

import bigram_classifier as B
import ngram_classifier as C

HERE = os.path.dirname(os.path.abspath(__file__))
S = C.START_TOKEN
ALL_N = range(C.MIN_N, C.MAX_N + 1)

# A tiny corpus: 3 sentences, 9 tokens.
#   unigrams: the 2, cat 2, sat 2, dog 1, a 1, ran 1
#   bigrams:  (<s>, the) 2, and 7 others seen once
#   trigrams: (<s>, <s>, the) 2, and 7 others seen once
TOY = [['the', 'cat', 'sat'], ['the', 'dog', 'sat'], ['a', 'cat', 'ran']]
OTHER = [['a', 'dog', 'barked'], ['the', 'bird', 'sang']]


def trained(sentences, n, author='toy'):
    lm = C.LanguageModel(author=author, n=n)
    lm.train(sentences)
    return lm


def smoothed(n, *corpora):
    """Train one order-n model per corpus and smooth them together."""
    models = [trained(s, n, author=f'author{i}') for i, s in enumerate(corpora)]
    C.smooth_models(models)
    return models


def shared_vocab(models):
    vocab = set()
    for lm in models:
        vocab.update(lm.unigrams)
    return vocab


def total_prob(lm, context, vocab):
    """P(w | context) summed over the vocabulary plus one unseen word."""
    return (sum(C.gt_get_prob_of_w(lm, w, context) for w in vocab)
            + C.gt_get_prob_of_w(lm, 'never-seen-word', context))


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


class TestTraining(unittest.TestCase):
    def test_n_must_be_in_range(self):
        for n in (0, C.MAX_N + 1):
            with self.assertRaises(ValueError):
                C.LanguageModel('toy', n=n)

    def test_unigram_model_has_no_ngram_tables(self):
        lm = trained(TOY, 1)
        self.assertEqual(lm.ngrams, {})
        self.assertEqual(lm.unigrams['cat'], 2)
        self.assertEqual(lm.total_words, 9)

    def test_trigram_counts(self):
        lm = trained(TOY, 3)
        self.assertEqual(lm.ngrams[3], Counter({
            (S, S, 'the'): 2, (S, 'the', 'cat'): 1, ('the', 'cat', 'sat'): 1,
            (S, 'the', 'dog'): 1, ('the', 'dog', 'sat'): 1,
            (S, S, 'a'): 1, (S, 'a', 'cat'): 1, ('a', 'cat', 'ran'): 1}))
        self.assertEqual(lm.frequencies[3], Counter({1: 7, 2: 1}))

    def test_lower_orders_match_the_bigram_classifier(self):
        # padding with extra START_TOKENs mustn't change the bigram counts
        bigram = B.LanguageModel('toy')
        bigram.train(TOY)
        for n in range(2, C.MAX_N + 1):
            lm = trained(TOY, n)
            self.assertEqual(lm.ngrams[2], bigram.bigrams, f'n={n}')
            self.assertEqual(lm.unigrams, bigram.unigrams, f'n={n}')

    def test_ngrams_do_not_cross_sentences_or_end_on_padding(self):
        lm = trained(TOY, 4)
        for k, counts in lm.ngrams.items():
            for gram in counts:
                self.assertNotEqual(gram[-1], S)
                self.assertEqual(len(gram), k)
            self.assertEqual(sum(counts.values()), 9)   # one per token
        self.assertNotIn(('cat', 'sat', 'the'), lm.ngrams[3])

    def test_start_token_is_not_a_word(self):
        self.assertNotIn(S, trained(TOY, 3).unigrams)


class TestGoodTuring(unittest.TestCase):
    def test_each_order_uses_its_own_frequency_table(self):
        lm = trained(TOY + [['the', 'cat', 'ran']], 3)
        # bigrams: (<s>, the) 3, (the, cat) 2, ...; trigrams: (<s>, <s>, the) 3
        for k in (2, 3):
            n1, n2 = lm.frequencies[k][1], lm.frequencies[k][2]
            if n2:
                self.assertAlmostEqual(lm.gt_adjusted_count(k, 1), 2 * n2 / n1)
            else:
                self.assertEqual(lm.gt_adjusted_count(k, 1), 1)
        self.assertNotEqual(lm.frequencies[2], lm.frequencies[3])

    def test_raw_count_at_or_above_max(self):
        lm = trained(TOY, 3)
        lm.frequencies[3] = Counter({C.GT_MAX_COUNT: 2, C.GT_MAX_COUNT + 1: 1})
        self.assertEqual(lm.gt_adjusted_count(3, C.GT_MAX_COUNT), C.GT_MAX_COUNT)


class TestSmoothing(unittest.TestCase):
    def test_probabilities_add_up_to_1_at_every_order(self):
        for n in ALL_N:
            models = smoothed(n, TOY, OTHER)
            vocab = shared_vocab(models)
            for lm in models:
                self.assertAlmostEqual(total_prob(lm, (), vocab), 1, msg=f'n={n} unigrams')
                for k in range(2, n + 1):
                    for context in lm.context_counts[k]:
                        self.assertAlmostEqual(total_prob(lm, context, vocab), 1,
                                               msg=f'n={n}, after {context}')

    def test_probabilities_are_valid(self):
        for n in ALL_N:
            lm = smoothed(n, TOY, OTHER)[0]
            for k in range(2, n + 1):
                for gram in lm.ngrams[k]:
                    p = C.gt_get_prob_of_w(lm, gram[-1], gram[:-1])
                    self.assertTrue(0 < p <= 1, f'n={n}, {gram}: {p}')

    def test_unseen_trigram_backs_off_to_bigram(self):
        lm = smoothed(3, TOY, OTHER)[0]
        context = (S, 'a')        # seen context, unseen trigram (<s>, a, dog)
        self.assertIn(context, lm.context_counts[3])
        self.assertNotIn(context + ('dog',), lm.ngrams[3])
        self.assertAlmostEqual(C.gt_get_prob_of_w(lm, 'dog', context),
                               lm.backoff_weight[3][context]
                               * C.gt_get_prob_of_w(lm, 'dog', ('a',)))

    def test_unseen_context_uses_next_order_down(self):
        lm = smoothed(3, TOY, OTHER)[0]
        self.assertEqual(C.gt_get_prob_of_w(lm, 'cat', ('zebra', 'the')),
                         C.gt_get_prob_of_w(lm, 'cat', ('the',)))
        self.assertEqual(C.gt_get_prob_of_w(lm, 'cat', ('zebra', 'yak')),
                         lm.unigram_prob('cat'))

    def test_fallback_when_nothing_is_discounted(self):
        # 'x y' is always followed by 'z', 5 times, so Good-Turing frees up
        # nothing after ('x', 'y'). Unseen words must still be possible, and
        # the probabilities must still add up to 1.
        models = smoothed(3, [['x', 'y', 'z']] * C.GT_MAX_COUNT + TOY, OTHER)
        lm = models[0]
        self.assertIn(('x', 'y'), lm.seen_scale[3])
        self.assertGreater(C.gt_get_prob_of_w(lm, 'cat', ('x', 'y')), 0)
        self.assertAlmostEqual(total_prob(lm, ('x', 'y'), shared_vocab(models)), 1)


class TestScoring(unittest.TestCase):
    def test_cost_uses_n_minus_1_words_of_context(self):
        lm = smoothed(3, TOY, OTHER)[0]
        tokens = ['the', 'cat', 'ran']
        expected = (-math.log10(C.gt_get_prob_of_w(lm, 'the', (S, S)))
                    - math.log10(C.gt_get_prob_of_w(lm, 'cat', (S, 'the')))
                    - math.log10(C.gt_get_prob_of_w(lm, 'ran', ('the', 'cat'))))
        self.assertAlmostEqual(C.get_prob_of_sentence(lm, tokens), expected)

    def test_unigram_cost_ignores_word_order(self):
        lm = smoothed(1, TOY, OTHER)[0]
        self.assertAlmostEqual(C.get_prob_of_sentence(lm, ['the', 'cat', 'sat']),
                               C.get_prob_of_sentence(lm, ['sat', 'cat', 'the']))

    def test_seen_sentence_costs_less_than_shuffled(self):
        for n in range(2, C.MAX_N + 1):
            lm = smoothed(n, TOY, OTHER)[0]
            self.assertLess(C.get_prob_of_sentence(lm, ['the', 'cat', 'sat']),
                            C.get_prob_of_sentence(lm, ['sat', 'cat', 'the']), f'n={n}')

    def test_classify_picks_lowest_cost_author(self):
        austen_like = [['the', 'lady', 'smiled'], ['the', 'lady', 'danced']]
        dickens_like = [['a', 'boy', 'ran'], ['a', 'boy', 'cried']]
        for n in ALL_N:
            models = smoothed(n, austen_like, dickens_like)
            self.assertIs(C.classify(models, ['the', 'lady', 'danced']), models[0], f'n={n}')
            self.assertIs(C.classify(models, ['a', 'boy', 'smiled']), models[1], f'n={n}')

    def test_bigram_matches_bigram_classifier(self):
        # with n = 2 every probability is the same as bigram_classifier.py's
        ngram_models = smoothed(2, TOY, OTHER)
        bigram_models = []
        for i, corpus in enumerate((TOY, OTHER)):
            lm = B.LanguageModel(f'author{i}')
            lm.train(corpus)
            bigram_models.append(lm)
        B.smooth_models(bigram_models)
        words = shared_vocab(ngram_models) | {'never-seen-word'}
        for ng, bg in zip(ngram_models, bigram_models):
            for prev in words | {S}:
                for word in words:
                    self.assertAlmostEqual(C.gt_get_prob_of_w(ng, word, (prev,)),
                                           B.gt_get_prob_of_w(bg, word, prev))


class TestAuthorListAndDownload(unittest.TestCase):
    def test_paths_are_relative_to_the_list(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('austen.txt', 'dickens.txt'):
                with open(os.path.join(d, name), 'w') as f:
                    f.write('Some text.\n')
            path = os.path.join(d, 'authors')
            with open(path, 'w') as f:
                f.write('austen.txt\n\ndickens.txt\n')
            self.assertEqual(C.read_authorlist(path),
                             [os.path.join(d, 'austen.txt'), os.path.join(d, 'dickens.txt')])

    def test_failed_download_exits_with_package_name(self):
        with mock.patch.object(C.nltk.data, 'find', side_effect=LookupError), \
             mock.patch.object(C.nltk, 'download', return_value=False):
            with self.assertRaises(SystemExit) as cm:
                C.download_nltk_data()
        self.assertIn("'punkt'", str(cm.exception.code))


def run_cli(*args, cwd=HERE):
    return subprocess.run([sys.executable, os.path.join(HERE, 'ngram_classifier.py'), *args],
                          cwd=cwd, capture_output=True, text=True)


class TestCommandLine(unittest.TestCase):
    def test_no_arguments_prints_help(self):
        result = run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage: ngram_classifier.py', result.stdout)
        self.assertIn('--ngram N', result.stdout)

    def test_ngram_must_be_1_to_5(self):
        for bad in ('0', '6', 'abc'):
            result = run_cli('authorlist1', '--ngram', bad)
            self.assertEqual(result.returncode, 2, bad)
            self.assertIn('argument --ngram', result.stderr)

    def test_missing_author_list_is_an_error(self):
        result = run_cli('nope.txt', '--ngram', '3')
        self.assertEqual(result.returncode, 2)
        self.assertIn("author list 'nope.txt' not found", result.stderr)


class TestRealModels(unittest.TestCase):
    """Checks on trigram models trained on the real author files."""

    @classmethod
    def setUpClass(cls):
        cls.models = []
        for fname in C.read_authorlist(os.path.join(HERE, 'authorlist1')):
            lm = C.LanguageModel(author=C.author_name(fname), n=3)
            lm.create_lm(fname)
            cls.models.append(lm)
        C.smooth_models(cls.models)
        cls.vocab = shared_vocab(cls.models)

    def test_unseen_word_less_likely_than_word_seen_once(self):
        for lm in self.models:
            seen_once = next(w for w, c in lm.unigrams.items() if c == 1)
            self.assertLess(lm.unigram_prob('never-seen-word'),
                            lm.unigram_prob(seen_once), lm.author)

    def test_probabilities_add_up_to_1(self):
        # the 10 most common contexts of each order, plus a few that needed
        # the fallback in smooth(), for one author
        lm = self.models[0]
        for k in (2, 3):
            common = [c for c, _ in Counter(lm.context_counts[k]).most_common(10)]
            for context in common + list(lm.seen_scale[k])[:5]:
                self.assertAlmostEqual(total_prob(lm, context, self.vocab), 1,
                                       msg=f'after {context}')


class TestEndToEnd(unittest.TestCase):
    """Runs both modes on the real author files (takes several seconds)."""

    def default_mode(self, run, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run(os.path.join(HERE, 'authorlist1'), *args)
        return out.getvalue()

    def test_default_n_matches_bigram_classifier(self):
        self.assertEqual(self.default_mode(C.run_no_test_flag),
                         self.default_mode(B.run_no_test_flag))

    def test_trigram_default_mode_accuracy(self):
        right = total = 0
        for line in self.default_mode(C.run_no_test_flag, 3).splitlines():
            name, score, _ = line.split()
            self.assertIn(name, ('austen', 'dickens', 'wilde', 'tolstoy'))
            r, n = map(int, score.split('/'))
            right, total = right + r, total + n
        self.assertGreater(right / total, 0.75)   # 80.8% when written

    def test_trigram_test_mode_from_another_folder(self):
        parent = os.path.dirname(HERE)
        folder = os.path.basename(HERE)
        result = run_cli('-test', os.path.join(folder, 'test.txt'), '--ngram', '3',
                         os.path.join(folder, 'authorlist1'), cwd=parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        guesses = [line.split()[2] for line in result.stdout.splitlines()]
        self.assertEqual(len(guesses), 33)
        passages = {'austen': range(0, 7), 'dickens': range(7, 13),
                    'wilde': range(13, 26), 'tolstoy': range(26, 33)}
        for author, sentences in passages.items():
            right = sum(guesses[i] == author for i in sentences)
            self.assertGreaterEqual(right, len(sentences) / 2, author)


if __name__ == '__main__':
    unittest.main()
