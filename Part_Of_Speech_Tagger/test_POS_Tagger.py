# Tests for POS_Tagger.py
#
# Run from this folder with either of:
#     python3 -m unittest test_POS_Tagger
#     python3 test_POS_Tagger.py
#
# Most tests swap NLTK's Brown corpus and word list for a tiny made-up
# corpus (TOY_CORPUS), so every count, probability and best tagging can be
# worked out by hand. The tests at the bottom use the real Brown corpus and
# the saved model_dat.json, run the script from the command line, and check
# the output given in the README; they need NLTK's brown, words and punkt
# data (see the top-level README.md) and take several seconds. Every file
# the tests write goes in a temporary folder.

import contextlib
import io
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import nltk

import POS_Tagger as P

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL = os.path.join(HERE, 'model_dat.json')

# A tiny tagged corpus: 4 sentences, 11 words.
#   tag counts:   AT 3, NN 4, VBZ 4
#   word counts:  the 2 (The + the), a 1, dog 3, cat 1, runs 3, sleeps 1
#   TOY_WORDS (the stand-in for NLTK's word list) has the, a, dog and cat;
#   runs is added to the vocabulary because it appears more than once, and
#   sleeps (once, not in TOY_WORDS) becomes 'UNK'.
TOY_CORPUS = [
    [('The', 'AT'), ('dog', 'NN'), ('runs', 'VBZ')],
    [('the', 'AT'), ('cat', 'NN'), ('runs', 'VBZ')],
    [('a', 'AT'), ('dog', 'NN'), ('sleeps', 'VBZ')],
    [('dog', 'NN'), ('runs', 'VBZ')],
]
TOY_WORDS = ['The', 'a', 'dog', 'cat']
TOY_VOCAB = {'the', 'a', 'dog', 'cat', 'runs'}


def cost(p):
    """-log10 of a probability, as the model stores it."""
    return abs(math.log10(p))


# Hand-worked costs for TOY_CORPUS
TOY_EMISSIONS = {               # P(word | tag)
    ('the', 'AT'): cost(2 / 3), ('a', 'AT'): cost(1 / 3),
    ('dog', 'NN'): cost(3 / 4), ('cat', 'NN'): cost(1 / 4),
    ('runs', 'VBZ'): cost(3 / 4), ('UNK', 'VBZ'): cost(1 / 4),
}
TOY_TRANSITIONS = {             # P(tag | previous tag)
    ('<ST>', 'AT'): cost(3 / 4), ('<ST>', 'NN'): cost(1 / 4),
    ('AT', 'NN'): cost(3 / 3), ('NN', 'VBZ'): cost(4 / 4),
}


class FakeCorpus:
    """Stands in for nltk.corpus.brown and nltk.corpus.words."""
    def tagged_sents(self):
        return TOY_CORPUS

    def words(self):
        return TOY_WORDS


@contextlib.contextmanager
def toy_nltk():
    fake = FakeCorpus()
    with mock.patch('nltk.corpus.brown', fake), \
            mock.patch('nltk.corpus.words', fake):
        yield


@contextlib.contextmanager
def in_folder(folder):
    old = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(old)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)

    def write(self, name, text):
        with open(self.path(name), 'w') as f:
            f.write(text)
        return self.path(name)

    def toy_model(self):
        """Build the toy model into the temporary folder; return its path."""
        with toy_nltk(), in_folder(self.tmp):
            P.hmm_model_build()
        return self.path(P.DEFAULT_MODEL)


# --- Building the model ---

class TestCreateVocab(unittest.TestCase):
    def test_word_list_plus_words_seen_twice(self):
        with toy_nltk():
            self.assertEqual(P.create_vocab(TOY_CORPUS), TOY_VOCAB)

    def test_lower_case(self):
        # "The" (word list) and "The"/"the" (corpus) all become "the"
        with toy_nltk():
            vocab = P.create_vocab(TOY_CORPUS)
        self.assertIn('the', vocab)
        self.assertNotIn('The', vocab)


class TestCreateMatrices(unittest.TestCase):
    def setUp(self):
        with toy_nltk():
            self.m1, self.m2 = P.create_matrices()

    def test_emissions(self):
        expected = {(tag, (word, tag)): c
                    for (word, tag), c in TOY_EMISSIONS.items()}
        self.assertEqual(self.m1.keys(), expected.keys())
        for key, c in expected.items():
            self.assertAlmostEqual(self.m1[key], c, msg=key)

    def test_transitions(self):
        expected = {(prev, (tag, prev)): c
                    for (prev, tag), c in TOY_TRANSITIONS.items()}
        self.assertEqual(self.m2.keys(), expected.keys())
        for key, c in expected.items():
            self.assertAlmostEqual(self.m2[key], c, msg=key)

    def test_probabilities_add_up_to_one(self):
        # For each tag, P(word | tag) over all words sums to 1, and so does
        # P(tag | previous tag) over all tags
        def totals(matrix, group):
            sums = {}
            for key, c in matrix.items():
                sums[group(key)] = sums.get(group(key), 0) + 10 ** -c
            return sums
        for total in totals(self.m1, lambda k: k[0]).values():
            self.assertAlmostEqual(total, 1)
        for total in totals(self.m2, lambda k: k[0]).values():
            self.assertAlmostEqual(total, 1)

    def test_unknown_words(self):
        # "sleeps" is seen once and isn't in the word list, so it is 'UNK'
        self.assertIn(('VBZ', ('UNK', 'VBZ')), self.m1)
        self.assertNotIn(('VBZ', ('sleeps', 'VBZ')), self.m1)


class TestModelFile(TempDirTestCase):
    def test_format_matrix(self):
        matrix = {('NN', ('cat', 'NN')): 0.42, ('NN', ('dog', 'NN')): 0.1,
                  ('AT', ('the', 'AT')): 0.2}
        self.assertEqual(P._format_matrix(matrix),
                         {'NN': {'cat': 0.42, 'dog': 0.1},
                          'AT': {'the': 0.2}})

    def test_build_writes_model_in_current_folder(self):
        model = self.toy_model()
        with open(model) as f:
            saved = json.load(f)
        self.assertEqual(set(saved), {'m1', 'm2'})
        self.assertAlmostEqual(saved['m1']['NN']['dog'], cost(3 / 4))
        self.assertAlmostEqual(saved['m2']['<ST>']['AT'], cost(3 / 4))

    def test_read_back_matches_built(self):
        model = self.toy_model()
        with toy_nltk():
            built = P.create_matrices()
        self.assertEqual(P.read_matrix_file(model), built)


# --- Scoring a tagged sentence (--prob) ---

class TestHmmSequence(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.toy_model()

    def score(self, text):
        fname = self.write('tagged.txt', text)
        with toy_nltk():
            return P.hmm_sequence(self.model, fname)

    def test_hand_worked_sentence(self):
        # P(AT|<ST>) P(the|AT) P(NN|AT) P(dog|NN) P(VBZ|NN) P(runs|VBZ)
        expected = (3 / 4) * (2 / 3) * 1 * (3 / 4) * 1 * (3 / 4)
        self.assertAlmostEqual(self.score('the/AT dog/NN runs/VBZ\n'),
                               expected)

    def test_case_and_unknown_words(self):
        # "The" is lower-cased; "jumps" isn't in the vocabulary, so it is
        # scored as 'UNK' (P(UNK|VBZ) = 1/4)
        expected = (3 / 4) * (2 / 3) * 1 * (3 / 4) * 1 * (1 / 4)
        self.assertAlmostEqual(self.score('The/AT dog/NN jumps/VBZ\n'),
                               expected)

    def test_word_never_seen_with_tag(self):
        # "cat" was only seen as NN; as VBZ it falls back to P(UNK|VBZ)
        expected = (1 / 4) * (3 / 4) * 1 * (1 / 4)
        self.assertAlmostEqual(self.score('dog/NN cat/VBZ\n'), expected)

    def test_unseen_tag_and_transition(self):
        # JJ was never seen: its emission and the NN -> JJ transition both
        # cost UNSEEN_COST
        expected = (1 / 4) * (3 / 4) * 10 ** -(2 * P.UNSEEN_COST)
        self.assertAlmostEqual(
            math.log10(self.score('dog/NN dog/JJ\n')),
            math.log10(expected))

    def test_malformed_tokens_skipped(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            score = self.score('dog/NN nonsense /VBZ runs/\n')
        self.assertAlmostEqual(score, (1 / 4) * (3 / 4))
        for token in ['nonsense', '/VBZ', 'runs/']:
            self.assertIn("skipping '%s'" % token, err.getvalue())

    def test_slash_in_word(self):
        # Split on the last '/': "dog/cat/NN" is the word "dog/cat" with the
        # tag NN. The word is unseen (UNSEEN_COST, since NN has no 'UNK'
        # emission) but NN is a known tag, so the start transition is
        # P(NN|<ST>) = 1/4. Splitting on the first '/' would give the unseen
        # tag "cat/NN" and two UNSEEN_COSTs.
        self.assertAlmostEqual(math.log10(self.score('dog/cat/NN\n')),
                               -P.UNSEEN_COST + math.log10(1 / 4))


# --- Finding the best tagging (--tag) ---

class TestViterbiTag(unittest.TestCase):
    def test_hand_worked_sentence(self):
        # "dog" can be NN or VB, and VB has the cheaper emission, so a
        # word-by-word choice would pick VB. But VB -> VBZ was never seen
        # (it costs UNSEEN_COST), so the best path is NN VBZ:
        # 0.3 + 0.5 + 0.1 + 0.2 = 1.1, against 0.3 + 0.1 + 10 + 0.2 for VB
        emissions = {'dog': {'NN': 0.5, 'VB': 0.1},
                     'barks': {'VBZ': 0.2},
                     'UNK': {'NN': 3}}
        transitions = {('<ST>', 'NN'): 0.3, ('<ST>', 'VB'): 0.3,
                       ('NN', 'VBZ'): 0.1}
        tags, total = P.viterbi_tag(['dog', 'barks'], {'dog', 'barks'},
                                    emissions, transitions)
        self.assertEqual(tags, ['NN', 'VBZ'])
        self.assertAlmostEqual(total, 0.3 + 0.5 + 0.1 + 0.2)

    def test_one_word(self):
        emissions = {'dog': {'NN': 0.5, 'VB': 0.1}, 'UNK': {'NN': 3}}
        transitions = {('<ST>', 'NN'): 0.1, ('<ST>', 'VB'): 1.0}
        tags, total = P.viterbi_tag(['dog'], {'dog'}, emissions, transitions)
        self.assertEqual(tags, ['NN'])       # 0.6 beats 1.1
        self.assertAlmostEqual(total, 0.6)

    def test_unknown_word_uses_unk(self):
        emissions = {'UNK': {'NN': 0.5, 'JJ': 0.4}}
        transitions = {('<ST>', 'NN'): 0.1, ('<ST>', 'JJ'): 1.0}
        tags, _ = P.viterbi_tag(['Zork'], set(), emissions, transitions)
        self.assertEqual(tags, ['NN'])

    def test_in_vocab_but_not_in_model_uses_unk(self):
        emissions = {'UNK': {'NN': 0.5}}
        tags, _ = P.viterbi_tag(['cat'], {'cat'}, emissions,
                                {('<ST>', 'NN'): 0})
        self.assertEqual(tags, ['NN'])

    def test_case_insensitive(self):
        emissions = {'dog': {'NN': 0.5}, 'UNK': {'VB': 0.1}}
        tags, _ = P.viterbi_tag(['DOG'], {'dog'}, emissions,
                                {('<ST>', 'NN'): 0, ('<ST>', 'VB'): 0})
        self.assertEqual(tags, ['NN'])

    def test_matches_brute_force(self):
        # On small random models, Viterbi must find the same lowest cost as
        # trying every possible tagging
        rng = random.Random(0)
        tag_names = ['A', 'B', 'C']
        for trial in range(30):
            words = ['w%d' % i for i in range(4)]
            emissions = {w: {t: rng.uniform(0, 3) for t in tag_names
                             if rng.random() < 0.8} or {'A': 1.0}
                         for w in words}
            emissions['UNK'] = {'A': 5.0}
            transitions = {(p, t): rng.uniform(0, 3)
                           for p in ['<ST>'] + tag_names for t in tag_names
                           if rng.random() < 0.7}
            tokens = [rng.choice(words) for _ in range(rng.randint(1, 5))]

            def path_cost(tags):
                total, prev = 0, '<ST>'
                for token, tag in zip(tokens, tags):
                    total += transitions.get((prev, tag), P.UNSEEN_COST)
                    total += emissions[token][tag]
                    prev = tag
                return total
            best = min(path_cost(tags) for tags in itertools.product(
                *[list(emissions[t]) for t in tokens]))
            tags, total = P.viterbi_tag(tokens, set(words), emissions,
                                        transitions)
            self.assertAlmostEqual(total, best, msg=trial)
            self.assertAlmostEqual(path_cost(tags), total, msg=trial)


class TestHmmViterbi(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            nltk.word_tokenize('Test sentence.')
        except LookupError:
            raise unittest.SkipTest("needs NLTK's punkt data")

    def test_tags_each_line(self):
        model = self.toy_model()
        text = self.write('in.txt', 'The dog runs\n\na cat sleeps\n')
        with toy_nltk():
            results = P.hmm_viterbi(model, text)
        self.assertEqual(len(results), 2)        # the blank line is skipped
        tokens, tags, probability = results[0]
        self.assertEqual(tokens, ['The', 'dog', 'runs'])
        self.assertEqual(tags, ['AT', 'NN', 'VBZ'])
        self.assertAlmostEqual(probability,
                               (3 / 4) * (2 / 3) * 1 * (3 / 4) * 1 * (3 / 4))
        self.assertEqual(results[1][1], ['AT', 'NN', 'VBZ'])


# --- Command-line handling ---

class TestArguments(TempDirTestCase):
    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, 'argv', ['POS_Tagger.py', *args]), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            try:
                P.main()
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_help(self):
        for args in [(), ('--help',), ('-h',)]:
            code, out, _ = self.run_main(*args)
            self.assertEqual(code, 0)
            self.assertIn('Usage:', out)

    def test_usage_errors(self):
        text = self.write('in.txt', 'the/AT\n')
        for args, message in [
                (('--frobnicate',), "unknown option '--frobnicate'"),
                (('--build', 'extra'), '--build takes no arguments'),
                (('--prob',), '--prob takes a text file'),
                (('--tag', text, 'model', 'extra'), '--tag takes a text file')]:
            code, _, err = self.run_main(*args)
            self.assertEqual(code, 2, args)
            self.assertIn(message, err)
            self.assertIn('Usage:', err)

    def test_missing_files(self):
        # sys.exit() with a message: the message is the exit "code"
        text = self.write('in.txt', 'the/AT\n')
        message, _, _ = self.run_main('--prob', self.path('nope.txt'))
        self.assertIn('not found', message)
        self.assertIn('text file', message)
        message, _, _ = self.run_main('--prob', text, self.path('nope.json'))
        self.assertIn('model file', message)
        self.assertIn("Run 'python3 POS_Tagger.py --build'", message)

    def test_default_model(self):
        text = self.write('in.txt', 'the/AT\n')
        open(self.path(P.DEFAULT_MODEL), 'w').close()
        other = self.write('other.json', '{}')
        with in_folder(self.tmp):
            self.assertEqual(P.parse_file_args('--prob', [text]),
                             (text, P.DEFAULT_MODEL))
            self.assertEqual(P.parse_file_args('--prob', [text, other]),
                             (text, other))

    def test_download_nltk_data(self):
        missing = LookupError('missing')
        with mock.patch('nltk.data.find', side_effect=missing), \
                mock.patch('nltk.download', return_value=True) as download:
            P.download_nltk_data()
        self.assertEqual([c.args[0] for c in download.call_args_list],
                         ['brown', 'words', 'punkt', 'punkt_tab'])
        with mock.patch('nltk.data.find', side_effect=missing), \
                mock.patch('nltk.download', return_value=False):
            with self.assertRaises(SystemExit) as stop:
                P.download_nltk_data()
        self.assertIn("could not download NLTK package 'brown'",
                      str(stop.exception.code))
        with mock.patch('nltk.data.find'), \
                mock.patch('nltk.download') as download:
            P.download_nltk_data()
        download.assert_not_called()


# --- End to end: the real Brown corpus and the saved model ---

class TestCommandLine(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            for path in ['corpora/brown', 'corpora/words', 'tokenizers/punkt']:
                nltk.data.find(path)
        except LookupError:
            raise unittest.SkipTest("needs NLTK's brown, words and punkt "
                                    "data (see the top-level README.md)")

    def run_script(self, *args, cwd=HERE):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, 'POS_Tagger.py'), *args],
            capture_output=True, text=True, cwd=cwd)

    def test_prob_readme_example(self):
        result = self.run_script('--prob', 'test_tagged_sentence.txt')
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.splitlines()[-1]
        self.assertTrue(line.startswith('Total probability: '), line)
        probability = float(line.split(': ')[1])
        self.assertAlmostEqual(math.log10(probability),
                               math.log10(1.016489051401748e-82))

    def test_tag_readme_examples(self):
        for fname, expected in [
                ('test_untagged.txt',
                 'They/PPSS can/MD fish/NN later/RBR in/IN the/AT night/NN '
                 'when/WRB the/AT moon/NN is/BEZ bright/JJ ./.'),
                ('test_untagged_2.txt',
                 'The/AT quick/JJ fox/NN jumped/VBD over/IN the/AT lazy/JJ '
                 'dog/NN ./.')]:
            result = self.run_script('--tag', fname)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout.splitlines(), fname)
            self.assertIn('Probability of best tagging:', result.stdout)

    def test_build_matches_saved_model(self):
        # --build writes model_dat.json to the current folder; the saved
        # model is pretty-printed, so compare the contents, not the bytes
        result = self.run_script('--build', cwd=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Model saved to model_dat.json', result.stdout)
        with open(self.path('model_dat.json')) as built, \
                open(SAVED_MODEL) as saved:
            self.assertEqual(json.load(built), json.load(saved))

    def test_missing_model(self):
        result = self.run_script('--prob', os.path.join(
            HERE, 'test_tagged_sentence.txt'), cwd=self.tmp)
        self.assertEqual(result.returncode, 1)
        self.assertIn("model file 'model_dat.json' not found", result.stderr)


if __name__ == '__main__':
    unittest.main()
