# Tests for spellchecker.py (and make_brown_words.py)
#
# Run from this folder with either of:
#     python3 -m unittest test_spellchecker
#     python3 test_spellchecker.py
#
# The unit tests mostly use a tiny made-up dictionary (TOY_WORDS) so the
# expected answers can be worked out by hand. The tests further down load
# the real word lists in files/ (about half a second) and run the script
# from the command line. Every file the tests write goes in a temporary
# folder.

import collections
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

import spellchecker as sc
import make_brown_words

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, 'files')

TOY_WORDS = ['a', 'an', 'and', 'abandon', 'abandoned', 'help', 'will',
             'willing', 'willow', 'wild', 'wile', 'city', 'tax', 'wish',
             'hope', 'see', 'big', 'run', 'happy', 'final', 'simple',
             'self', 'knife', 'woman', 'child', 'have', 'do', 'they',
             'people', 'America', 'this', 'cat']


def toy_trie(words=TOY_WORDS):
    root = sc.TrieNode('*')
    for word in words:
        sc.insert(root, word)
    return root


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


class TestTrie(TempDirTestCase):
    def test_word_exists(self):
        root = toy_trie()
        self.assertTrue(sc.word_exists(root, 'will'))
        self.assertTrue(sc.word_exists(root, 'willow'))
        self.assertFalse(sc.word_exists(root, 'wil'))      # only a prefix
        self.assertFalse(sc.word_exists(root, 'wills'))
        self.assertFalse(sc.word_exists(root, 'Will'))     # exact case

    def test_empty_trie(self):
        root = sc.TrieNode('*')
        self.assertFalse(sc.word_exists(root, 'a'))
        self.assertEqual(sc.find_prefix(root, 'a'), (False, 0))
        self.assertEqual(sc.get_similar_words(root, 'a'), [])

    def test_find_prefix_counts_words(self):
        root = toy_trie()
        # will, willing, willow, wild, wile and wish start with "wi"
        self.assertEqual(sc.find_prefix(root, 'wi'), (False, 6))
        self.assertEqual(sc.find_prefix(root, 'will'), (True, 3))
        self.assertEqual(sc.find_prefix(root, 'xyz'), (False, 0))

    def test_get_all_children_finds_every_word(self):
        root = toy_trie()
        self.assertEqual(sorted(sc.get_all_children(root, '')),
                         sorted(TOY_WORDS))

    def test_get_all_children_goes_past_complete_words(self):
        root = toy_trie()
        node = root
        for ch in 'will':
            node = next(c for c in node.children if c.char == ch)
        self.assertEqual(sorted(sc.get_all_children(node, 'will')),
                         ['willing', 'willow'])

    def test_trie_dict_and_several_lists(self):
        first = self.write('first', 'cat\ndog\n')
        second = self.write('second', 'cow\n')
        root = sc.trie_dict(first)
        self.assertTrue(sc.word_exists(root, 'dog'))
        self.assertFalse(sc.word_exists(root, 'cow'))
        root = sc.load_dictionaries([first, second])
        for word in ['cat', 'dog', 'cow']:
            self.assertTrue(sc.word_exists(root, word))


class TestEditDistance(unittest.TestCase):
    def test_book_example(self):
        # Jurafsky & Martin: intention -> execution is 8 with these costs
        self.assertEqual(sc.edit_dist('intention', 'execution'), 8)

    def test_costs(self):
        self.assertEqual(sc.edit_dist('cat', 'cat'), 0)
        self.assertEqual(sc.edit_dist('cat', 'cats'), 1)    # insertion
        self.assertEqual(sc.edit_dist('cats', 'cat'), 1)    # deletion
        self.assertEqual(sc.edit_dist('cat', 'cut'), 2)     # substitution
        self.assertEqual(sc.edit_dist('wioll', 'will'), 1)

    def test_empty_words(self):
        self.assertEqual(sc.edit_dist('', 'abc'), 3)
        self.assertEqual(sc.edit_dist('abc', ''), 3)
        self.assertEqual(sc.edit_dist('', ''), 0)

    def test_symmetric(self):
        for a, b in [('kitten', 'sitting'), ('hte', 'the'), ('a', 'xyz')]:
            self.assertEqual(sc.edit_dist(a, b), sc.edit_dist(b, a))


class TestSuggest(unittest.TestCase):
    def test_closest_first(self):
        self.assertEqual(sc.suggest(['wile', 'will', 'wild'], 'wioll'),
                         ['will', 'wile', 'wild'])

    def test_closer_word_replaces_further_ones(self):
        self.assertEqual(
            sc.suggest(['zzzzzz', 'yyyyyy', 'xxxxxx', 'cat'], 'cat'),
            ['cat', 'zzzzzz', 'yyyyyy'])

    def test_ties_keep_the_first(self):
        # "spelding" and "spelling" are both 1 from "speling"
        self.assertEqual(sc.suggest(['spelding', 'spelling'], 'speling')[0],
                         'spelding')

    def test_fewer_than_three(self):
        self.assertEqual(sc.suggest(['cat'], 'cta'), ['cat'])
        self.assertEqual(sc.suggest([], 'cta'), [])


class TestGetSimilarWords(unittest.TestCase):
    def setUp(self):
        self.root = toy_trie()

    def test_words_after_the_first_wrong_letter(self):
        # "wi" matches, "o" doesn't: every word starting with "wi"
        self.assertEqual(sorted(sc.get_similar_words(self.root, 'wioll')),
                         ['wild', 'wile', 'will', 'willing', 'willow',
                          'wish'])

    def test_matched_part_is_a_candidate(self):
        self.assertEqual(sc.get_similar_words(self.root, 'abandonx'),
                         ['abandon', 'abandoned'])

    def test_whole_word_matches_but_isnt_a_word(self):
        self.assertEqual(sorted(sc.get_similar_words(self.root, 'wil')),
                         ['wild', 'wile', 'will', 'willing', 'willow'])

    def test_first_letter_not_in_trie(self):
        self.assertEqual(sorted(sc.get_similar_words(self.root, 'xyz')),
                         sorted(TOY_WORDS))


class TestEndings(unittest.TestCase):
    def setUp(self):
        self.root = toy_trie()

    def check(self, words, expected):
        for word in words:
            self.assertEqual(sc.is_spelled_correctly(self.root, word),
                             expected, word)

    def test_dictionary_words(self):
        self.check(['help', 'will', 'cat'], True)

    def test_regular_endings(self):
        self.check(['helps', 'helped', 'helping', 'helper', 'cities',
                    'taxes', 'wishes', 'hoped', 'hoping', 'seeing',
                    'bigger', 'biggest', 'running', 'happier', 'happiest',
                    'happily', 'finally', 'simply', 'selves', 'knives',
                    'women', 'abandons'], True)

    def test_possessives_and_contractions(self):
        self.check(["people's", "child's", "they're", "they'll", "don't",
                    "won't"], True)

    def test_irregular_forms(self):
        self.check(['has', 'children', "children's"], True)

    def test_spelling_checks_reject_misspellings(self):
        self.check(['taxs', 'wishs', 'hopeing', 'hopeed', 'biger', 'runed',
                    'helpd', 'thier'], False)

    def test_known_limits(self):
        # Accepted although misspelled (README "Limitations")
        self.check(['childs', 'happyly'], True)

    def test_base_forms(self):
        self.assertIn('city', sc.base_forms('cities'))
        self.assertIn('plan', sc.base_forms('planned'))
        self.assertIn('make', sc.base_forms('making'))
        self.assertIn('have', sc.base_forms('has'))
        self.assertIn('do', sc.base_forms("don't"))
        self.assertIn('child', sc.base_forms("children's"))
        self.assertNotIn('i', sc.base_forms('is'))      # stem too short


class TestCapitals(unittest.TestCase):
    def setUp(self):
        self.root = toy_trie()

    def test_capitals_checked_in_lower_case(self):
        for word in ['This', 'THIS', 'Help', 'HELPED']:
            self.assertTrue(sc.is_spelled_correctly(self.root, word), word)

    def test_lower_case_name_is_flagged(self):
        self.assertTrue(sc.is_spelled_correctly(self.root, 'America'))
        self.assertFalse(sc.is_spelled_correctly(self.root, 'america'))

    def test_match_case(self):
        self.assertEqual(sc.match_case('will', 'wioll'), 'will')
        self.assertEqual(sc.match_case('will', 'Wioll'), 'Will')
        self.assertEqual(sc.match_case('will', 'WIOLL'), 'WILL')
        self.assertEqual(sc.match_case('America', 'amercia'), 'America')

    def test_correct_keeps_capitals(self):
        self.assertEqual(sc.correct(self.root, 'wioll')[0], 'will')
        self.assertEqual(sc.correct(self.root, 'Wioll')[0], 'Will')
        self.assertEqual(sc.correct(self.root, 'WIOLL')[0], 'WILL')
        self.assertEqual(sc.correct(self.root, 'Amercia')[0], 'America')


class TestFindingWords(TempDirTestCase):
    def words(self, line):
        return [m.group() for m in sc.WORD_PATTERN.finditer(line)
                if sc.is_checkable(m.group())]

    def test_punctuation_and_full_stops(self):
        self.assertEqual(self.words('Mr. Chief Justice, my friends,'),
                         ['Mr', 'Chief', 'Justice', 'my', 'friends'])
        self.assertEqual(self.words('the U.S.A. e.g. order: credits;'),
                         ['the', 'U', 'S', 'A', 'e', 'g', 'order',
                          'credits'])

    def test_dashes_and_hyphens(self):
        self.assertEqual(self.words('itself—nameless self-seekers'),
                         ['itself', 'nameless', 'self', 'seekers'])

    def test_apostrophes(self):
        self.assertEqual(self.words("he's 'going to the beach'"),
                         ["he's", 'going', 'to', 'the', 'beach'])
        self.assertEqual(self.words("citizens' rights, don't, it’s"),
                         ['citizens', 'rights', "don't", 'it’s'])

    def test_numbers_skipped(self):
        self.assertEqual(self.words('48 States, 3.5, 1,000, 1930s, 3rd'),
                         ['States'])

    def test_tokenize_file(self):
        self.assertEqual(
            sc.tokenize(os.path.join(FILES, 'test_tokenize.txt')),
            ['My', 'name', 'is', 'jake', 'I', 'like', 'to', 'eat', 'good',
             'food'])

    def test_show_in_line(self):
        line = 'short line with tyop\n'
        start = line.index('tyop')
        self.assertEqual(sc.show_in_line(line, start, start + 4),
                         'short line with tyop')
        long_line = 'x' * 50 + ' tyop ' + 'y' * 50
        start = long_line.index('tyop')
        shown = sc.show_in_line(long_line, start, start + 4)
        self.assertTrue(shown.startswith('...') and shown.endswith('...'))
        self.assertIn('tyop', shown)
        self.assertLessEqual(len(shown), 2 * sc.CONTEXT + 4 + 6)


class TestMakeBrownWords(unittest.TestCase):
    # new_words() with made-up counts, so these don't need NLTK
    def test_new_words(self):
        root = toy_trie()
        counts = collections.Counter({
            'near': 5,          # new: kept
            'Chicago': 3,       # new name: kept
            "Chicago's": 2,     # left out: the rules handle 's
            'runing': 1,        # seen once: left out
            'helped': 9,        # the rules accept it already
            'The': 50,          # "the"? not in TOY_WORDS, so kept
            'cat': 7,           # already a word
            '1930s': 4,         # has digits
            '--': 9,            # not a word
        })
        self.assertEqual(make_brown_words.new_words(counts, root),
                         ['Chicago', 'near', 'The'])


# --- The real word lists ---

class TestRealDictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = sc.load_dictionaries(sc.DEFAULT_DICTS)

    def test_sizes(self):
        with open(sc.DEFAULT_DICT) as f:
            self.assertEqual(sum(1 for _ in f), 235886)
        self.assertTrue(os.path.isfile(sc.BROWN_DICT))

    def test_common_words(self):
        text = ('This is a day of national consecration and I am certain '
                'that my fellow Americans expect that on my induction into '
                "the Presidency I will address them. It has helped people's "
                "children, and they're near the box. Don't worry")
        for word in text.replace('.', '').replace(',', '').split():
            self.assertTrue(sc.is_spelled_correctly(self.root, word), word)

    def test_brown_words_fill_gaps(self):
        for word in ['near', 'box', 'kid', 'proud', 'okay', 'Chicago']:
            self.assertTrue(sc.is_spelled_correctly(self.root, word), word)

    def test_sample_typos(self):
        for typo, fix in [('wioll', 'will'), ('yesterdy', 'yesterday'),
                          ('welfure', 'welfare'), ('Saterday', 'Saturday'),
                          ('alredy', 'already'), ('abandonx', 'abandon')]:
            self.assertFalse(sc.is_spelled_correctly(self.root, typo), typo)
            self.assertEqual(sc.correct(self.root, typo)[0], fix, typo)


class TestCommandLine(TempDirTestCase):
    def run_script(self, *args, cwd=HERE):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, 'spellchecker.py'), *args],
            capture_output=True, text=True, cwd=cwd)

    def test_short1(self):
        out = self.path('out.txt')
        result = self.run_script(os.path.join('files', 'short1.txt'),
                                 '-o', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Line 1: changing wioll to will', result.stdout)
        self.assertIn('Line 3: changing yesterdy to yesterday',
                      result.stdout)
        self.assertIn('Line 4: changing welfure to welfare', result.stdout)
        self.assertIn('Changed 3 words.', result.stdout)
        with open(os.path.join(FILES, 'short1.txt')) as f:
            expected = (f.read().replace('wioll', 'will')
                        .replace('yesterdy', 'yesterday')
                        .replace('welfure', 'welfare'))
        with open(out) as f:
            self.assertEqual(f.read(), expected)   # layout kept exactly

    def test_text_1_keeps_layout(self):
        out = self.path('out.txt')
        result = self.run_script(os.path.join(FILES, 'text_1.txt'),
                                 '-o', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Changed 1 word.', result.stdout)
        with open(os.path.join(FILES, 'text_1.txt')) as f:
            expected = f.read().replace('preeminently', 'preen')
        with open(out) as f:
            self.assertEqual(f.read(), expected)

    def test_saved_outputs_are_up_to_date(self):
        for name in ['short1.txt', 'text_1.txt', 'apostrophe_test.txt']:
            out = self.path(name)
            result = self.run_script(os.path.join(FILES, name), '-o', out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out) as made, \
                    open(os.path.join(FILES, 'corrected_' + name)) as saved:
                self.assertEqual(made.read(), saved.read(), name)

    def test_abbreviations_and_quotes(self):
        text = self.write('in.txt', 'Dr. Smith’s visit, etc.\nDr Smith\n')
        result = self.run_script(text)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Line 1', result.stdout)
        self.assertIn('Line 2: changing Dr to', result.stdout)

    def test_default_output_and_replacing(self):
        text = self.write('notes.txt', 'teh cat\n')
        first = self.run_script(text, cwd=self.tmp)
        self.assertIn('Saved the corrected text to', first.stdout)
        self.assertTrue(os.path.isfile(self.path('corrected_notes.txt')))
        second = self.run_script(text, cwd=self.tmp)
        self.assertIn('Replaced', second.stdout)

    def test_own_word_list(self):
        words = self.write('words', 'the\ncat\n')
        text = self.write('in.txt', 'teh cat\n')
        result = self.run_script(text, '-d', words, '-o', self.path('o'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('changing teh to the', result.stdout)

    def test_no_arguments_prints_help(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage: spellchecker.py', result.stdout)

    def test_errors(self):
        text = self.write('in.txt', 'cat\n')
        with open(self.path('binary'), 'wb') as f:
            f.write(bytes([0xff, 0xfe, 0xab]))
        cases = [
            (['nope.txt'], 2, "input file 'nope.txt' not found"),
            ([text, '-d', 'nope'], 2, "dictionary 'nope' not found"),
            ([text, '-o', text], 2, 'same as the input'),
            ([text, '-o', sc.DEFAULT_DICT], 2, 'same as the dictionary'),
            ([text, '-o', os.path.join('nodir', 'x')], 2, 'does not exist'),
            ([self.path('binary'), '-o', self.path('o')], 1,
             'is not a UTF-8 text file'),
        ]
        for args, code, message in cases:
            result = self.run_script(*args)
            self.assertEqual(result.returncode, code, args)
            self.assertIn(message, result.stderr, args)


if __name__ == '__main__':
    unittest.main()
