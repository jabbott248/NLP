# Goal: Create a basic spellchecker
from typing import Tuple
import argparse
import os
import sys
import re

# Trie Node system, used to store our dictionary
class TrieNode(object):
    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.word_finished = False  # Determines if it is the last character of the word
        self.counter = 1            # How many times this character appeared in the insertion process

# inserts a word into the trie structure
def insert(root, word: str):
    # defines a pointer to the root of the trie
    cur_node = root 
    # for every character in the given word:
    for char in word:
        found_in_child = False
        # try to find same char in the children of the current node
        for child in cur_node.children:
            if child.char == char: # match
                child.counter += 1      # increase the amount of times this char appeard in the insertion process
                cur_node = child        # point the current node to this child
                found_in_child = True   # set found_in_child to true
                break                   # can stop searching nodes children for matches as one has already been found

        if not found_in_child:  # If a match has not been found
            new_node = TrieNode(char)           # create a new node representing the current char
            cur_node.children.append(new_node)  # appends new node to current nodes children
            cur_node = new_node                 # point node to the newly created node

    cur_node.word_finished = True

# returns the amount of times a prefix occurs in a trie, and a bool if it is a full word
def find_prefix(root, prefix: str) -> Tuple[bool, int]:
    cur_node = root  # pointer to the root of the trie
    if not root.children:   # If the root has no children, trie is empty so the prefix does not exist
        return False, 0

    for char in prefix:
        char_not_found = True
        for child in cur_node.children:
            if child.char == char: # match of current char
                char_not_found = False 
                cur_node = child # move the pointer to matched child
                break

        if char_not_found:  # not all chars were found, prefix occurs 0 times
            return False, 0

    # since we passed previous return statement, we have found a full match
    # check if this is the end of a word:
    return cur_node.word_finished, cur_node.counter

# checks if a word exists in the dictionary (exactly as spelled)
def word_exists(root, word: str) -> bool:
    ret_tuple = find_prefix(root, word)
    return ret_tuple[0]


## WORD ENDINGS
# dictionary/words has only the base form of most words ("help", not "helps",
# "helped", "helping"), so a word is also accepted if taking a common
# ending off it leaves a dictionary word. Each rule is (ending, what to put
# back): "cities" -> "citi" + "y" = "city". DOUBLE marks endings that may
# follow a doubled consonant ("planned" -> "plan", "bigger" -> "big").
DOUBLE = 'double'
ENDING_RULES = [
    ('ies', ['y']),             # cities -> city, carries -> carry
    ('ves', ['f', 'fe']),       # selves -> self, knives -> knife
    ('men', ['man']),           # women -> woman, salesmen -> salesman
    ('es', ['']),               # taxes -> tax, wishes -> wish
    ('s', ['']),                # friends -> friend, helps -> help
    ('ied', ['y']),             # multiplied -> multiply
    ('ed', ['', 'e', DOUBLE]),  # helped -> help, believed -> believe, planned -> plan
    ('ing', ['', 'e', DOUBLE]), # helping -> help, making -> make, putting -> put
    ('ier', ['y']),             # happier -> happy
    ('iest', ['y']),            # happiest -> happy
    ('er', ['', 'e', DOUBLE]),  # stronger -> strong, larger -> large, bigger -> big
    ('est', ['', 'e', DOUBLE]), # strongest -> strong, largest -> large
    ('ily', ['y']),             # primarily -> primary, easily -> easy
    ('ly', ['', 'le']),         # finally -> final, simply -> simple
]

# Common irregular forms that don't follow the rules and aren't in
# dictionary/words themselves (most are: "went", "said", "men" and "mice" are)
IRREGULAR_FORMS = {
    'has': 'have', 'became': 'become', 'began': 'begin', 'bled': 'bleed',
    'blew': 'blow', 'forgave': 'forgive', 'forgiven': 'forgive',
    'heard': 'hear', 'hid': 'hide', 'held': 'hold', 'paid': 'pay',
    'awoken': 'awake', 'woken': 'wake', 'withdrew': 'withdraw',
    'undertaken': 'undertake', 'undergone': 'undergo',
    'overcame': 'overcome', 'foresaw': 'foresee', 'foreseen': 'foresee',
    'overheard': 'overhear', 'overthrew': 'overthrow',
    'overthrown': 'overthrow', 'repaid': 'repay',
    'children': 'child', 'feet': 'foot', 'geese': 'goose', 'lice': 'louse',
    "can't": 'can', "won't": 'will', "shan't": 'shall', "o'clock": 'clock',
}

# Possessive 's and the endings of contractions, taken off before the
# other endings: people's -> people, don't -> do, they're -> they,
# children's -> children (-> child). (A plural possessive like citizens'
# doesn't need a rule: the apostrophe isn't part of the word, see
# WORD_PATTERN.)
CLITICS = ["'s", "n't", "'re", "'ve", "'ll", "'d", "'m"]

# Spelling rules for which stems an ending can follow
VOWEL_ENDINGS = ('ing', 'ed', 'er', 'est')
KEEP_E = ('ee', 'oe', 'ye')

VOWELS = 'aeiou'

# Short words that end consonant-vowel-consonant (run, sit, big) double
# their last letter before -ing, -ed, -er and -est. Only 3-letter words are
# checked, because longer ones often don't double (visiting, opening); w, x
# and y are never doubled (bowing, fixing, toying).
def must_double(stem: str) -> bool:
    return (len(stem) == 3
            and stem[0] not in VOWELS and stem[1] in VOWELS
            and stem[2] not in VOWELS + 'wxy')

def ending_fits(stem: str, ending: str) -> bool:
    if ending in ('ier', 'iest'):
        # happier -> happy, but "thier" isn't "thy" + -ier
        return len(stem) >= 3
    if ending == 'es':
        # -es only after a hissing sound or o: taxes, wishes, heroes
        return stem.endswith(('s', 'x', 'z', 'ch', 'sh', 'o'))
    if ending == 's':
        # and not plain -s after one: "taxs", "wishs" ("ch" can be a k
        # sound, as in "stomachs", so it is allowed)
        return not stem.endswith(('s', 'x', 'z', 'sh'))
    return True

# Returns the possible base forms of a word: what it could be with one
# ending taken off (after a possessive or contraction ending, if there is
# one).
def base_forms(word: str) -> list[str]:
    forms = []
    words = [word]
    for clitic in CLITICS:
        if word.endswith(clitic) and len(word) > len(clitic):
            words.append(word[:-len(clitic)])
    for w in words:
        if w in IRREGULAR_FORMS:
            forms.append(IRREGULAR_FORMS[w])
        for ending, replacements in ENDING_RULES:
            if not w.endswith(ending):
                continue
            stem = w[:-len(ending)]
            if len(stem) < 2:       # "is" isn't "i" + "s"
                continue
            if not ending_fits(stem, ending):
                continue
            for replacement in replacements:
                if (replacement == '' and ending in VOWEL_ENDINGS
                        and stem.endswith('e') and stem[-2:] not in KEEP_E):
                    # The e is dropped before these endings: "hoping",
                    # not "hopeing" (but "seeing", "hoeing", "dyeing")
                    continue
                if (replacement == '' and ending in VOWEL_ENDINGS
                        and must_double(stem)):
                    # "running", not "runing"
                    continue
                if replacement == DOUBLE:
                    # planned -> plann -> plan
                    if len(stem) >= 3 and stem[-1] == stem[-2]:
                        forms.append(stem[:-1])
                else:
                    forms.append(stem + replacement)
    forms.extend(words[1:])         # the word without its 's or n't
    return forms


## CHECKING A WORD
# A word is spelled correctly if it is in the dictionary, or if it is with
# one of the common endings above taken off. Case: a word with capital
# letters is also accepted if its lower-case form is a word, so "This" at
# the start of a sentence and "THIS" in a heading are fine, but "america"
# isn't accepted just because "America" is.
def case_forms(word: str) -> list[str]:
    forms = [word]
    if word != word.lower():
        forms.append(word.lower())
    return forms

def is_spelled_correctly(root, word: str) -> bool:
    for form in case_forms(word):
        if word_exists(root, form):
            return True
        for base in base_forms(form):
            if word_exists(root, base):
                return True
    return False

# Matches the case of a suggestion to the word it replaces: a suggestion
# for "Wioll" is "Will", and for "WIOLL" it is "WILL". Suggestions that
# already have capitals (names) are left as they are.
def match_case(suggestion: str, word: str) -> str:
    if suggestion != suggestion.lower():
        return suggestion
    if len(word) > 1 and word.isupper():
        return suggestion.upper()
    if word[0].isupper():
        return suggestion[0].upper() + suggestion[1:]
    return suggestion

# Finds the candidate replacements for a misspelled word, in the word's own
# case and, for a word with capitals, in lower case too ("Teh" gets the
# candidates for "teh" as well as for "Teh"), and returns the closest ones.
def correct(root, word: str) -> list[str]:
    candidates = []
    for form in case_forms(word):
        for candidate in get_similar_words(root, form):
            candidate = match_case(candidate, word)
            candidates.append(candidate)
    # The same candidate can be found twice ("Will" the name and "will"
    # turned into "Will"); keep the first
    candidates = list(dict.fromkeys(candidates))
    return suggest(candidates, word)

# creates a pointer to the root of a trie created by all words in a dictionary
# (or adds the words to an existing trie, to load several word lists)
def trie_dict(fname, root=None):
    if root is None:
        root = TrieNode('*')
    with open(fname) as f:
        for line in f:
            words = line.split()
            for word in words:
                insert(root, word)
    return root

# Find candidate replacements for a misspelled word: follow the trie for as
# long as the word's letters match, then return every dictionary word that
# starts with the part that matched (including that part itself, if it is a
# word). For "wioll", "wi" matches, so the candidates are "wi" words; for
# "abandonx" they are "abandon" and every word starting with it.
def get_similar_words(root, word: str) -> list[str]:
    # follow tree as far as possible with prefixes that do exist
    cur_node = root  # pointer to the root of the trie
    prefix = ''
    if not root.children:   # If the root has no children, trie is empty so the prefix does not exist
        return []           # return the empty array

    for char in word:                       # iterate through each charcter of the input word
        char_not_found = True
        for child in cur_node.children:     # iterate through each child of current char's child
            if child.char == char:
                char_not_found = False
                prefix = prefix + char  # add current char to the prefix string
                cur_node = child        # move the pointer to matched child
                break                   # continue to the next node

        if char_not_found:      # stop at the first letter that doesn't match
            break

    # Either a letter didn't match, or every letter matched but the word
    # isn't marked as finished (e.g. "Nor" is the start of "Norway"). In
    # both cases the candidates are the words below the node reached.
    candidates = []
    if cur_node.word_finished:      # the matched part is a word itself
        candidates.append(prefix)
    candidates.extend(get_all_children(cur_node, prefix))
    return candidates

# helper function to return all the words below a node in a trie structure
# (not counting the node's own word)
def get_all_children(root, prefix: str) -> list[str]:
    ret_lst = []
    for child in root.children:
        new_prefix = prefix + child.char
        if child.word_finished:         # this child ends a word...
            ret_lst.append(new_prefix)
        # ...and longer words can still continue below it ("will" ->
        # "willing"), so always keep going down
        ret_lst.extend(get_all_children(child, new_prefix))

    return ret_lst

# takes a word and given a list of words return the closest words
# Keeps the N_SUGGESTIONS candidates with the smallest edit distance to
# source, closest first. A candidate replaces the last one kept only if it
# is strictly closer, so among words the same distance away, the one that
# came first (alphabetical order in the dictionary) is kept.
N_SUGGESTIONS = 3

def suggest(words: list[str], source: str) -> list[str]:
    ret_lst = []    # (distance, word) pairs, closest first
    for word in words:
        dist = edit_dist(source, word)
        if len(ret_lst) < N_SUGGESTIONS:
            ret_lst.append((dist, word))
        elif dist < ret_lst[-1][0]:     # closer than the furthest one kept
            ret_lst[-1] = (dist, word)
        else:
            continue
        # sort by distance; sort() is stable, so ties keep their order
        ret_lst.sort(key=lambda pair: pair[0])

    return [word for dist, word in ret_lst]

# TOKENIZATION
# Instead of splitting the text into tokens and putting them back together,
# the words are found in each line where they are, so that everything
# between them (spaces, punctuation, line breaks) is kept exactly as it was.
#
# A word is a run of letters, with apostrophes allowed inside it (he's,
# don't, o'clock). Everything else separates words: spaces, punctuation,
# hyphens and dashes ("self-seekers" is checked as "self" and "seekers",
# "itself—nameless" as "itself" and "nameless"), and full stops ("Mr." is
# "Mr", "U.S.A." is "U", "S" and "A"). An apostrophe at the start or end of
# a word is a quote mark, not part of it ('going' is "going"), which also
# takes the apostrophe off a plural possessive (citizens' -> citizens).
# The curly apostrophe (’) counts as an apostrophe too.
#
# Runs of letters and digits are matched together so that "1930s" or "3rd"
# isn't split into "s" and "rd"; anything with a digit in it is then
# skipped (see is_checkable).
LETTER = r"[^\W_]"     # a letter or digit, in any alphabet
WORD_PATTERN = re.compile(LETTER + r"+(?:['’]" + LETTER + r"+)*")

def is_checkable(word: str) -> bool:
    return not any(ch.isdigit() for ch in word)

# Abbreviations that aren't in dictionary/words, accepted when a full stop
# follows them ("Dr." is fine, "Dr" on its own is checked as a word)
ABBREVIATIONS = {
    'Dr', 'St', 'Jr', 'Sr', 'Mt', 'Ave', 'Blvd', 'Prof', 'Gov', 'Sen',
    'Rep', 'Gen', 'Capt', 'Lt', 'Col', 'Sgt', 'Rev', 'Hon', 'Inc', 'Ltd',
    'Corp', 'Bros', 'etc', 'vs', 'approx', 'dept', 'esp', 'misc',
    'Jan', 'Feb', 'Apr', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec',
}

# param: fname, file name to be read
# returns: a list of the words in the file, in order (punctuation and
# numbers are left out)
def tokenize(fname):
    tokens = []
    with open(fname) as fp:
        for line in fp:
            for match in WORD_PATTERN.finditer(line):
                if is_checkable(match.group()):
                    tokens.append(match.group())
    return tokens

# EDIT DISTANCE, from book figure 3.25
cost_insertion = 1
cost_deletion = 1
cost_substitution = 2
# param: t(arget), intended word
# param: s(ource), actual word
# returns: minimum edit distance from target to source
def edit_dist(s,t):
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]

    # intialize source compared to the empty string
    for i in range(1, rows):
        dist[i][0] = i

    # intialize target compared to the empty string
    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                sub_cost = 0
            else:
                sub_cost = cost_substitution
            dist[row][col] = min(dist[row-1][col] + cost_deletion,      # deletion
                                 dist[row][col-1] + cost_insertion,     # insertion
                                 dist[row-1][col-1] + sub_cost)         # substitution
    # helper code to print out matrices
    # for r in range(rows):
    #   print(dist[r])
    # bottom-right cell: the distance between the whole of s and all of t
    # (dist[-1][-1] rather than dist[row][col], which don't exist when s or
    # t is empty and a loop never runs)
    return dist[-1][-1]

# INTERFACE
# Folders next to this script, so the defaults are found no matter which
# folder the script is run from:
#   dictionary/   the word lists
#   samples/      sample texts to check
#   corrected/    where corrected texts are saved by default
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_DIR = os.path.join(SCRIPT_DIR, 'dictionary')
SAMPLES_DIR = os.path.join(SCRIPT_DIR, 'samples')
CORRECTED_DIR = os.path.join(SCRIPT_DIR, 'corrected')
DEFAULT_DICT = os.path.join(DICTIONARY_DIR, 'words')
# Words from the Brown corpus that dictionary/words doesn't cover (names,
# modern words like "okay", and common words it is missing like "near"),
# made by make_brown_words.py
BROWN_DICT = os.path.join(DICTIONARY_DIR, 'brown_words')
DEFAULT_DICTS = [DEFAULT_DICT, BROWN_DICT]
OUTPUT_PREFIX = 'corrected_'

def default_output_name(input_fname):
    # samples/short1.txt -> corrected/corrected_short1.txt, in the corrected
    # folder next to this script, wherever the input is
    name = os.path.basename(input_fname)
    return short_path(os.path.join(CORRECTED_DIR, OUTPUT_PREFIX + name))

def short_path(fname):
    # Show a path relative to the current folder if it's inside it or one
    # folder up (samples/ -> ../corrected/...), and the full path otherwise
    full = os.path.abspath(fname)
    rel = os.path.relpath(full)
    if not rel.startswith(os.path.join(os.pardir, os.pardir)):
        return rel
    return full

def read_text_file(read, fname):
    # Call read(fname), stopping with a clear message if the file isn't text
    try:
        return read(fname)
    except UnicodeDecodeError:
        sys.exit("Error: '%s' is not a UTF-8 text file" % fname)

# Shows a misspelled word in its line, cut down to CONTEXT characters on
# each side if the line is long
CONTEXT = 25

def show_in_line(line: str, start: int, end: int) -> str:
    line = line.rstrip('\n')
    left = max(0, start - CONTEXT)
    right = min(len(line), end + CONTEXT)
    return (('...' if left > 0 else '') + line[left:right].strip()
            + ('...' if right < len(line) else ''))

# Loads one or more word lists into one trie
def load_dictionaries(dict_fnames):
    root = TrieNode('*')
    for fname in dict_fnames:
        read_text_file(lambda f: trie_dict(f, root), fname)
    return root

def run(input_fname, output_fname, dict_fnames):
    # root of dictionary
    root = load_dictionaries(dict_fnames)
    lines = read_text_file(read_lines, input_fname)

    # Go through the text line by line, replacing each misspelled word where
    # it is, and printing it with its line
    n_changed = 0
    corrected_lines = []
    for line_no, line in enumerate(lines, 1):
        def fix(match):
            nonlocal n_changed
            word = match.group()
            if not is_checkable(word):
                return word
            check = word.replace('’', "'")
            if is_spelled_correctly(root, check):
                return word
            if check in ABBREVIATIONS and line[match.end():match.end() + 1] == '.':
                return word
            suggestions = correct(root, check)
            context = show_in_line(line, match.start(), match.end())
            if len(suggestions) > 0: # if there is a spell correction:
                print('Line %i: changing %s to %s' % (line_no, word, suggestions[0]))
                print('    ' + context)
                n_changed += 1
                return suggestions[0]               # write correct spelling
            print('Line %i: no suggestions for %s' % (line_no, word))
            print('    ' + context)
            return word                             # write same word
        corrected_lines.append(WORD_PATTERN.sub(fix, line))

    replacing = os.path.exists(output_fname)
    with open(output_fname, 'w') as f:
        f.writelines(corrected_lines)
    print('Changed %i word%s.' % (n_changed, '' if n_changed == 1 else 's'))
    if replacing:
        print('Replaced %s with the corrected text.' % output_fname)
    else:
        print('Saved the corrected text to %s.' % output_fname)

def read_lines(fname):
    with open(fname) as f:
        return f.readlines()

def interface():
    parser = argparse.ArgumentParser(
        prog='spellchecker.py',
        description='Find the misspelled words in a text file, replace each '
                    'one with\nthe closest dictionary word, and save the '
                    'corrected text to a new file.',
        epilog='examples:\n'
               '  python3 spellchecker.py samples/short1.txt             '
               '-> corrected/corrected_short1.txt\n'
               '  python3 spellchecker.py samples/short1.txt -o out.txt  '
               '-> out.txt\n'
               '  python3 spellchecker.py samples/short1.txt -d mywords  '
               'check against the words in mywords only',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', metavar='TEXTFILE',
                        help='text file to check')
    parser.add_argument('-o', '--output', metavar='OUTFILE',
                        help='where to save the corrected text (default: '
                             'the input file\'s name with %s in front, in '
                             'the corrected folder next to this script). An '
                             'existing file is replaced.' % OUTPUT_PREFIX)
    parser.add_argument('-d', '--dict', metavar='WORDLIST', dest='dict_fnames',
                        action='append',
                        help='word list to check against, one word per line. '
                             'Give -d more than once to use several lists '
                             '(default: dictionary/words and '
                             'dictionary/brown_words next to this script)')
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    input_fname = args.input
    output_fname = args.output
    if output_fname is None:
        output_fname = default_output_name(input_fname)
        os.makedirs(CORRECTED_DIR, exist_ok=True)   # in case it was deleted
    dict_fnames = args.dict_fnames
    if dict_fnames is None:
        dict_fnames = [short_path(f) for f in DEFAULT_DICTS]

    # Check the files before doing any work, so mistakes fail fast
    if not os.path.isfile(input_fname):
        parser.error("input file '%s' not found" % input_fname)
    for dict_fname in dict_fnames:
        if not os.path.isfile(dict_fname):
            parser.error("dictionary '%s' not found" % dict_fname)
    for other, what in ([(input_fname, 'input')]
                        + [(d, 'dictionary') for d in dict_fnames]):
        if os.path.abspath(output_fname) == os.path.abspath(other):
            parser.error("output file '%s' is the same as the %s file"
                         % (output_fname, what))
    output_dir = os.path.dirname(output_fname)
    if output_dir and not os.path.isdir(output_dir):
        parser.error("folder '%s' for the output file does not exist"
                     % output_dir)

    run(input_fname, output_fname, dict_fnames)

if __name__ == "__main__":
    interface()
