# Goal: Create a basic spellchecker
from typing import List, Tuple
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
# files/words has only the base form of most words ("help", not "helps",
# "helped", "helping"), so a word is also accepted if taking a common
# ending off it leaves a dictionary word. Each rule is (ending, what to put
# back): "cities" -> "citi" + "y" = "city". DOUBLE marks endings that may
# follow a doubled consonant ("planned" -> "plan", "bigger" -> "big").
DOUBLE = 'double'
ENDING_RULES = [
    ("'s", ['']),               # possessive: people's -> people
    ("'", ['']),                # plural possessive: citizens' -> citizens
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
# files/words themselves (most are: "went", "said", "men" and "mice" are)
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
}

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
# ending taken off (after a possessive 's or ', if there is one).
def base_forms(word: str) -> list[str]:
    forms = []
    if word in IRREGULAR_FORMS:
        forms.append(IRREGULAR_FORMS[word])
    words = [word]
    for possessive in ["'s", "'"]:
        if word.endswith(possessive) and len(word) > len(possessive):
            words.append(word[:-len(possessive)])
    for w in words:
        for ending, replacements in ENDING_RULES[2:]:   # not the possessives
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
    forms.extend(words[1:])         # the word without its possessive
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
def trie_dict(fname):
    root = TrieNode('*')
    with open(fname) as f:
        for line in f:
            words = line.split()
            for word in words:
                insert(root, word)
    return root

def add_punction_to_trie(root):
    insert(root, ',')
    insert(root, '.')
    insert(root, '\\')
    insert(root, '\'')
    insert(root, '"')
    insert(root, '/')
    insert(root, '?')
    insert(root, '*')
    insert(root, '!')
    insert(root, '@')
    insert(root, '#')
    insert(root, '$')
    insert(root, '%')
    insert(root, '^')
    insert(root, '&')
    insert(root, '(')
    insert(root, ')')
    insert(root, '_')
    insert(root, '-')
    insert(root, '+')
    insert(root, '=')
    insert(root, '~')
    insert(root, '`')
    insert(root, ';')
    insert(root, '—')

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
# param: fname, file name to be read
# returns: a list of strings containing individual tokens
def tokenize(fname):
    with open(fname) as fp:
        tokens = []
        for line in fp:
            # get rid of unambiguous punctuation
            line = re.sub(r'([\\?!()\";/\\|`])', r' \1 ', line)
            # seperate out commas unless they're in numbers
            line = re.sub(r'([^0-9]),', r'\1 , ', line)
            line = re.sub(r',([^0-9])', r' , \1', line)
            # distinguish apostrophes from single quotes (single qoutes not preceded by a letter)
           # line = re.sub(r'([^\'])', r'\1 ', line)
           # line = re.sub(r"([^A-Za-z0-9])'", r"\1 '", line) # non letter followed by an apostrophe
            # pull off word-final clitics.
            # deal with periods at the end of words
            line = re.sub(r'(.)\.', r'\1 . ', line)
            tokens.extend(line.split())
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
    return dist[row][col]

# INTERFACE
# Folder this script is in, so the default dictionary is found no matter
# which folder the script is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DICT = os.path.join(SCRIPT_DIR, 'files', 'words')
OUTPUT_PREFIX = 'corrected_'

def default_output_name(input_fname):
    # files/short1.txt -> files/corrected_short1.txt (saved next to the input)
    folder, name = os.path.split(input_fname)
    return os.path.join(folder, OUTPUT_PREFIX + name)

def short_path(fname):
    # Show a path relative to the current folder if it's inside it, and the
    # full path otherwise
    cwd = os.getcwd()
    full = os.path.abspath(fname)
    if os.path.commonpath([cwd, full]) == cwd:
        return os.path.relpath(full)
    return full

def read_text_file(read, fname):
    # Call read(fname), stopping with a clear message if the file isn't text
    try:
        return read(fname)
    except UnicodeDecodeError:
        sys.exit("Error: '%s' is not a UTF-8 text file" % fname)

def run(input_fname, output_fname, dict_fname):
    # root of dictionary
    root = read_text_file(trie_dict, dict_fname)
    # add punctuation to dictionary
    add_punction_to_trie(root)

    tokens = read_text_file(tokenize, input_fname)
    replacing = os.path.exists(output_fname)
    with open(output_fname, 'w') as f:
        for word in tokens:
            if is_spelled_correctly(root, word):
                f.write(word + ' ')
            else:
                suggestions = correct(root, word)
                if len(suggestions) > 0: # if there is a spell correction:
                    print('Changing %s to %s'%(word,suggestions[0]))    # prints word and line
                    f.write(suggestions[0] + ' ')                             # write correct spelling
                else:
                    print('No suggestions for: %s'%(word))              # print no correction found
                    f.write(word + ' ')                                       # write same word
    print('Saved the corrected text to %s%s.'
          % (output_fname, ' (replaced the old file)' if replacing else ''))

    # put the file back together using regex

    # load a file whose name is given as a command-line argument.
    # look for words that are incorrectly spelled. prints out the line that the word appeared in
    # save its corrected version to a new file with the prefix `corrected_`.

def interface():
    parser = argparse.ArgumentParser(
        prog='spellchecker.py',
        description='Find the misspelled words in a text file, replace each '
                    'one with\nthe closest dictionary word, and save the '
                    'corrected text to a new file.',
        epilog='examples:\n'
               '  python3 spellchecker.py files/short1.txt             '
               '-> files/corrected_short1.txt\n'
               '  python3 spellchecker.py files/short1.txt -o out.txt  '
               '-> out.txt\n'
               '  python3 spellchecker.py files/short1.txt -d mywords  '
               'check against the words in mywords',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', metavar='TEXTFILE',
                        help='text file to check')
    parser.add_argument('-o', '--output', metavar='OUTFILE',
                        help='where to save the corrected text (default: '
                             'the input file\'s name with %s in front, in '
                             'the same folder as the input). An existing '
                             'file is replaced.' % OUTPUT_PREFIX)
    parser.add_argument('-d', '--dict', metavar='WORDLIST', dest='dict_fname',
                        help='dictionary to check against, one word per line '
                             '(default: files/words in this script\'s '
                             'folder)')
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    input_fname = args.input
    output_fname = args.output
    if output_fname is None:
        output_fname = default_output_name(input_fname)
    dict_fname = args.dict_fname
    if dict_fname is None:
        dict_fname = short_path(DEFAULT_DICT)

    # Check the files before doing any work, so mistakes fail fast
    if not os.path.isfile(input_fname):
        parser.error("input file '%s' not found" % input_fname)
    if not os.path.isfile(dict_fname):
        parser.error("dictionary '%s' not found" % dict_fname)
    for other, what in [(input_fname, 'input'), (dict_fname, 'dictionary')]:
        if os.path.abspath(output_fname) == os.path.abspath(other):
            parser.error("output file '%s' is the same as the %s file"
                         % (output_fname, what))
    output_dir = os.path.dirname(output_fname)
    if output_dir and not os.path.isdir(output_dir):
        parser.error("folder '%s' for the output file does not exist"
                     % output_dir)

    run(input_fname, output_fname, dict_fname)

def test_trie():
    root = trie_dict(DEFAULT_DICT)
    print(len(root.children))
    print(str(word_exists(root, 'abondon')))

def test_edit_distance(x,y):
    dist = edit_dist(x,y)
    print(dist)


def test_find_similar_words():
    root = trie_dict('words')
    example_word = 'abondon'
    if word_exists(root, example_word):
        print('word exists')
        return []
    else:
        res = get_similar_words(root, example_word)
        for word in res:
            print(word)
        return res

def test_tokenize():
    ret = tokenize('test_tokenize.txt')
    for word in ret:
        print(word)

if __name__ == "__main__":
    interface()
