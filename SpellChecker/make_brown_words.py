# Make files/brown_words: the words in the Brown corpus that files/words
# doesn't already cover, so the spell checker can use them as a second word
# list without needing NLTK itself.
#
# The Brown corpus is about a million words of American English from 1961
# (news, fiction, science, ...), and comes with NLTK. Unlike files/words it
# has inflected forms, names and modern words (near, box, kid, Chicago).
#
# A word is kept if:
#   - it is a word as spellchecker.py finds them (letters, with apostrophes
#     only inside), with no digits,
#   - it appears at least MIN_COUNT times, which leaves out most one-off
#     typos in the corpus ("runing" and "publically" each appear once),
#   - and files/words with the ending and case rules doesn't accept it
#     already, so the list only has what is new.
# A possessive or contraction ("Atlanta's") is also left out when the word
# it is made from ("Atlanta") is kept, since the rules handle the ending.
#
# Run from this folder (needs nltk and its 'brown' data, see the top-level
# README.md):
#     python3 make_brown_words.py

import collections
import os
import sys

import spellchecker as sc

MIN_COUNT = 2
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'files', 'brown_words')


def brown_word_counts():
    try:
        from nltk.corpus import brown
        return collections.Counter(brown.words())
    except (ImportError, LookupError) as e:
        sys.exit("Error: this needs nltk and its 'brown' data (see the "
                 "top-level README.md): %s" % str(e).strip().splitlines()[0])


def new_words(counts, root):
    kept = set()
    for word, count in counts.items():
        if count < MIN_COUNT:
            continue
        if not sc.WORD_PATTERN.fullmatch(word) or not sc.is_checkable(word):
            continue
        if sc.is_spelled_correctly(root, word):
            continue
        kept.add(word)
    # Leave out "Atlanta's" when "Atlanta" is kept
    for word in list(kept):
        for clitic in sc.CLITICS:
            if word.endswith(clitic) and word[:-len(clitic)] in kept:
                kept.discard(word)
                break
    return sorted(kept, key=lambda w: (w.lower(), w))


def main():
    counts = brown_word_counts()
    root = sc.trie_dict(sc.DEFAULT_DICT)
    words = new_words(counts, root)
    with open(OUTPUT, 'w') as f:
        f.write('\n'.join(words) + '\n')
    print('Saved %i words to %s.' % (len(words), os.path.relpath(OUTPUT)))


if __name__ == '__main__':
    main()
