Spell Checker
=============

What it does
------------
spellchecker.py is meant to read a text file, find the words that aren't
in its dictionary, replace each one with the closest dictionary word, and
save the corrected text to a new file whose name starts with
"corrected_". It prints each change it makes ("Changing wioll to
will").

It fixes all 3 misspellings in files/short1.txt. But it still flags far
too many correct words: files/text_1.txt has no misspellings, yet 252 of
its 2,080 words are flagged and changed, every one for the worse:
"President" becomes "Presley", "has" becomes "hash" and "efforts"
becomes "effort". Almost all of these are capitalised words and word
endings the dictionary doesn't have (Known issues 6 and 7).

How it works
------------
The steps below describe what the code is meant to do, and where it
does something different (see "Known issues" for details).

1. Load the dictionary into a trie (trie_dict, insert)
   files/words has 235,886 words, one per line. Each is added to a trie:
   a tree with one node per letter, where the path from the root spells
   a word and the last letter's node is marked word_finished. Words that
   start the same way share nodes, so "abandon" and "abandoned" share
   their first 7 letters. Each node's children are kept in a list and
   searched one by one. Building the trie takes about 0.4 seconds.

   add_punction_to_trie() also adds 25 punctuation marks as one-letter
   "words", so punctuation tokens count as spelled correctly.

2. Split the text into tokens (tokenize)
   Regular expressions put spaces around punctuation and the text is
   split on whitespace: "Mr. Chief Justice, my friends," becomes
   Mr . Chief Justice , my friends ,
   Commas inside numbers ("1,000") are kept. Apostrophes are left alone.

3. Check each token (word_exists, find_prefix)
   A token is spelled correctly if following its letters from the root
   ends on a node marked word_finished. The check is case-sensitive.

4. Find candidate replacements (get_similar_words, get_all_children)
   For a misspelled token, the trie is followed for as long as its
   letters match. The candidates are all the dictionary words that start
   with the part that matched, including that part itself if it is a
   word. For "wioll", "wi" matches and "o" doesn't, so the candidates are
   the 734 words starting with "wi". For "abandonx" they are "abandon"
   and the 6 words that start with it. If every letter matches but the
   token isn't a word ("Nor" is the start of "Norway"), the candidates
   are the words that continue it.

   get_all_children() collects the words below a node by walking every
   branch to the end. Each branch can hold several words ("will",
   "willing", "willingly"), so it keeps going past a complete word.

   This means a mistake at or before the letter where the token leaves
   the trie can't be fixed, because the right word is never a candidate
   (Known issue 16). A token whose first character isn't in the trie at
   all, like "48", gets the whole dictionary as candidates, which takes
   about half a second.

5. Pick the closest candidate (suggest, edit_dist)
   Each candidate is compared with the token by minimum edit distance
   (Jurafsky & Martin, figure 3.25): the cheapest way to turn one word
   into the other, where inserting or deleting a letter costs 1 and
   substituting one letter for another costs 2. "wioll" to "will" is 1
   (delete the o). edit_dist("intention", "execution") gives 8, matching
   the book.

   suggest() works out each candidate's distance once and keeps the 3
   closest, closest first; the first one replaces the token. When
   several candidates are the same distance away, the one that comes
   first in the dictionary (alphabetical order) wins: "speling" is 1
   away from both "spelding" and "spelling", and gets "spelding".

6. Write the output (run)
   Each token, or its replacement, is written to corrected_<input name>
   followed by a space. Every changed word is printed. A token with no
   candidates at all would be printed as "No suggestions" and kept as it
   is, but with the fixes above that only happens if the dictionary is
   empty.

Files in this folder
--------------------
  spellchecker.py   the script
  BSTNode.py        a binary search tree for storing the dictionary,
                    from an earlier version. Nothing uses it (Known
                    issue 13).
  files/            the dictionary and sample texts:
    words                    the dictionary, 235,886 words, one per
                             line. It looks like a copy of the Unix
                             word list /usr/share/dict/words (from
                             Webster's Second International
                             Dictionary, 1934), with 3 entries spelled
                             differently and 93 words missing compared
                             with the copy on macOS (Known issue 7).
    short_words              the first 100 words of words, for quick
                             tests
    short1.txt               5 lines from Frankenstein with 3
                             misspellings: wioll, yesterdy, welfure
    text_1.txt               the opening of Franklin D. Roosevelt's
                             first inaugural address (1933), 2,080
                             tokens, no misspellings
    apostraphe_test.txt      one line with an apostrophe and single
                             quotes
    test_tokenize.txt        one line with commas and full stops, for
                             trying the tokenizer
    corrected_short1.txt     saved output of the script for these
    corrected_text_1.txt     inputs. The first two match what the
    corrected_apostraphe_    script prints today; the apostrophe one
      test.txt               is from an older version (Known issue 15).

Setup
-----
Nothing to install: the script only uses the Python standard library
(Python 3.9 or newer, for the list[str] type hints).

How to run
----------
      python3 spellchecker.py TEXTFILE [-o OUTFILE] [-d WORDLIST]

  TEXTFILE       the text file to check.
  -o OUTFILE     where to save the corrected text. If left out, it's the
                 input file's name with corrected_ in front, in the same
                 folder as the input (files/short1.txt ->
                 files/corrected_short1.txt). An existing file is
                 replaced, and the script says so.
  -d WORDLIST    the dictionary, one word per line. If left out,
                 files/words in the script's folder is used, so the
                 script can be run from any folder.
  -h             show the help message (also shown when the script is
                 run with no arguments).

For short1.txt it takes about half a second:

      $ python3 spellchecker.py files/short1.txt
      Changing wioll to will
      Changing yesterdy to yesterday
      Changing welfure to welfare
      Saved the corrected text to files/corrected_short1.txt.

and files/corrected_short1.txt holds the whole text on one line:

      you will rejoice to hear that no disaster have accompany the
      commencement of an enterprise which you have regard with such
      evil foreboding . I arrive here yesterday , and my first task is
      ...

(wrapped here; "have accompany" and "regard" are grammar mistakes in
the sample, not spelling mistakes, so a spell checker shouldn't change
them).

text_1.txt takes about a second. It flags 252 of its 2,080 tokens and
changes all of them, all wrongly. The most common changes:

      Changing It to Ita                   (11 times)
      Changing helped to helper            (7 times)
      Changing has to hash                 (7 times)
      Changing We to Wea                   (7 times)
      Changing This to Think               (6 times)
      Changing In to Ind                   (6 times)
      Changing They to The                 (5 times)
      Changing Nation to Natica            (5 times)

97 of the 252 are changed to a shorter form of themselves, such as
efforts -> effort and years -> year.

A missing input file or dictionary, an output folder that doesn't
exist, or an output file that is the same as the input or the
dictionary stops the script with a short error before any work is done:

      $ python3 spellchecker.py nope.txt
      usage: spellchecker.py [-h] [-o OUTFILE] [-d WORDLIST] TEXTFILE
      spellchecker.py: error: input file 'nope.txt' not found

A file that isn't text (e.g. a .wav file) gives:

      Error: 'sa1.wav' is not a UTF-8 text file

Known issues
------------
Found by running the script and each of its functions on the files in
files/.

 1. [FIXED] The dictionary was opened as '/files/words'. The leading
    slash made it an absolute path, so the file was never found and the
    script crashed before doing anything. test_trie() had the same path.
    The dictionary is now files/words found relative to the script, or
    any word list given with -d, and test_trie() uses the same default.

 2. [FIXED] The output file was named by adding "corrected_" to the front
    of the whole input path, so any input outside the current folder
    failed: files/short1.txt -> corrected_files/short1.txt, a folder
    that doesn't exist. An existing output file was overwritten without
    a word, and a wrong number of arguments printed "Incorrect number of
    arguments" and exited with status 0 (success). The script now uses
    argparse (see "How to run"): corrected_ goes in front of the file
    name only, -o picks another output file, replacing a file is
    reported, the files are checked before any work is done, and
    mistakes exit with a usage message and status 2.

 3. [FIXED] suggest() didn't pick the closest candidate. It kept the
    first 3 candidates it saw, whatever their distance, and after that a
    new candidate replaced the third one only if it was exactly as close,
    never if it was closer. "wioll" became "wice" (distance 5) though
    "will" (distance 1) was a candidate, and suggest(['zzzzzz', 'yyyyyy',
    'xxxxxx', 'cat'], 'cat') returned ['zzzzzz', 'yyyyyy', 'xxxxxx']. It
    now keeps the 3 closest, closest first, and works out each distance
    only once: "wioll" -> "will", and the example gives ['cat',
    'zzzzzz', 'yyyyyy'].

 4. [FIXED] get_all_children() stopped at the first complete word on
    each branch and never looked below it, so none of the 41 longer
    words starting with "will" were candidates. Only 37 of the 734 "wi"
    words were found for "wioll", and from the root it found 52 words
    out of 235,886. It now always keeps going down, and finds all 734
    "wi" words and every word in the dictionary from the root.

 5. [FIXED] get_similar_words() missed candidates:
      - If every letter of the token followed a path in the trie but the
        token wasn't a word, it returned no candidates, so "Nor", "So",
        "It", "We" and "teh" got "No suggestions". It now returns the
        words that continue the token (Nor -> Nora).
      - The part that matched was never a candidate itself: "abandonx"
        couldn't become "abandon". Now it is (abandonx -> abandon).
    It also said that a mistake early in a word can never be fixed. That
    is how this way of finding candidates works rather than a bug in the
    code, so it is now issue 16.

 6. The check is case-sensitive, and the dictionary has most words only
    in lower case, so any word at the start of a sentence is flagged:
    "This", "It", "We", "In". That is 88 of the 252 tokens flagged in
    text_1.txt. The replacement is then found among words starting with
    the capital letter, which are mostly names or rare words: It -> Ita,
    This -> Think, United -> Unitarian, President -> Presley.

 7. The dictionary has no inflected forms: no plurals, -ed, -ing or -s
    verb forms. "has", "friends", "helped", "needed" and "years" aren't
    in it. About 150 of the 252 tokens flagged in text_1.txt are
    inflections of dictionary words, and they get "corrected" to the
    closest word with the same start: efforts -> effort, helped ->
    helper, has -> hash, friends -> friendship. The dictionary also
    lacks most names (Hoover -> Hooverism, Congress -> Congresso).
    Compared with macOS's /usr/share/dict/words, files/words is missing
    93 words (e.g. acknowledgment, administered) and has 3 misspelled
    entries (administerd, stucturelessness, transvaal), so "administerd"
    is accepted as correct.

 8. The tokenizer:
      - splits off every full stop, so "Mr." becomes Mr . and "3.5"
        becomes 3 . 5, and "U.S.A." becomes six tokens.
      - doesn't split off colons or the em dash: "order:", "credits:"
        and "itself—nameless" are checked as single words (order: ->
        order, itself—nameless -> itself, losing "nameless").
      - doesn't separate quotes from words: in apostraphe_test.txt,
        'going and beach' are checked with the quote attached
        ('going -> ', beach' -> beach), and he's -> he. The code meant
        to handle apostrophes is commented out.
      - lets numbers through to the spell check: "48" -> "A".

 9. The output loses the text's layout. Every token is written followed
    by a space, so the whole file comes out on one line, with spaces
    before punctuation ("foreboding ." and "yesterday ,") and no
    paragraph breaks. The comment "put the file back together using
    regex" was never done. The comment at the end of run() also
    says the script should print the line each misspelled word appeared
    in, but it only prints the word.

10. Punctuation is handled by adding 25 marks to the dictionary as words
    (add_punction_to_trie). It would be simpler to skip punctuation
    tokens. The list includes a backslash and the em dash but not the
    colon, and the function name is misspelled ("punction").

11. edit_dist() crashes if either word is empty: its result is read
    from the loop variables (dist[row][col]), which are never set when
    a loop doesn't run, giving "UnboundLocalError: cannot access local
    variable 'row'". It should return dist[-1][-1]. (suggest() also used
    to work out the same edit distances over and over; that was fixed
    with issue 3.)

12. The trie's per-node counter is updated on every insert but only
    returned by find_prefix(), and nothing uses it. "from typing import
    List, Tuple" imports List, which isn't used.

13. BSTNode.py isn't used by spellchecker.py, and it doesn't work for
    the real dictionary. words is in alphabetical order, so inserting
    the words one by one into an unbalanced binary search tree makes
    one long chain, and create_dict('files/words') crashes with
    "RecursionError: maximum recursion depth exceeded" (short_words
    works). insert() also treats an empty string as an empty tree
    ("if not self.val"), the middle child is never used, and the file
    ends with a note about Dijkstra's algorithm that was never written.

14. There are no tests. test_trie(), test_find_similar_words() and
    test_tokenize() print things for checking by eye, are never called,
    and test_find_similar_words() and test_tokenize() crash because
    their files are opened as 'words' and 'test_tokenize.txt', relative
    to the current folder.

15. The saved outputs in files/ are from older versions of the script.
    corrected_apostraphe_test.txt has one token per line and no changes,
    while the script now gives "jake Abba said he ' to the beach".
    corrected_short1.txt still has "wice" and corrected_text_1.txt has
    the 157 changes from before issues 3-5 were fixed. ("apostraphe" in
    the two file names is also a misspelling of "apostrophe".)

16. A mistake at or before the letter where the token leaves the trie
    can't be fixed, because candidates only come from words that share
    every letter before that point. "hte" leaves the trie after "h", so
    "the" is never considered and it becomes "hate". After issues 3-5,
    common typos give:
        hte -> hate          teh -> tehsil        recieve -> recidive
        acommodate -> acoma  wierd -> wierangle   beleive -> beleave
        speling -> spelding (as close as "spelling", but first in the
        dictionary)
    Fixing this means looking further back up the trie, e.g. trying
    the words that start with one or two fewer letters as well, or
    comparing against the whole dictionary. The second is slow in pure
    Python: "48" already gets the whole dictionary as candidates and
    takes about half a second for that one word.
