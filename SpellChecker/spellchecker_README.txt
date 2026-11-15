Spell Checker
=============

What it does
------------
spellchecker.py reads a text file, finds the words that aren't in its
dictionary, replaces each one with the closest dictionary word, and
saves the corrected text to a new file whose name starts with
"corrected_". Everything else in the text (line breaks, spacing,
punctuation) is kept exactly as it was. Each change is printed with its
line number and the part of the line it is in.

It checks against two word lists: files/words, a 1934 dictionary word
list, and files/brown_words, words from the Brown corpus that the first
list doesn't cover (names, modern words, and some common words it is
missing). Words with capital letters, common endings and contractions
("This", "helped", "cities", "people's", "don't") are recognised, even
though the dictionary only lists base forms. It fixes all 3
misspellings in files/short1.txt. files/text_1.txt has no misspellings,
and 1 of its 2,080 words is still changed, because neither list has it
(see "Limitations").

How it works
------------
The steps below describe what the code does, and where it still does
something different from what was meant (see "Known issues").

1. Load the word lists into a trie (load_dictionaries, trie_dict, insert)
   files/words has 235,886 words, one per line, and files/brown_words
   has 2,645 more (see "Files in this folder"). Each is added to a trie:
   a tree with one node per letter, where the path from the root spells
   a word and the last letter's node is marked word_finished. Words that
   start the same way share nodes, so "abandon" and "abandoned" share
   their first 7 letters. Each node's children are kept in a list and
   searched one by one. Building the trie takes about 0.4 seconds. -d
   loads other word lists instead.

   Each node also counts how many dictionary words pass through it,
   which find_prefix() returns: find_prefix(root, 'aba') gives (True, 68)
   because "aba" is a word and 68 words start with it.

2. Find the words in each line (WORD_PATTERN, tokenize)
   The text is read line by line and the words are found where they are
   with a regular expression, so nothing between them is changed. A word
   is a run of letters, with apostrophes allowed inside it (he's, don't,
   o'clock). Everything else separates words:
     - spaces and punctuation, including colons: "order:" is "order".
     - hyphens and dashes: "self-seekers" is checked as "self" and
       "seekers", and "itself—nameless" as "itself" and "nameless".
     - full stops: "Mr." is "Mr" and "U.S.A." is "U", "S" and "A", which
       are all in the dictionary. A short list of abbreviations that
       aren't (ABBREVIATIONS: Dr, St, etc, vs, Jan, ...) is accepted when
       a full stop follows them.
     - an apostrophe at the start or end of a word, which is a quote
       mark: in 'going to the beach' the words are "going" and "beach".
       This also takes the apostrophe off a plural possessive
       (citizens' -> citizens). The curly apostrophe (’) is treated like
       the straight one.
   Anything with a digit in it ("48", "1930s", "3rd") is skipped.

3. Check each word (is_spelled_correctly, word_exists, base_forms)
   word_exists() checks whether a word is in the trie exactly as spelled:
   following its letters from the root must end on a node marked
   word_finished. is_spelled_correctly() accepts a word if word_exists()
   is true for:
     - the word itself, or its lower-case form if it has capitals, so
       "This" at the start of a sentence and "THIS" in a heading are
       fine. A lower-case word isn't accepted because of a capitalised
       entry: "america" is flagged even though "America" is a word.
     - or any of its base forms (base_forms()): what the word would be
       with one common ending taken off. files/words lists only the base
       form of most words ("help", not "helps", "helped", "helping"),
       so the endings are handled by rules:
           's           possessive        people's
           n't 're 've  contractions      don't, they're, I've, we'll,
           'll 'd 'm                      I'd, I'm
           -s, -es      plurals, verbs    friends, taxes, goes
           -ies, -ied   y -> i            cities, multiplied
           -ves         f, fe -> v        selves, knives
           -men         man -> men        women, salesmen
           -ed, -ing    verbs             helped, believed, planned,
                                          making, putting
           -er, -est    comparatives      larger, greatest, bigger
           -ier, -iest  y -> i            happier, happiest
           -ly, -ily    adverbs           finally, simply, primarily
       The possessive or contraction comes off first, so "children's"
       -> "children" -> "child" works too.
       A few spelling checks stop some rules from accepting misspellings:
       -es only follows s, x, z, ch, sh or o ("taxs" and "wishs" are
       rejected); an e is dropped before -ing, -ed, -er and -est
       ("hopeing" is rejected, but "seeing" and "dyeing" are fine);
       3-letter words like run, sit and big must double their last letter
       ("biger" is rejected); and -ier and -iest need at least 3 letters
       in front ("thier" isn't "thy" + -ier).
       Common irregular forms the dictionary doesn't have are listed in
       IRREGULAR_FORMS: has, began, held, heard, paid, children, feet,
       can't, won't, o'clock, ... (most irregular forms, like "went",
       "said" and "men", are in files/words already).

   With files/words alone, 0.39% of the lower-case words in the Brown
   corpus (about a million words of 1960s American English, from NLTK)
   are flagged, mostly words the list doesn't have, such as near, box,
   kid and proud. files/brown_words adds those.

4. Find candidate replacements (correct, get_similar_words,
   get_all_children)
   For a misspelled word, the trie is followed for as long as its
   letters match. The candidates are all the dictionary words that start
   with the part that matched, including that part itself if it is a
   word. For "wioll", "wi" matches and "o" doesn't, so the candidates are
   the 734 words starting with "wi". For "abandonx" they are "abandon"
   and the 6 words that start with it. If every letter matches but the
   word isn't in the dictionary ("Nor" is the start of "Norway"), the
   candidates are the words that continue it.

   get_all_children() collects the words below a node by walking every
   branch to the end. Each branch can hold several words ("will",
   "willing", "willingly"), so it keeps going past a complete word.

   This means a mistake at or before the letter where the word leaves
   the trie can't be fixed, because the right word is never a candidate
   (see "Limitations"). A word whose first letter isn't in the trie at all
   gets the whole dictionary as candidates, which takes about half a
   second.

   correct() handles capitals: a word with capitals gets the candidates
   for its lower-case form too, and each lower-case candidate is given
   the word's capitals (match_case()), so "Wioll" gets "Will" and
   "WIOLL" gets "WILL". Candidates that already have capitals, like
   names, are left as they are: "Amercia" gets "America".

5. Pick the closest candidate (suggest, edit_dist)
   Each candidate is compared with the word by minimum edit distance
   (Jurafsky & Martin, figure 3.25): the cheapest way to turn one word
   into the other, where inserting or deleting a letter costs 1 and
   substituting one letter for another costs 2. "wioll" to "will" is 1
   (delete the o). edit_dist("intention", "execution") gives 8, matching
   the book, and the distance from "" to "abc" is 3.

   suggest() works out each candidate's distance once and keeps the 3
   closest, closest first; the first one replaces the word. When
   several candidates are the same distance away, the one that comes
   first in the dictionary (alphabetical order) wins: "speling" is 1
   away from both "spelding" and "spelling", and gets "spelding".

6. Write the output (run)
   Each misspelled word is replaced where it is in its line, and the
   lines are written to the output file, so the corrected text has the
   same layout as the original. Each change is printed with its line
   number and up to 25 characters on each side of the word. A word with
   no candidates at all would be printed as "no suggestions" and kept as
   it is, but that only happens if the dictionary is empty.

Files in this folder
--------------------
  spellchecker.py        the script
  test_spellchecker.py   tests (see "Running the tests")
  make_brown_words.py    makes files/brown_words from the Brown corpus
                         (needs NLTK; only needed to remake the list)
  BSTNode.py             a binary search tree, kept as a simple example
                         of a tree structure. An earlier version of the
                         spell checker stored the dictionary in it;
                         spellchecker.py doesn't use it.
  files/                 the word lists and sample texts:
    words                    235,886 words, one per line, base forms
                             only. It is the same list as NLTK's
                             "words" corpus: the Unix word list from
                             Webster's Second International Dictionary
                             (1934).
    brown_words              2,645 words from the Brown corpus (about a
                             million words of American English from
                             1961, which comes with NLTK) that words
                             and the ending rules don't already cover:
                             mostly names and places, plus words like
                             near, box, kid, proud, okay and statewide.
                             make_brown_words.py keeps words that
                             appear at least twice, which leaves out
                             most one-off typos in the corpus ("runing"
                             and "publically" appear once each), and
                             leaves out possessives like "Atlanta's"
                             when "Atlanta" is kept.
    short_words              the first 100 words of words, for quick
                             tests
    short1.txt               5 lines from Frankenstein with 3
                             misspellings: wioll, yesterdy, welfure
    text_1.txt               the opening of Franklin D. Roosevelt's
                             first inaugural address (1933), 2,080
                             words, no misspellings
    apostrophe_test.txt      one line with an apostrophe and single
                             quotes
    test_tokenize.txt        one line with commas and full stops, for
                             trying the tokenizer
    corrected_short1.txt     the script's output for short1.txt,
    corrected_text_1.txt     text_1.txt and apostrophe_test.txt, as
    corrected_apostrophe_    examples (the tests check that they match
      test.txt               what the script gives)

Setup
-----
Nothing to install: the script only uses the Python standard library
(Python 3.9 or newer, for the list[str] type hints). NLTK is only needed
to remake files/brown_words with make_brown_words.py (see the top-level
README.md for installing NLTK and its 'brown' data):

      python3 make_brown_words.py
      Saved 2645 words to files/brown_words.

How to run
----------
      python3 spellchecker.py TEXTFILE [-o OUTFILE] [-d WORDLIST]

  TEXTFILE       the text file to check.
  -o OUTFILE     where to save the corrected text. If left out, it's the
                 input file's name with corrected_ in front, in the same
                 folder as the input (files/short1.txt ->
                 files/corrected_short1.txt). An existing file is
                 replaced, and the script says so.
  -d WORDLIST    a word list to check against, one word per line. Give
                 -d more than once to use several lists. If left out,
                 files/words and files/brown_words in the script's
                 folder are used, so the script can be run from any
                 folder.
  -h             show the help message (also shown when the script is
                 run with no arguments).

For short1.txt it takes about half a second:

      $ python3 spellchecker.py files/short1.txt
      Line 1: changing wioll to will
          you wioll rejoice to hear that no...
      Line 3: changing yesterdy to yesterday
          ...oreboding. I arrive here yesterdy, and my first task is to...
      Line 4: changing welfure to welfare
          my dear sister of my welfure and increasing confidenc...
      Changed 3 words.
      Replaced files/corrected_short1.txt with the corrected text.

(the last line says "Saved the corrected text to ..." when the output
file doesn't exist yet). files/corrected_short1.txt then has the same
lines as short1.txt with the three words fixed:

      you will rejoice to hear that no disaster have accompany the
      commencement of an enterprise which you have regard with such evil
      foreboding. I arrive here yesterday, and my first task is to assure
      my dear sister of my welfare and increasing confidence in the success
      of my undertaking.

("have accompany" and "regard" are grammar mistakes in the sample, not
spelling mistakes, so a spell checker shouldn't change them.)

text_1.txt has no misspellings. 1 of its 2,080 words is changed,
because neither word list has it (see "Limitations"):

      $ python3 spellchecker.py files/text_1.txt
      Line 3: changing preeminently to preen
          ...r people impels. This is preeminently the time to speak the tr...
      Changed 1 word.
      Replaced files/corrected_text_1.txt with the corrected text.

Running the tests
-----------------
From inside this folder:

      python3 -m unittest test_spellchecker

  (or run the file with python3). test_spellchecker.py has 48 tests and
  takes about 5 seconds. NLTK isn't needed.

  Most use a tiny made-up dictionary of 32 words, so the right answers
  can be worked out by hand: the trie and its prefix counts, edit
  distances (including the book's intention -> execution = 8 and empty
  words), that suggest() keeps the closest words, which candidates
  get_similar_words() finds, the ending rules and the spelling checks
  that reject "taxs", "hopeing" and "biger", capitals, finding words in a
  line (full stops, dashes, apostrophes, numbers), and which Brown words
  make_brown_words.py keeps. The rest load the real word lists and run
  the script from the command line: the sample typos, that short1.txt
  and text_1.txt come out with only the misspelled words changed, that
  the saved corrected_* files match what the script gives, abbreviations,
  -d, and every error above.

  To check that the tests catch real mistakes, bugs that were fixed in
  this script were put back one at a time: suggest() replacing only on
  ties, get_all_children() stopping at the first word, the matched part
  not being a candidate, a case-sensitive check, no ending rules, no
  doubling check, no contractions, splitting words on spaces only,
  writing the output on one line, edit_dist() reading the wrong cell,
  and leaving out files/brown_words. Each one made at least one test
  fail.

A missing input file or dictionary, an output folder that doesn't
exist, or an output file that is the same as the input or the
dictionary stops the script with a short error before any work is done:

      $ python3 spellchecker.py nope.txt
      usage: spellchecker.py [-h] [-o OUTFILE] [-d WORDLIST] TEXTFILE
      spellchecker.py: error: input file 'nope.txt' not found

A file that isn't text (e.g. a .wav file) gives:

      Error: 'sa1.wav' is not a UTF-8 text file

Limitations
-----------
These come from how the spell checker works rather than from bugs.

Mistakes early in a word can't be fixed. Candidates only come from
words that share every letter before the point where the word leaves
the trie (see "How it works", step 4). "hte" leaves the trie after "h",
so "the" is never considered and it becomes "hate". Common typos give:

      hte -> hate          teh -> tehsil        recieve -> recidive
      acommodate -> acoma  wierd -> wierangle   beleive -> beleave
      Thier -> Thierry     speling -> spelding (as close as "spelling",
                                                but first alphabetically)

Fixing this would mean looking further back up the trie, e.g. also
trying the words that start with one or two fewer letters, or comparing
against the whole word list, which is slow in pure Python (a word whose
first letter isn't in the trie already gets every word as a candidate
and takes about half a second).

The ending rules accept some misspellings, because they only check that
the stem is a word, not that the word takes that ending:
  - regular endings on irregular words: runned, goed, childs, mouses,
    tooths.
  - endings on the wrong kind of word, or the wrong stem: happyly
    (happy + -ly), writting (writ + -ing), runing (rune + -ing),
    begining (begin + -ing without the doubled n; only 3-letter words
    are checked for doubling).
  - teachs: plain -s is allowed after "ch" because of words like
    "stomachs".

The word lists still have gaps and mistakes. "preeminently" is in
neither list (it appears only once in the Brown corpus), so it becomes
"preen". files/words has 3 misspelled entries compared with macOS's
/usr/share/dict/words (administerd, stucturelessness, transvaal), and
some rare old entries that are usually mistakes ("dont"), so these
aren't flagged.

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
      - If every letter of the word followed a path in the trie but the
        word wasn't in the dictionary, it returned no candidates, so
        "Nor", "So", "It", "We" and "teh" got "No suggestions". It now
        returns the words that continue it (Nor -> Nora).
      - The part that matched was never a candidate itself: "abandonx"
        couldn't become "abandon". Now it is (abandonx -> abandon).
    It also said that a mistake early in a word can never be fixed. That
    is how this way of finding candidates works rather than a bug in the
    code, so it is described under "Limitations".

 6. [FIXED] The check was case-sensitive, and the dictionary has most
    words only in lower case, so any word at the start of a sentence was
    flagged: "This", "It", "We", "In", 88 of the 252 tokens flagged in
    text_1.txt. The replacement was then found among words starting with
    the capital letter, mostly names: This -> Thiospira, It -> Ita.
    A word with capitals is now also checked in lower case, and its
    candidates include the lower-case ones, given the word's capitals
    (see "How it works", steps 3 and 4): "Wioll" -> "Will".

 7. [FIXED] The dictionary has no inflected forms: no plurals, -ed, -ing
    or -s verb forms. "has", "friends", "helped", "needed" and "years"
    aren't in it, and about 150 of the 252 tokens flagged in text_1.txt
    were inflections of dictionary words, "corrected" to other words with
    the same start (helped -> helper, has -> hash). files/words is kept
    as it is, and words are now also accepted if taking a common ending
    off them leaves a dictionary word, with a list of irregular forms
    (see "How it works", step 3). Together with issue 6 this took
    text_1.txt from 252 flagged tokens to 8. The rules have limits of
    their own (see "Limitations"), and some words were missing from
    files/words altogether (issue 15).

 8. [FIXED] The tokenizer split off every full stop ("Mr." -> Mr .,
    "3.5" -> 3 . 5, "U.S.A." -> six tokens), didn't split off colons or
    the em dash ("order:" -> order, "itself—nameless" -> itself, losing
    "nameless"), left quote marks attached to words ('going -> '), and
    let numbers through to the spell check ("48" -> "A"). The code meant
    to handle apostrophes was commented out. Words are now found with
    one regular expression (see "How it works", step 2): apostrophes
    count only inside a word, everything else separates words, words
    with digits are skipped, and possessives and contractions (people's,
    don't, they're) are recognised by the ending rules. This took
    text_1.txt from 8 changed words to 2.

 9. [FIXED] The output lost the text's layout. Every token was written
    followed by a space, so the whole file came out on one line, with
    spaces before punctuation ("foreboding ." and "yesterday ,") and no
    paragraph breaks, and the script printed only the misspelled word,
    not the line it was in as a comment said it should. Misspelled words
    are now replaced where they are in each line, so the output keeps
    the input's layout (for text_1.txt and short1.txt, only the changed
    words differ), and each change is printed with its line number and
    the part of the line around it.

10. [FIXED] Punctuation was handled by adding 25 marks to the dictionary
    as words (add_punction_to_trie), a list that had a backslash and the
    em dash but no colon. Punctuation never reaches the spell check now
    (issue 8), so the function was removed.

11. [FIXED] edit_dist() crashed if either word was empty: its result was
    read from the loop variables (dist[row][col]), which are never set
    when a loop doesn't run, giving "UnboundLocalError: cannot access
    local variable 'row'". It now returns the bottom-right cell,
    dist[-1][-1]: the distance from "" to "abc" is 3, and from "" to ""
    is 0.

12. [FIXED] "from typing import List, Tuple" imported List, which wasn't
    used; it now imports only Tuple. The trie's per-node counter, which
    nothing used, was kept: it is correct (it counts the dictionary words
    that start with a prefix, see "How it works", step 1) and is part of
    showing how a trie works.

13. [FIXED] There were no tests. test_trie(), test_find_similar_words()
    and test_tokenize() printed things for checking by eye, were never
    called, and two of them crashed because their files were opened
    relative to the current folder. They are replaced by
    test_spellchecker.py, 48 unit tests (see "Running the tests").

14. [FIXED] The saved outputs in files/ were from older versions of the
    script: corrected_apostraphe_test.txt had one token per line,
    corrected_short1.txt had "wice" and everything on one line, and
    corrected_text_1.txt had 157 wrong changes. They have been made again
    with the script as it is now, and a test checks that they stay the
    same as its output. The misspelling "apostraphe" in two file names is
    fixed: they are now apostrophe_test.txt and
    corrected_apostrophe_test.txt.

15. [FIXED] files/words (NLTK's "words" corpus, a 1934 word list) was
    missing some common words: near, box, kid, proud, hang and fond were
    the most frequent lower-case words in the Brown corpus that it
    rejected, and in text_1.txt "preeminently" and "falsity" were changed
    to "preen" and "falsify". files/words is kept as it is, and
    files/brown_words adds the 2,645 Brown words it doesn't cover (see
    "Files in this folder"), which also brings in names like Chicago and
    Abbott. text_1.txt now has 1 change. What is still missing is under
    "Limitations".
