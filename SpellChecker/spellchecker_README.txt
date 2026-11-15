Spell Checker
=============

What it does
------------
spellchecker.py reads a text file, finds the words that aren't in its
dictionary, replaces each one with the closest dictionary word, and
saves the corrected text in the corrected/ folder, with "corrected_"
in front of its name. Everything else in the text (line breaks, spacing,
punctuation) is kept exactly as it was. Each change is printed with its
line number and the part of the line it is in.

It checks against two word lists: dictionary/words, a 1934 dictionary
word list, and dictionary/brown_words, words from the Brown corpus that
the first list doesn't cover (names, modern words, and some common words
it is missing). Words with capital letters, common endings and
contractions ("This", "helped", "cities", "people's", "don't") are
recognised, even though the dictionary only lists base forms. It fixes
all 3 misspellings in samples/short1.txt. samples/text_1.txt has no
misspellings, and 1 of its 2,080 words is still changed, because neither
list has it (see "Limitations").

How it works
------------
The steps below describe what the code does; "Limitations" covers what
it can't do.

1. Load the word lists into a trie (load_dictionaries, trie_dict, insert)
   dictionary/words has 235,886 words, one per line, and dictionary/brown_words
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
       with one common ending taken off. dictionary/words lists only the base
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
       "said" and "men", are in dictionary/words already).

   With dictionary/words alone, 0.39% of the lower-case words in the Brown
   corpus (about a million words of 1960s American English, from NLTK)
   are flagged, mostly words the list doesn't have, such as near, box,
   kid and proud. dictionary/brown_words adds those.

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
  make_brown_words.py    makes dictionary/brown_words from the Brown corpus
                         (needs NLTK; only needed to remake the list)
  BSTNode.py             a binary search tree, kept as a simple example
                         of a tree structure (spellchecker.py doesn't
                         use it)
  dictionary/            the word lists:
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
  samples/               sample texts to check:
    short1.txt               5 lines from Frankenstein with 3
                             misspellings: wioll, yesterdy, welfure
    text_1.txt               the opening of Franklin D. Roosevelt's
                             first inaugural address (1933), 2,080
                             words, no misspellings
    apostrophe_test.txt      one line with an apostrophe and single
                             quotes
    test_tokenize.txt        one line with commas and full stops, for
                             trying the tokenizer
  corrected/             where corrected texts are saved by default.
                         It holds the script's output for three of the
                         samples, as examples (the tests check that
                         they match what the script gives):
    corrected_short1.txt
    corrected_text_1.txt
    corrected_apostrophe_test.txt

Setup
-----
Nothing to install: the script only uses the Python standard library
(Python 3.9 or newer, for the list[str] type hints). NLTK is only needed
to remake dictionary/brown_words with make_brown_words.py (see the top-level
README.md for installing NLTK and its 'brown' data):

      python3 make_brown_words.py
      Saved 2645 words to dictionary/brown_words.

How to run
----------
      python3 spellchecker.py TEXTFILE [-o OUTFILE] [-d WORDLIST]

  TEXTFILE       the text file to check.
  -o OUTFILE     where to save the corrected text. If left out, it's the
                 input file's name with corrected_ in front, in the
                 corrected/ folder next to the script, wherever the input
                 is (samples/short1.txt -> corrected/corrected_short1.txt).
                 An existing file is replaced, and the script says so.
  -d WORDLIST    a word list to check against, one word per line. Give
                 -d more than once to use several lists. If left out,
                 dictionary/words and dictionary/brown_words next to the
                 script are used, so the script can be run from any
                 folder.
  -h             show the help message (also shown when the script is
                 run with no arguments).

For short1.txt it takes about half a second:

      $ python3 spellchecker.py samples/short1.txt
      Line 1: changing wioll to will
          you wioll rejoice to hear that no...
      Line 3: changing yesterdy to yesterday
          ...oreboding. I arrive here yesterdy, and my first task is to...
      Line 4: changing welfure to welfare
          my dear sister of my welfure and increasing confidenc...
      Changed 3 words.
      Replaced corrected/corrected_short1.txt with the corrected text.

(the last line says "Saved the corrected text to ..." when the output
file doesn't exist yet). corrected/corrected_short1.txt then has the same
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

      $ python3 spellchecker.py samples/text_1.txt
      Line 3: changing preeminently to preen
          ...r people impels. This is preeminently the time to speak the tr...
      Changed 1 word.
      Replaced corrected/corrected_text_1.txt with the corrected text.

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
  and leaving out dictionary/brown_words. Each one made at least one test
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
"preen". dictionary/words has 3 misspelled entries compared with macOS's
/usr/share/dict/words (administerd, stucturelessness, transvaal), and
some rare old entries that are usually mistakes ("dont"), so these
aren't flagged.
