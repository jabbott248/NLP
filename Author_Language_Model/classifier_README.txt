Author Language Model Classifier
================================

What it does
------------
classifier.py guesses who wrote a piece of text. It builds one language.  
model per author from sample text, then scores each line of new text with
every model and picks the author whose model fits best.

The sample data is four authors (one excerpt each, ~10,000-20,000 lines):

  austen.txt    Jane Austen, Emma
  dickens.txt   Charles Dickens, A Tale of Two Cities
  wilde.txt     Oscar Wilde, The Picture of Dorian Gray
  tolstoy.txt   Leo Tolstoy, War and Peace

How it works
------------
1. Language model (LanguageModel.create_lm)
   Each author's file is split into tokens with nltk.word_tokenize(), and
   the model counts:
     - unigrams: how often each token occurs
     - bigrams:  how often each pair of adjacent tokens occurs
     - N_c:      how many different bigrams occur exactly c times
                 (the "frequency of frequencies" Good-Turing needs)
   It's a bigram model: the probability of a sentence is the product of
   P(word | previous word) for every word.

2. Smoothing (gt_get_prob_of_w)
   A bigram that never appears in an author's text would get probability
   0 and rule that author out completely. Good-Turing smoothing fixes this
   by moving some probability from seen bigrams to unseen ones. It replaces
   each count c with an adjusted count
       c* = (c + 1) * N_(c+1) / N_c
   and gives unseen bigrams a share of the probability N_1 / N, where N_1 is
   the number of bigrams seen exactly once and N is the total number of
   bigrams. (See Known Issues: the current implementation doesn't do this
   correctly.)

3. Scoring and classification (get_prob_of_sentence)
   Each line is scored against every author's model, and the author with
   the best score is the guess. Scores are kept as -log10 probabilities so
   that multiplying many small probabilities doesn't underflow to 0.

Files in this folder
--------------------
  classifier.py          the script
  authorlist1            the author files to use, one file name per line
  austen.txt, dickens.txt, wilde.txt, tolstoy.txt
                         the author text (ASCII)
  test.txt               sample input for -test mode (12 lines of Emma)
  dev_*.txt, training_*.txt
                         train/test split written by default mode (see
                         Known Issues #7 and #8; the names are swapped)
  txt_ascii/             identical copies of the four author files
  txt_utf8/              UTF-8 versions of the author files (not used)
  words                  English word list (not used; a copy of
                         SpellChecker/files/words)
  test.py                scratch file for trying things out (not used)

Setup
-----
Follow the setup in the top-level README.md (install requirements.txt and
download the NLTK data). This script only needs nltk and its 'punkt' /
'punkt_tab' tokenizer data.

How to run
----------
Run from inside this folder. The file names in the author list are opened
relative to the current directory.

  Default mode: split each author's text, train on one part and measure
  accuracy on the rest:

      python3 classifier.py authorlist1

  For each author file it writes dev_<author>.txt (90% of the lines; used
  to build the model) and training_<author>.txt (every 10th line; used for
  testing), overwriting any existing ones. Each test line is classified, and
  it prints how many lines of each author were guessed correctly.

  Expected output:

      austen   797/1300 correct
      dickens   488/1300 correct
      wilde   376/1000 correct
      tolstoy   836/2000 correct

  That's 44.6% overall. Guessing at random would give 25%. Austen's score
  includes 230 blank lines that are always given to the first author (see
  Known Issues #6); without them it's 50.2%.

  -test mode: train on the full author files and classify every line of
  another file:

      python3 classifier.py authorlist1 -test test.txt

  The -test flag must come second, after the author list. Output is one
  line per input line, naming the author file that scored best:

      Line 1: austen.txt
      Line 2: austen.txt
      ...
      Line 12: austen.txt

  test.txt is all Austen, and all 12 lines are labeled austen.txt. (Before
  issues #1-#3 were fixed it got 8 of 12.) It's an easy test, though: see
  Known Issues #9.


Known Issues
============
These were confirmed by running the script with NLTK 3.10.3 and Python 3.14.
Line numbers refer to classifier.py. Accuracy figures use the default-mode
split, with blank lines left out unless stated otherwise.

Accuracy so far, and what fixing the rest would do:

  Original code (blank lines included)                       32.9%
  #1-#3 fixed: current code (blank lines included)           44.6%
  #1-#3 fixed, blank lines skipped (#6)                      50.2%
  Replace Good-Turing with add-one smoothing (#4, #5)        68.6%
  Replace Good-Turing with bigram/unigram interpolation      76.1%
    (0.7 * P(word | previous) + 0.3 * P(word), add-one on the unigram)

Each test line is a fragment of about 13 tokens (see #10), so 100% isn't
realistic, but a correctly smoothed bigram model gets about three lines in
four right.

Wrong results in scoring
------------------------
1. [FIXED] Arguments to gt_get_prob_of_w() were swapped.
   gt_get_prob_of_w() returns P(word | previous word), with the current
   word first, but get_prob_of_sentence() called
       gt_get_prob_of_w(lm, prev_token, token)
   so it looked up the bigram with the two words reversed and divided by
   the count of the current word instead of the previous one. It now calls
       gt_get_prob_of_w(lm, token, prev_token)
   The parameters were renamed from w0 / w1 to word / prev_word, and the
   function has a header comment explaining them.

2. [FIXED] Log costs were multiplied instead of added.
   -log10(p) turns each probability into a cost, and costs must be added:
   -log(a*b) = -log(a) + -log(b). The product of costs didn't mean
   anything. get_prob_of_sentence() now returns the sum of the costs.
   It uses -math.log10(p) rather than abs(math.log10(p)). They're the same
   for real probabilities, but while #4 still lets some values go above 1,
   abs() would wrongly turn those into a penalty.
   (The POS tagger had the same bug, fixed there as issue #5.)

3. [FIXED] The highest score won, but a higher cost means less likely.
   With costs, the most likely author has the lowest total. Both
   run_no_test_flag() and run_with_test_flag() now pick the author with
   the lowest cost. Fixing #2 and #3 without #1 actually dropped accuracy
   to 22.8%; all three together raise it from 35.7% to 50.2% (blank lines
   skipped).

4. The Good-Turing frequency table N_c is wrong (lines 51-54, 59-60).
   - When a bigram's count goes from c-1 to c, N_c is increased but N_(c-1)
     isn't decreased. So N_c ends up as "bigrams that ever reached count c",
     not "bigrams with count exactly c". For Austen, N_2 is stored as
     14,435 but the true value is 6,491.
   - Line 59 then overwrites N_1 with 1. The true N_1 for Austen is 38,351.
   - Together these give a bigram seen once an adjusted count of
     c* = 2 * N_2 / N_1 = 2 * 14,435 / 1 = 28,870 (it should be under 1).
   Result: 56% of seen bigrams get a "probability" above 1, and the
   probabilities of every word seen after "the" add up to 5,872 instead of
   at most 1. N_c can be computed correctly after counting with
   collections.Counter(self.bigrams.values()).
   frequencies[0] (line 60) is computed but never used.

5. Other Good-Turing problems in gt_get_prob_of_w() (lines 93-125).
   - Unseen bigrams get N_1 / V divided by the previous word's count,
     where V is the number of distinct words. That isn't the Good-Turing
     estimate, and with N_1 forced to 1 it's just 1 / V.
   - If the previous word was never seen, it divides by unknown_unigram,
     which is a fraction (unigrams seen once / total words). Dividing by a
     small fraction makes the probability bigger, not smaller.
   - When N_(c+1) doesn't exist (common for large c), the KeyError fallback
     uses c* = c + 1, which mixes Good-Turing with add-one. The usual
     approach is to apply Good-Turing only to small counts (e.g. c < 5) and
     use the raw count above that.
   - "prob = c_star / 1" (line 123) can never run: if the bigram was seen,
     the previous word was seen too.
   Good-Turing is fiddly to get right for bigrams. A simpler method such
   as interpolation (76.1%) or add-one (68.6%) does much better than the
   current code.

6. Blank lines are always given to the first author (lines 230-236 and
   267-273).
   A blank line has no tokens, so every model gives it a cost of 0. It's a
   tie, and the first model in the author list always wins it. 1,088
   of the 5,600 test lines are blank, so Austen (first in authorlist1) gets
   230 free "correct" answers. -test mode also prints an author for blank
   lines. Blank lines should be skipped.

Evaluation and data
-------------------
7. The train/test file names are swapped (lines 204-210).
   Every 10th line goes to training_<author>.txt and is used for testing.
   The other 90% goes to dev_<author>.txt and is used to build the models.
   A 90/10 split is fine, but the names are backwards. The comment
   "if %5, write to training" is also out of date.

8. Default mode overwrites the dev_*.txt and training_*.txt files every
   time it runs. These generated files are committed to the repo. They
   come out identical each time, but they don't need to be stored or
   written to disk at all: the split can be done in memory.

9. test.txt is part of the training data.
   All 12 lines of test.txt appear word for word in austen.txt, which is
   what -test mode trains on (11 of them are also in dev_austen.txt). So
   it doesn't test text the model hasn't seen, which is why it now gets
   all 12 right. A better test file would use Austen text that isn't in
   austen.txt.

10. It classifies physical lines, not sentences.
    The text is hard-wrapped at about 70 characters, so each "line" is a
    fragment of about 13 tokens (words and punctuation) that often starts
    and ends mid-sentence.
    That's very little evidence to identify an author from. Classifying
    sentences or paragraphs would be more accurate.

11. Training and scoring treat line breaks differently.
    During training, bigrams carry on across line breaks (prev_token is not
    reset), but when scoring, every line starts again from ''. The ''
    start token is counted only once per file (line 34), so P(first word
    | '') is based on a count of 1.

Usability
---------
12. Command-line handling (lines 278-291).
    The -test flag has to be the second argument. There's no usage
    message: running with no arguments only prints "Please supply an
    authorlist", and wrong arguments print "Wrong number of arguments" or
    "No -test flag" without saying what's expected.

13. Author names in the output.
    -test mode prints the file name ("austen.txt") instead of the author.
    Default mode strips the names with re.sub('.txt', '', ...), where '.'
    matches any character; name.removesuffix('.txt') is safer.

14. File names in the author list are relative to the current directory,
    not to the author list's folder, so the script only works when run from
    this folder.

15. It doesn't check for or download the NLTK tokenizer data. On a fresh
    machine nltk.word_tokenize() raises LookupError. (The top-level README
    has the download command; the POS tagger downloads anything that's
    missing automatically.)

Cleanup
-------
16. Unused and broken code (lines 69-90, 143-180).
    - test_prob_of_sentence(), test_good_turing() and test_lm() call
      LanguageModel() without the required author argument (TypeError), and
      use paths like '/txt_ascii/austen.txt', which point to the root of
      the file system.
    - get_prob_of_w() (an add-one version) is only used by a test function.
      It also checks "token in tokens" against a list, which gets very slow
      on large files.

17. Small cleanups:
    - "from tokenize import Double" (line 17) is unused. It comes from
      Python's own source-code tokenizer, not NLTK.
    - The "x in d.keys()" counting blocks can use collections.Counter.
    - The tokens list in create_lm() is only used for its length.
    - The docstring says "Good-Touring"; it's Good-Turing.

18. Unused files: words, txt_ascii/ (identical copies of the top-level
    author files), txt_utf8/ and test.py.
