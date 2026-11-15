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

2. Smoothing (LanguageModel.smooth and gt_get_prob_of_w)
   A bigram that never appears in an author's text would get probability
   0 and rule that author out completely. Smoothing moves some probability
   from seen bigrams to unseen ones. The script uses Good-Turing smoothing
   with Katz backoff:

   - Seen bigrams: each count c is replaced with the Good-Turing adjusted
     count
         c* = (c + 1) * N_(c+1) / N_c
     and P(word | prev_word) = c* / Count(prev_word). c* is a bit smaller
     than c (for Austen, a bigram seen once gets c* = 0.34), which frees up
     some probability after each previous word. Counts of 5 or more
     (GT_MAX_COUNT) keep their raw count, because Good-Turing gets
     unreliable for large counts.

   - Unseen bigrams (Katz backoff): the probability freed up after
     prev_word is shared among the words never seen after it, in
     proportion to how common each word is on its own:
         P(word | prev_word) = alpha(prev_word) * P(word)
     alpha is chosen so that P(w | prev_word) adds up to 1 over all words.
     If prev_word itself was never seen, P(word) is used directly.

   - Unseen words: unigram Good-Turing says the words an author never used
     share a total probability of N_1 / N (the fraction of their tokens
     that were seen only once). That share is split evenly between the
     words in the vocabulary the author never used. The vocabulary is
     shared by all the authors (every word in anyone's text, plus one for
     words none of them used), so every model splits it over the same
     words. That's why smoothing runs in a separate step (smooth_models)
     after all the models are built.

3. Scoring and classification (get_prob_of_sentence)
   Each line is scored against every author's model, and the author whose
   model gives it the lowest cost (the highest probability) is the guess.
   The cost is the sum of -log10 P(word | prev_word) over the line's words.
   Working in logs stops the product of many small probabilities from
   underflowing to 0.

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
  testing), overwriting any existing ones. Each non-blank test line is
  classified, and it prints how many lines of each author were guessed
  correctly.

  Expected output:

      austen   845/1070 correct
      dickens   705/1012 correct
      wilde   582/800 correct
      tolstoy   1327/1630 correct

  That's 76.7% overall. Guessing at random would give 25%.

  -test mode: train on the full author files and classify every line of
  another file:

      python3 classifier.py authorlist1 -test test.txt

  The -test flag must come second, after the author list. Output is one
  line per non-blank input line, naming the author file that scored best.
  The line numbers are the line numbers in the input file, so they still
  match it when blank lines are skipped:

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

Accuracy so far:

  Original code (blank lines included)                       32.9%
  #1-#3 fixed (blank lines included)                         44.6%
  #1-#3 fixed, blank lines skipped                           50.2%
  #1-#5 fixed (blank lines included)                         65.9%
  #1-#6 fixed: current code (blank lines skipped)            76.7%

  For comparison, other ways of smoothing the same bigram counts:
    add-one (Laplace)                                        68.6%
    interpolation: 0.7 * P(word | previous) + 0.3 * P(word)  76.1%

Each test line is a fragment of about 13 tokens (see #10), so 100% isn't
realistic. The fixed Good-Turing model gets about three lines in four
right.

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
   for real probabilities, but before #4 was fixed some values went above
   1, and abs() turned those into a penalty.
   (The POS tagger had the same bug, fixed there as issue #5.)

3. [FIXED] The highest score won, but a higher cost means less likely.
   With costs, the most likely author has the lowest total. Both
   run_no_test_flag() and run_with_test_flag() now pick the author with
   the lowest cost. Fixing #2 and #3 without #1 actually dropped accuracy
   to 22.8%; all three together raise it from 35.7% to 50.2% (blank lines
   skipped).

4. [FIXED] The Good-Turing frequency table N_c was wrong.
   - When a bigram's count went from c-1 to c, N_c was increased but
     N_(c-1) wasn't decreased, so N_c ended up as "bigrams that ever
     reached count c". For Austen, N_2 was stored as 14,435; the true
     value is 6,491.
   - N_1 was then overwritten with 1. The true N_1 for Austen is 38,351.
   - Together these gave a bigram seen once an adjusted count of
     c* = 2 * 14,435 / 1 = 28,870 instead of 0.34.
   56% of seen bigrams got a "probability" above 1, and the probabilities
   of the words seen after "the" added up to 5,872 instead of 1.
   N_c is now computed once counting is finished, with
   Counter(self.bigrams.values()), and nothing overwrites it. The unused
   frequencies[0] was removed.

5. [FIXED] Other Good-Turing problems in gt_get_prob_of_w().
   - Unseen bigrams got N_1 / V divided by the previous word's count. That
     isn't the Good-Turing estimate, and with N_1 forced to 1 it was just
     1 / V. They now get the probability Good-Turing frees up from the seen
     bigrams, shared out with Katz backoff (see "How it works").
   - If the previous word was never seen, it divided by unknown_unigram, a
     fraction, which made the probability bigger, not smaller. It now uses
     the unigram probability P(word).
   - When N_(c+1) didn't exist, a KeyError fallback used c* = c + 1, mixing
     Good-Turing with add-one. Good-Turing is now only used for counts
     below GT_MAX_COUNT (5); larger counts, and the rare small count with
     no N_(c+1), keep their raw count.
   - The "prob = c_star / 1" branch that could never run is gone.

   Two more problems turned up while fixing these:
   - The simple Good-Turing estimate for an unseen bigram, c*_0 = N_1 / N_0,
     is the same for every previous word. After a rare previous word that
     adds up to far more than 1: after '' (the start of a line, seen once)
     the probabilities added up to 5.25. Katz backoff fixes this by
     sharing out only what was freed up after that particular word.
   - unknown_unigram (N_1 / N) is the total probability of all the words
     an author never used, but it was given to each one. An unseen word
     scored about 4,000 times higher than a word seen once, so a line full
     of words an author never used counted in that author's favour. It's
     now split evenly over the unseen words in a vocabulary shared by all
     the models (smooth_models()).

   Now every word's next-word probabilities add up to exactly 1, except
   for 19-32 previous words per author (out of 8,000-11,000) whose
   followers were all seen 5 or more times. Nothing gets freed up there, so
   unseen bigrams get Good-Turing's overall unseen share (N_1 / N) instead,
   and the total comes to up to 1.34.

6. [FIXED] Blank lines were always given to the first author.
   A blank line has no tokens, so every model gave it a cost of 0. It was a
   tie, and the first model in the author list always won it. 1,088 of
   the 5,600 test lines are blank, so Austen (first in authorlist1) got
   230 free "correct" answers and every other author's blank lines counted
   as wrong. -test mode also printed an author for blank lines.
   Both modes now skip blank lines. In -test mode the printed line numbers
   still match the input file (the loop uses enumerate), so skipped lines
   just leave a gap.

Evaluation and data
-------------------
7. The train/test file names are swapped (lines 259-265).
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
    start token is counted only once per file (line 44), so P(first word
    | '') is based on a count of 1.

Usability
---------
12. Command-line handling (lines 341-354).
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
16. Unused and broken code (lines 128-149, 188-225).
    - test_prob_of_sentence(), test_good_turing() and test_lm() call
      LanguageModel() without the required author argument (TypeError), and
      use paths like '/txt_ascii/austen.txt', which point to the root of
      the file system. They also don't call smooth_models(), which every
      model now needs before it can score anything.
    - get_prob_of_w() (an add-one version) is only used by a test function.
      It also checks "token in tokens" against a list, which gets very slow
      on large files.

17. Small cleanups:
    - "from tokenize import Double" (line 17) is unused. It comes from
      Python's own source-code tokenizer, not NLTK.
    - The "x in d.keys()" counting blocks in create_lm() can use
      collections.Counter.
    - The tokens list in create_lm() is no longer used.
    (The docstring's "Good-Touring" was corrected as part of #4 and #5.)

18. Unused files: words, txt_ascii/ (identical copies of the top-level
    author files), txt_utf8/ and test.py.
