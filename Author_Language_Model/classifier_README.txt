Author Language Model Classifier
================================

What it does
------------
classifier.py guesses who wrote a piece of text. It builds one language
model per author from sample text, then scores each sentence of new text
with every model and picks the author whose model fits best.

The sample data is four authors, with excerpts from several of their books
(10,000-20,000 lines each):

  austen.txt    Jane Austen: Emma, Sense and Sensibility, Pride and Prejudice
  dickens.txt   Charles Dickens: A Tale of Two Cities, David Copperfield
  wilde.txt     Oscar Wilde: The Picture of Dorian Gray, The Canterville
                Ghost and other stories, The Portrait of Mr. W. H.
  tolstoy.txt   Leo Tolstoy: War and Peace, Anna Karenina

How it works
------------
1. Sentences (read_sentences)
   The text files are hard-wrapped at about 70 characters, so a line is
   usually a fragment of a sentence. Each file is split into paragraphs at
   blank lines, the lines of each paragraph are joined back together, and
   nltk.sent_tokenize() splits the paragraph into sentences. Each sentence
   is split into tokens with nltk.word_tokenize(). Sentences average 17-25
   tokens, depending on the author.

2. Language model (LanguageModel.train / create_lm)
   For each author the model counts:
     - unigrams: how often each token occurs
     - bigrams:  how often each pair of adjacent tokens occurs
     - N_c:      how many different bigrams occur exactly c times
                 (the "frequency of frequencies" Good-Turing needs)
   Every sentence starts with a start token, <s>, so the model also learns
   which words tend to begin a sentence, and bigrams never run from one
   sentence into the next. It's a bigram model: the probability of a
   sentence is the product of P(word | previous word) for every word.

3. Smoothing (LanguageModel.smooth and gt_get_prob_of_w)
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

4. Scoring and classification (get_prob_of_sentence, classify)
   Each sentence is scored against every author's model, and the author
   whose model gives it the lowest cost (the highest probability) is the
   guess. The cost is the sum of -log10 P(word | prev_word) over the
   sentence's words. Working in logs stops the product of many small
   probabilities from underflowing to 0.

Files in this folder
--------------------
  classifier.py          the script
  authorlist1            the author files to use, one file name per line
  austen.txt, dickens.txt, wilde.txt, tolstoy.txt
                         the author text (ASCII), used for training
  test.txt               sample input for -test mode: one passage per
                         author, removed from the author files (see
                         "How to run")
  txt_ascii/             the original author files, before the test.txt
                         passages were taken out (not used)
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

  Each author's text is split into sentences. Every 10th sentence
  (TEST_EVERY) is held out for testing and the model is trained on the
  other 90%. The split is done in memory; no files are written. Each test
  sentence is classified, and it prints how many of each author's test
  sentences were guessed correctly.

  Expected output:

      austen   484/576 correct
      dickens   557/694 correct
      wilde   495/677 correct
      tolstoy   900/1064 correct

  That's 80.9% overall. Guessing at random would give 25%.

  -test mode: train on the full author files and classify every sentence
  of another file:

      python3 classifier.py authorlist1 -test test.txt

  The -test flag must come second, after the author list. Output is one
  line per sentence: its number, the author file that scored best, and
  the start of the sentence.

  test.txt has one passage per author. Each passage was taken out of its
  author file, so none of it is in the training data:

      Sentences  1-7   Austen, Emma
      Sentences  8-13  Dickens, David Copperfield
      Sentences 14-26  Wilde, The Picture of Dorian Gray
      Sentences 27-33  Tolstoy, Anna Karenina

  Expected output (abridged):

      Sentence 1:   austen.txt   "There is one thing, Emma, which a man can alwa...
      Sentence 2:   austen.txt   It is Frank Churchill's duty to pay this attent...
      ...
      Sentence 7:   tolstoy.txt  '-If he would say so to her at once, in the ton...
      Sentence 8:   dickens.txt  I thought it affecting to see such a sturdy fel...
      ...
      Sentence 22:  dickens.txt  Yes.
      ...
      Sentence 33:  tolstoy.txt  For the first time in his life he knew the bitt...

  26 of the 33 sentences are right: Austen 6/7, Dickens 3/6, Wilde 11/13
  and Tolstoy 6/7. The misses include a one-word sentence ("Yes."), a
  fragment where sent_tokenize split a sentence at a quote (sentence 7),
  and half of the Dickens passage, whose short first-person sentences ("I
  was affected by the story altogether.") could belong to anyone.

  To try your own text, put it in a file (it can be hard-wrapped;
  separate paragraphs with blank lines) and pass that instead of
  test.txt.


Known Issues
============
These were confirmed by running the script with NLTK 3.10.3 and Python 3.14.
Line numbers refer to classifier.py. Accuracy figures use the default-mode
split. Up to #6 the script classified physical lines; from #10 on it
classifies sentences.

Accuracy so far:

  Original code, lines (blank lines included)                32.9%
  #1-#3 fixed, lines (blank lines included)                  44.6%
  #1-#3 fixed, lines (blank lines skipped)                   50.2%
  #1-#5 fixed, lines (blank lines included)                  65.9%
  #1-#6 fixed, lines (blank lines skipped)                   76.7%
  #1-#11 fixed: current code, sentences                      80.9%

  For comparison, other ways of smoothing the same bigram counts (lines,
  blank lines skipped):
    add-one (Laplace)                                        68.6%
    interpolation: 0.7 * P(word | previous) + 0.3 * P(word)  76.1%

The line and sentence figures aren't measured on exactly the same test
text: holding out every 10th sentence picks different text from holding
out every 10th line. But a sentence (17-25 tokens) gives the model more to
go on than a line fragment (about 13), which is where most of the gain
comes from.

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
   Both modes now skipped blank lines. (Since #10 the script classifies
   sentences, which are never empty, so blank lines no longer come up.)

Evaluation and data
-------------------
7. [FIXED] The train/test file names were swapped.
   Every 10th line went to training_<author>.txt and was used for testing,
   and the other 90% went to dev_<author>.txt and was used to build the
   models. The split is now done in memory (see #8) with clear names:
   every TEST_EVERY-th (10th) sentence goes into the test set and the rest
   into the training set.

8. [FIXED] Default mode overwrote the dev_*.txt and training_*.txt files
   every time it ran, and those generated files were committed to the
   repo. run_no_test_flag() now splits each author's sentences in memory
   and writes nothing, and the eight split files were deleted from the
   repo.

9. [FIXED] test.txt was part of the training data.
   All 12 lines of the old test.txt (a passage from Emma) appeared word
   for word in austen.txt, which -test mode trains on. test.txt now has one
   passage per author, and each passage was removed from its author file,
   so none of it is in the training data (see "How to run"). The old
   Austen passage is still the first one, but it's no longer in
   austen.txt. On the new file the classifier gets 26 of 33 sentences
   right.

10. [FIXED] It classified physical lines, not sentences.
    The text is hard-wrapped at about 70 characters, so each "line" was a
    fragment of about 13 tokens that often started and ended mid-sentence.
    read_sentences() now splits each file into paragraphs at blank lines,
    joins each paragraph's lines, and splits it into sentences with
    nltk.sent_tokenize(). Both modes classify sentences: default mode holds
    out every 10th sentence (80.9% correct), and -test mode prints one
    guess per sentence of the input file.
    sent_tokenize isn't perfect on this text: for example, it splits
    "...set off to-morrow.'-If he would say so..." into two sentences at
    the quote.

11. [FIXED] Training and scoring treated line breaks differently.
    During training, bigrams carried on across line breaks, but when
    scoring every line started again from ''. The '' start token was
    counted only once per file, so P(first word | '') was based on a count
    of 1. Now training and scoring both work on sentences, and every
    sentence starts with START_TOKEN ('<s>') in both. The start token is
    counted once per sentence (5,759 times for Austen), and it isn't
    counted as a word, so unigram probabilities still add up to 1.

Usability
---------
12. Command-line handling (lines 318-331).
    The -test flag has to be the second argument. There's no usage
    message: running with no arguments only prints "Please supply an
    authorlist", and wrong arguments print "Wrong number of arguments" or
    "No -test flag" without saying what's expected.

13. Author names in the output.
    -test mode prints the file name ("austen.txt") instead of the author.
    (Default mode now uses name.removesuffix('.txt'), part of the #7 / #8
    rewrite; it used re.sub('.txt', '', ...), where '.' matches any
    character.)

14. File names in the author list are relative to the current directory,
    not to the author list's folder, so the script only works when run from
    this folder.

15. It doesn't check for or download the NLTK tokenizer data. On a fresh
    machine nltk.word_tokenize() raises LookupError. (The top-level README
    has the download command; the POS tagger downloads anything that's
    missing automatically.)

Cleanup
-------
16. Unused and broken code (lines 163-184, 222-259).
    - test_prob_of_sentence(), test_good_turing() and test_lm() call
      LanguageModel() without the required author argument (TypeError), and
      use paths like '/txt_ascii/austen.txt', which point to the root of
      the file system. They also don't call smooth_models(), which every
      model now needs before it can score anything, and
      test_prob_of_sentence() passes a string to get_prob_of_sentence(),
      which now takes a list of tokens.
    - get_prob_of_w() (an add-one version) is only used by a test function.
      It also checks "token in tokens" against a list, which gets very slow
      on large files.

17. Small cleanups:
    - "from tokenize import Double" (line 17) is unused. It comes from
      Python's own source-code tokenizer, not NLTK.
    - The "x in d.keys()" counting blocks in LanguageModel.train() can use
      collections.Counter.
    (The docstring's "Good-Touring" was corrected as part of #4 and #5, and
    the unused tokens list was removed with the #10 rewrite.)

18. Unused files: words, txt_ascii/, txt_utf8/ and test.py.
    txt_ascii/ and txt_utf8/ are the original author files, so they still
    contain the test.txt passages. Don't train on them with test.txt, or
    the test passages will be in the training data again.
