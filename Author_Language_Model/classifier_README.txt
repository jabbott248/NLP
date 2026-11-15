Author Language Model Classifier
================================

What it does
------------
These scripts guess who wrote a piece of text. They build one language
model per author from sample text, then score each sentence of new text
with every model and pick the author whose model fits best.

  bigram_classifier.py   uses bigram models (each word given the one
                         before it)
  ngram_classifier.py    the same script, but the n-gram order can be set
                         from 1 to 5 with --ngram N (default 2, bigrams).
                         See "n-gram models" below.

The two give identical results with bigrams. "How it works" describes
bigram_classifier.py, and "n-gram models" explains what ngram_classifier.py
does differently.

The sample data is four authors, with excerpts from several of their books
(10,000-20,000 lines each):

  austen.txt    Jane Austen: Emma, Sense and Sensibility, Pride and Prejudice
  dickens.txt   Charles Dickens: A Tale of Two Cities, David Copperfield
  wilde.txt     Oscar Wilde: The Picture of Dorian Gray, The Canterville
                Ghost and other stories, The Portrait of Mr. W. H.
  tolstoy.txt   Leo Tolstoy: War and Peace, Anna Karenina

Setup
-----
Follow the setup in the top-level README.md (install requirements.txt).
The scripts only need nltk and its 'punkt' / 'punkt_tab' tokenizer data.
If the tokenizer data is missing, they download it (to ~/nltk_data)
on the first run. If a download fails, they stop and say which package
failed; on macOS that usually means the certificate step in the top-level
README was skipped.

How to run
----------
  Usage:

      python3 bigram_classifier.py <authorlist>                  (accuracy)
      python3 bigram_classifier.py <authorlist> -test <textfile> (guess authors)
      python3 bigram_classifier.py --help                        (help)

  ngram_classifier.py takes the same arguments plus --ngram N, the n-gram
  order from 1 to 5 (default 2). It can go anywhere on the command line:

      python3 ngram_classifier.py <authorlist> --ngram 3
      python3 ngram_classifier.py <authorlist> -test <textfile> --ngram 3

  Everything below applies to both scripts. The expected output is for
  bigrams; see "n-gram models" for the other orders.

  -test can also be written --test, and can go before or after the author
  list. Running with no arguments prints the help. If an argument is
  missing or a file can't be found, the script says what's wrong and shows
  the usage.

  The author list has one author text file per line. File names in it are
  relative to the author list's own folder, so the script works from any
  directory, e.g. from the top of the repo:

      python3 Author_Language_Model/bigram_classifier.py Author_Language_Model/authorlist1

  Each author is named after their file without the extension
  (austen.txt -> austen). The list needs at least two authors.

  Default mode: split each author's text, train on one part and measure
  accuracy on the rest:

      python3 bigram_classifier.py authorlist1

  Each author's text is split into sentences. Every 10th sentence
  (TEST_EVERY) is held out for testing and the model is trained on the
  other 90%. The split is done in memory; no files are written. Each test
  sentence is classified, and it prints how many of each author's test
  sentences were guessed correctly.

  Expected output:

      austen   484/576 correct
      dickens   556/694 correct
      wilde   494/677 correct
      tolstoy   900/1064 correct

  That's 80.8% overall. Guessing at random would give 25%.

  -test mode: train on the full author files and classify every sentence
  of another file:

      python3 bigram_classifier.py authorlist1 -test test.txt

  Output is one line per sentence: its number, the author whose model
  scored best, and the start of the sentence.

  test.txt has one passage per author. Each passage was taken out of its
  author file, so none of it is in the training data:

      Sentences  1-7   Austen, Emma
      Sentences  8-13  Dickens, David Copperfield
      Sentences 14-26  Wilde, The Picture of Dorian Gray
      Sentences 27-33  Tolstoy, Anna Karenina

  Expected output (abridged):

      Sentence 1:   austen    "There is one thing, Emma, which a man can alwa...
      Sentence 2:   austen    It is Frank Churchill's duty to pay this attent...
      ...
      Sentence 7:   tolstoy   '-If he would say so to her at once, in the ton...
      Sentence 8:   dickens   I thought it affecting to see such a sturdy fel...
      ...
      Sentence 22:  dickens   Yes.
      ...
      Sentence 33:  tolstoy   For the first time in his life he knew the bitt...

  26 of the 33 sentences are right: Austen 6/7, Dickens 3/6, Wilde 11/13
  and Tolstoy 6/7. The misses include a one-word sentence ("Yes."), a
  fragment where sent_tokenize split a sentence at a quote (sentence 7),
  and half of the Dickens passage, whose short first-person sentences ("I
  was affected by the story altogether.") could belong to anyone.

  To try your own text, put it in a file (it can be hard-wrapped;
  separate paragraphs with blank lines) and pass that instead of
  test.txt.

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
     For the few previous words (22-38 per author) where every word after
     them was seen 5 or more times, Good-Turing frees up nothing. There,
     unseen words get Good-Turing's overall unseen share (N_1 / N) and the
     seen words are scaled down to make room (seen_scale), so the total is
     still 1.

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

n-gram models (ngram_classifier.py)
-----------------------------------
An n-gram model predicts each word from the n-1 words before it (its
context): a trigram model uses P(word | two previous words). With
--ngram 1 it uses each word's own probability and ignores word order.
ngram_classifier.py generalizes each step above:

  - Training: each sentence is padded with n-1 start tokens
    (<s> <s> the cat ... for trigrams), and the model counts k-grams for
    every order k from 2 to n, each with its own N_c table.
  - Smoothing: Katz backoff is applied one order at a time, from bigrams
    up. An unseen k-gram backs off to the next order down, using the
    context minus its first word:
        P(word | a b) = alpha(a b) * P(word | b)        if "a b word" unseen
    and P(word | b) backs off to P(word) in the same way. So an unseen
    n-gram backs off one word at a time until it reaches a context the
    model has seen. The fallback scaling and the shared vocabulary for
    unseen words work exactly as for bigrams, at every order.
  - Scoring: each word is scored with the n-1 words before it.

Results on the same data (default mode, and the 33 test.txt sentences):

  --ngram   default mode        test.txt   run time
     1      78.2% (2355/3011)   23/33      ~1 s
     2      80.8% (2434/3011)   26/33      ~2 s     (default)
     3      80.8% (2433/3011)   25/33      ~2 s
     4      80.1% (2412/3011)   25/33      ~3 s
     5      79.8% (2402/3011)   25/33      ~4 s

Longer n-grams don't help with this much data. Austen's text has 146,000
tokens but 108,000 different trigrams, so most trigrams were seen only
once, a new sentence's trigrams are usually unseen, and the model backs off
to bigrams most of the time. Word choice alone (--ngram 1) already gets
78%, so word order only adds a few points. A larger corpus per author is
what would let n > 2 pay off, which is why the default is 2.

Files in this folder
--------------------
  bigram_classifier.py   the bigram script
  ngram_classifier.py    the n-gram script (--ngram 1-5)
  authorlist1            the author files to use, one file name per line
  austen.txt, dickens.txt, wilde.txt, tolstoy.txt
                         the author text (ASCII), used for training
  test.txt               sample input for -test mode: one passage per
                         author, removed from the author files (see
                         "How to run")
  txt_ascii/             the original author files, before the test.txt
                         passages were taken out (not used)
  txt_utf8/              UTF-8 versions of the original author files (not
                         used). Both folders still contain the test.txt
                         passages, so don't train on them and test with
                         test.txt.
  words                  English word list (not used; a copy of
                         SpellChecker/dictionary/words)
  test_bigram_classifier.py, test_ngram_classifier.py
                         unit and end-to-end tests (see "Running the tests")
  test.py                scratch file for trying things out (not used)

Running the tests
-----------------
From inside this folder:

      python3 -m unittest test_bigram_classifier test_ngram_classifier

  (or run either file with python3). test_bigram_classifier.py has 33
  tests and takes about 4 seconds; test_ngram_classifier.py has 29 and
  takes about 9.

  Most use tiny made-up corpora, so the expected counts and
  probabilities can be worked out by hand: sentence splitting, the start
  token, N_c, Good-Turing adjusted counts, that probabilities add up to 1,
  backoff, scoring, the author list and the NLTK download check. The rest
  train on the real author files: the model checks, both modes, and the
  command line (including running from another folder).

  The n-gram tests also check that the probabilities add up to 1 at every
  order from 1 to 5, that unseen n-grams back off to the right shorter
  context, and that with n = 2 every probability and the default-mode
  output are exactly the same as bigram_classifier.py's.

  To check that the tests catch real mistakes, bugs that were fixed in
  this script were put back into bigram_classifier.py one at a time:
  bigrams looked up with the two words swapped, costs multiplied instead
  of added, the highest cost winning, a broken Good-Turing N_c table,
  unseen words getting the whole unseen probability each, bigrams running
  across sentences, author files looked up relative to the current
  directory, and the fallback scaling. The tests catch all of them. The
  n-gram tests were checked the same way against typical n-gram mistakes
  (backing off to the wrong context, a context window one word short, too
  little start padding, one N_c table for every order, smoothing the orders
  top-down, and the fallback scaling), and catch all of them.
