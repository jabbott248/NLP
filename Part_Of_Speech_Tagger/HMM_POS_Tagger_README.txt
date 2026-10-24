HMM Part-of-Speech Tagger
=========================

What it does
------------
HMM_POS_Tagger.py is a bigram Hidden Markov Model (HMM) part-of-speech tagger
trained on the Brown corpus (via NLTK). It has three parts:

  1. hmm_model_build()  - Reads the tagged Brown corpus and counts:
                            - how often each tag occurs
                            - how often each (word, tag) pair occurs
                            - how often each tag follows another tag
                          From these it computes two tables and writes them
                          to model.dat:
                            m1 = emission probabilities   P(word | tag)
                            m2 = transition probabilities P(tag | previous tag)
                          Probabilities are stored as -log10(p) ("costs").
                          Words not in NLTK's English word list are replaced
                          with the token 'UNK'.

  2. hmm_sequence()     - Loads model.dat and scores an already-tagged
                          sentence (word/TAG word/TAG ...), i.e. how likely
                          that tag sequence is under the model.

  3. hmm_viterbi()      - Meant to take an untagged sentence and predict the
                          best tag for each word. It doesn't work yet (see
                          Known Issues) and main() never calls it.

Files in this folder
--------------------
  HMM_POS_Tagger.py         the script
  model.dat                 a pre-built model (output of build mode)
  test_tagged_sentence.txt  sample input for sequence mode (word/TAG format)
  test_untagged.txt         sample input meant for hmm_viterbi()

Setup
-----
1. Install NLTK:

       pip3 install nltk

2. macOS only (python.org installer): Python can't verify SSL certificates
   until you run the bundled certificate installer, so NLTK's downloads fail
   with "CERTIFICATE_VERIFY_FAILED". Run this once (change the version to
   match yours):

       open "/Applications/Python 3.14/Install Certificates.command"

3. Download the NLTK data. The script tries to do this itself, but it asks
   for 'punkt' when newer NLTK versions (3.8.2 and later) need 'punkt_tab'
   (see issue #2), so download it all yourself:

       python3 -c "import nltk; [nltk.download(p) for p in ('brown','words','punkt','punkt_tab')]"

How to run
----------
Run from inside this folder. model.dat is always written to the current
directory.

  Build the model (overwrites model.dat; takes a couple of seconds):

      python3 HMM_POS_Tagger.py

  Score a tagged sentence with a model:

      python3 HMM_POS_Tagger.py model.dat test_tagged_sentence.txt

  Expected output:

      Running HMM sequence on test_tagged_sentence.txt using model model.dat
      Total probability: 7.219426280057337e-68

  Small numbers like this are normal: it's the joint probability of every
  word and tag in a 25-word sentence. Compare scores between taggings of
  the same sentence; a better tagging gets a higher score. Issues #6 to #8
  still affect the result.

  Run the (broken) tagger by hand:

      python3 -c "import HMM_POS_Tagger as h; h.hmm_viterbi('model.dat', 'test_untagged.txt')"

  Currently prints a list of empty strings: POS_sequence=['', '', ...]


Known Issues
============
These were confirmed by running the script with NLTK 3.10.3 and Python 3.14.
Line numbers refer to HMM_POS_Tagger.py.

Problems that stop it from running
----------------------------------
1. SSL certificate error on macOS (environment problem, not a code bug).
   All NLTK downloads fail until you run the certificate installer in Setup
   step 2.

2. Missing 'punkt_tab' (download_nltk_data, lines 236-239).
   NLTK 3.8.2 and later need the 'punkt_tab' resource for word_tokenize(),
   but the script only checks for and downloads 'punkt', so
   nltk.word_tokenize() raises LookupError. Add a check and download for
   'tokenizers/punkt_tab'.

3. Failed downloads are ignored (download_nltk_data).
   nltk.download() returns False when it fails, but the script carries on
   and crashes later with a long, confusing LookupError. Check the return
   value and exit with a clear message.

Wrong results / crashes in hmm_sequence()
-----------------------------------------
4. [FIXED] Transition lookup was backwards.
   The model stores transitions as (prev_tag, (tag, prev_tag)), but the
   lookup used (tag, (prev_tag, tag)), so it read P(prev_tag | tag) instead
   of P(tag | prev_tag). It now uses:
       transition_prob = transition_m[(prev_tag, (tag, prev_tag))]

5. [FIXED] Costs were multiplied instead of added.
   The stored values are -log10(p), and log-costs must be added:
   -log(a*b) = -log(a) + -log(b). hmm_sequence() now adds the emission and
   transition costs (the first word's transition cost is 0, meaning
   probability 1), then converts the total back to a probability with
   10 ** -total_cost. Before the fix, the sample sentence scored 10916655,
   which isn't a valid probability.
   (This answers the TODO on line 68: logs prevent floating-point underflow
   when you multiply many small probabilities, and abs() turns the negative
   logs into positive costs.)
   Still to clean up: read_matrix_file replaces a cost of 0.0 with 0.0003
   (lines 120-121). A cost of 0 means probability 1, which is valid. The
   replacement only existed to stop the product becoming 0. Now that costs
   are added it makes almost no difference, but it can be removed.

6. Crashes with KeyError on unseen combinations (lines 158 and 164).
   There is no smoothing, so any tag transition that never appeared in
   Brown, or any tag never seen with 'UNK', crashes the program. Example:
   the input "Hello/UH zzzq/FW-NN" gives
       KeyError: ('FW-NN', ('UH', 'FW-NN'))
   Use add-one (Laplace) smoothing or a default cost for missing keys.

7. Start-of-sentence transitions are never saved (lines 71-76).
   create_matricies() counts (tag, '<ST>') pairs, but the transition loop
   only loops over tags in tag_counts, and '<ST>' isn't one of them. Those
   counts are thrown away, and hmm_sequence() uses 1 as a placeholder for
   the first transition (line 161). Fix: count '<ST>' once per sentence so
   P(tag | <ST>) can be computed. An end-of-sentence tag is also worth
   adding.

8. The tokenizer breaks tagged input (lines 143-145).
   nltk.word_tokenize() splits word/TAG tokens on punctuation. With the
   sample file:
       "Atlanta's/NP$" -> "Atlanta's/NP", "$"   (wrong tag)
       "``/``"         -> "``", "/", "``"       (silently skipped)
       "./."           -> "./", "."             (silently skipped)
   The input is already split by spaces, so use line.split(). Use
   token.rsplit('/', 1) instead of the regex so words that contain '/'
   still work.

hmm_viterbi() doesn't work (lines 173-225)
------------------------------------------
9. It always returns empty tags.
   best_tag and best_prob are reset inside the loop over every emission
   key (lines 206-207). After the loop, best_tag holds only whatever the
   last emission key gave, which is almost always ''.

10. It isn't a Viterbi algorithm.
    - prev_tag is set but never used in the scoring, so transitions don't
      depend on the previous tag.
    - It picks the largest cost (">" on line 216). With -log costs, the best
      tag has the smallest total, and costs should be added, not multiplied.
    - It chooses one word at a time (greedy). Real Viterbi fills a table of
      best scores for every (position, tag) plus backpointers, then traces
      back from the end to find the best overall path.
    - For every word it loops over all ~32,000 emission keys and then all
      ~8,000 transition keys, which is very slow. Indexing emissions by
      word (word -> {tag: cost}) fixes this.

11. Only the last line of the input file is used (lines 183-185), and an
    empty file causes NameError on 'tokens'.

12. main() can't run it. There's no command-line option for viterbi mode.
    Something like "python3 HMM_POS_Tagger.py --tag model.dat file.txt"
    would work.

Model quality / smaller issues
------------------------------
13. Too many words become UNK (lines 25 and 36).
    The vocabulary comes from NLTK's English word list, which leaves out
    most inflected forms, names and punctuation. About 25% of Brown tokens
    become 'UNK'. In the sample sentence, "fulton", "atlanta's",
    "irregularities" and "." are all UNK. A common alternative is to build
    the vocabulary from the training data (e.g. words seen 2+ times) and
    map rare words to UNK.

14. Slow nested loops when building (lines 64-76). Each tag is compared
    against every (word, tag) pair. Looping over wordTag_counts once and
    dividing by tag_counts[tag] gives the same result. (Building still only
    takes a few seconds, so this is low priority.)

15. Storing the model as text and reading it back with a regex (lines
    81-127) is fragile. json or pickle would be simpler and safer.

16. Small cleanups: the regexes are compiled inside loops (lines 111 and
    145), "x in d.keys()" can be "x in d" or collections.Counter, the output
    path is hard-coded to 'model.dat', and "matricies" should be
    "matrices".
