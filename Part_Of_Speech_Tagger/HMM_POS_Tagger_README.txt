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
                          The vocabulary includes NLTK's English word list
                          plus words seen at least twice in Brown. Other
                          words are replaced with the token 'UNK'.

  2. hmm_sequence()     - Loads model.dat and scores an already-tagged
                          sentence (word/TAG word/TAG ...), i.e. how likely
                          that tag sequence is under the model.

  3. hmm_viterbi()      - Takes untagged text (one sentence per line) and
                          uses the Viterbi algorithm to find the most likely
                          tag for each word. It also reports the probability
                          of that best tagging. Run it with --tag.

Files in this folder
--------------------
  HMM_POS_Tagger.py         the script
  model.dat                 a pre-built model (output of build mode)
  test_tagged_sentence.txt  sample input for sequence mode (word/TAG format)
  test_untagged.txt         sample input for --tag mode (hmm_viterbi)

Setup
-----
1. Install NLTK:

       pip3 install nltk

2. macOS only (python.org installer): Python can't verify SSL certificates
   until you run the bundled certificate installer, so NLTK's downloads fail
   with "CERTIFICATE_VERIFY_FAILED". Run this once (change the version to
   match yours):

       open "/Applications/Python 3.14/Install Certificates.command"

3. NLTK data: nothing to do. On its first run the script downloads anything
   that's missing (brown, words, punkt, punkt_tab) to ~/nltk_data. If a
   download fails, it stops and says which package failed. On macOS that
   usually means step 2 was skipped.

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
      Total probability: 1.0446386439769002e-68

  Small numbers like this are normal: it's the joint probability of every
  word and tag in a 25-word sentence. Compare scores between taggings of
  the same sentence; a better tagging gets a higher score. Issue #8 still
  affects the result.

  Tag an untagged sentence (one sentence per line):

      python3 HMM_POS_Tagger.py --tag model.dat test_untagged.txt

  Expected output:

      Running HMM Viterbi on test_untagged.txt using model model.dat
      The/AT Fulton/NP-TL County/NN-TL Grand/JJ-TL Jury/NN-TL said/VBD Friday/NR an/AT investigation/NN of/IN Atlanta/NP 's/, recent/JJ primary/JJ election/NN produced/VBD no/AT evidence/NN that/CS any/DTI irregularities/NNS took/VBD place/NN ./.
      Probability of best tagging: 4.84368761480502e-71

  Rebuild model.dat after changing the tagger so it uses the updated
  vocabulary. Frequent punctuation and word forms now have their own
  emissions instead of sharing the 'UNK' emission. The probability is
  P(words, tags) for the best tagging only, not P(words) summed over all
  taggings.


Known Issues
============
These were confirmed by running the script with NLTK 3.10.3 and Python 3.14.
Line numbers refer to HMM_POS_Tagger.py.

Problems that stop it from running
----------------------------------
1. SSL certificate error on macOS (environment problem, not a code bug).
   All NLTK downloads fail until you run the certificate installer in Setup
   step 2.

2. [FIXED] Missing 'punkt_tab' (download_nltk_data).
   NLTK 3.8.2 and later need the 'punkt_tab' resource for word_tokenize(),
   but the script only checked for and downloaded 'punkt', so
   nltk.word_tokenize() raised LookupError. It now downloads both 'punkt'
   (older NLTK) and 'punkt_tab' (newer NLTK).

3. [FIXED] Failed downloads were ignored (download_nltk_data).
   nltk.download() returns False when it fails, but the script carried on
   and crashed later with a long, confusing LookupError. It now checks the
   return value and exits with a clear message naming the package.

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
   transition costs, then converts the total back to a probability with
   10 ** -total_cost. Before the fix, the sample sentence scored 10916655,
   which isn't a valid probability.
   (This answers the TODO on line 75: logs prevent floating-point underflow
   when you multiply many small probabilities, and abs() turns the negative
   logs into positive costs.)
   Still to clean up: read_matrix_file replaces a cost of 0.0 with 0.0003
   (lines 131-132). A cost of 0 means probability 1, which is valid. The
   replacement only existed to stop the product becoming 0. Now that costs
   are added it makes almost no difference, but it can be removed.

6. [FIXED] Crashed with KeyError on unseen combinations.
   There was no smoothing, so any tag transition that never appeared in
   Brown, or any tag never seen with 'UNK', crashed the program. Example:
   the input "Hello/UH zzzq/FW-NN" gave
       KeyError: ('FW-NN', ('UH', 'FW-NN'))
   Now any missing transition or emission gets UNSEEN_COST (a cost of 10,
   probability 1e-10), and that input scores 2.18e-15. For a word never
   seen with its tag, the emission still falls back to P(UNK | tag) first.
   This is a fixed default cost, not true smoothing; add-one (Laplace)
   smoothing would spread probability more evenly but makes the model
   file much larger.

7. [FIXED] Start-of-sentence transitions were never saved.
   create_matricies() counted (tag, '<ST>') pairs, but the transition loop
   only looped over tags in tag_counts, and '<ST>' wasn't one of them. Those
   counts were thrown away, and hmm_sequence() used 1 as a placeholder for
   the first transition. Now '<ST>' is counted once per sentence, the model
   stores P(tag | <ST>) (211 start tags, summing to 1), and both
   hmm_sequence() and hmm_viterbi() use it. Models built before this fix
   have no start transitions, so rebuild model.dat.
   Still to do: there is no end-of-sentence tag, so P(</S> | last tag) is
   not part of the score.

8. The tokenizer breaks tagged input (lines 154-156).
   nltk.word_tokenize() splits word/TAG tokens on punctuation. With the
   sample file:
       "Atlanta's/NP$" -> "Atlanta's/NP", "$"   (wrong tag)
       "``/``"         -> "``", "/", "``"       (silently skipped)
       "./."           -> "./", "."             (silently skipped)
   The input is already split by spaces, so use line.split(). Use
   token.rsplit('/', 1) instead of the regex so words that contain '/'
   still work.

hmm_viterbi() (all fixed)
-------------------------
These are all fixed. hmm_viterbi() is now a real Viterbi tagger. It indexes
emissions by word, adds -log10 costs, keeps backpointers, and traces back
the best path. Tag pairs never seen in Brown get UNSEEN_COST (probability
1e-10, see #6), and the first word uses the start probabilities from #7.

9. [FIXED] It always returns empty tags.
   best_tag and best_prob are reset inside the loop over every emission
   key. After the loop, best_tag holds only whatever the
   last emission key gave, which is almost always ''.

10. [FIXED] It isn't a Viterbi algorithm.
    - prev_tag is set but never used in the scoring, so transitions don't
      depend on the previous tag.
    - It picks the largest cost (">"). With -log costs, the best
      tag has the smallest total, and costs should be added, not multiplied.
    - It chooses one word at a time (greedy). Real Viterbi fills a table of
      best scores for every (position, tag) plus backpointers, then traces
      back from the end to find the best overall path.
    - For every word it loops over all ~32,000 emission keys and then all
      ~8,000 transition keys, which is very slow. Indexing emissions by
      word (word -> {tag: cost}) fixes this.

11. [FIXED] Only the last line of the input file is used, and an
    empty file causes NameError on 'tokens'.

12. [FIXED] main() couldn't run it. Use
    "python3 HMM_POS_Tagger.py --tag model.dat file.txt".

Model quality / smaller issues
------------------------------
13. [FIXED] Too many words become UNK.
    The vocabulary now includes words absent from NLTK's English word list
    when they occur at least twice in Brown. Words absent from both lists
    and seen only once in Brown remain mapped to 'UNK'. The same vocabulary
    is used when training, scoring and tagging. Rebuild model.dat to apply
    the change.

14. Slow nested loops when building (lines 71-87). Each tag is compared
    against every (word, tag) pair. Looping over wordTag_counts once and
    dividing by tag_counts[tag] gives the same result. (Building still only
    takes a few seconds, so this is low priority.)

15. Storing the model as text and reading it back with a regex (lines
    91-138) is fragile. json or pickle would be simpler and safer.

16. Small cleanups: the regexes are compiled inside loops (lines 122 and
    156), "x in d.keys()" can be "x in d" or collections.Counter, the output
    path is hard-coded to 'model.dat', and "matricies" should be
    "matrices".
