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
                          to model_dat.json:
                            m1 = emission probabilities   P(word | tag)
                            m2 = transition probabilities P(tag | previous tag)
                          The JSON objects "m1" and "m2" map tags and tag
                          transitions to -log10(p) costs.
                          The vocabulary includes NLTK's English word list
                          plus words seen at least twice in Brown. Other
                          words are replaced with the token 'UNK'.

  2. hmm_sequence()     - Loads model_dat.json and scores an already-tagged
                          sentence (word/TAG word/TAG ...), i.e. how likely
                          that tag sequence is under the model.

  3. hmm_viterbi()      - Takes untagged text (one sentence per line) and
                          uses the Viterbi algorithm to find the most likely
                          tag for each word. It also reports the probability
                          of that best tagging. Run it with --tag.

Files in this folder
--------------------
  HMM_POS_Tagger.py         the script
  model_dat.json            a pre-built model (output of build mode)
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
Run from inside this folder. model_dat.json is always written to the current
directory.

  Build the model (overwrites model_dat.json; takes a couple of seconds):

      python3 HMM_POS_Tagger.py

  Score a tagged sentence with a model:

      python3 HMM_POS_Tagger.py model_dat.json test_tagged_sentence.txt

  Expected output:

      Running HMM sequence on test_tagged_sentence.txt using model model_dat.json
      Total probability: 1.59524156182099e-75

  Small numbers like this are normal: it's the joint probability of every
  word and tag in a 25-word sentence. Compare scores between taggings of
  the same sentence; a better tagging gets a higher score. Issue #8 still
  affects the result.

  Tag an untagged sentence (one sentence per line):

      python3 HMM_POS_Tagger.py --tag model_dat.json test_untagged.txt

  Expected output:

      Running HMM Viterbi on test_untagged.txt using model model_dat.json
      The/AT Fulton/NP-TL County/NN-TL Grand/JJ-TL Jury/NN-TL said/VBD Friday/NR an/AT investigation/NN of/IN Atlanta/NP 's/NP$ recent/JJ primary/JJ election/NN produced/VBD no/AT evidence/NN that/CS any/DTI irregularities/NNS took/VBD place/NN ./.
      Probability of best tagging: 2.859351752525453e-78

  Rebuild model_dat.json after changing the tagger so it uses the updated
  vocabulary. Frequent punctuation and word forms now have their own
  emissions instead of sharing the 'UNK' emission. The probability is
  P(words, tags) for the best tagging only, not P(words) summed over all
  taggings.

Tag reference (Brown tagset)
----------------------------
The tagger outputs Brown corpus tags. The full list below comes from NLTK
(nltk.help.brown_tagset). To see example words for any tag, run:

    python3 -c "import nltk; nltk.download('tagsets_json'); nltk.help.brown_tagset('NN')"

brown_tagset() also accepts a regex, e.g. 'NN.*' for every noun tag.

Reading compound tags
  Many tags the model outputs (e.g. NP-TL) aren't in the list as-is. They
  are a base tag with one or more of these added:

    -TL     word is part of a title     NN-TL  = common noun in a title
    -HL     word is in a headline       NN-HL  = common noun in a headline
    -NC     cited word (mentioned,      NN-NC  = common noun, cited
            not used)
    FW-     foreign word                FW-NN  = foreign common noun
    *       negated (n't, not)          BEZ*   = "isn't"
    +       contraction of two words    PPS+BEZ = "it's"

  To read one, drop the -TL/-HL/-NC suffix and look up what's left. For
  example NP-TL = NP (proper noun, singular) in a title. FW- and + forms
  are often listed directly below; if not, look up each part.

Tags not in NLTK's list
  ``      opening quotation mark
  ''      closing quotation mark
  '       single quotation mark
  NIL     word that was left untagged in the Brown corpus
  A few rare tags (FW-IN+AT-T, PPSS+BER-N, WDT+BER+PP) contain pieces
  (AT-T, BER-N, PP) that appear to be typos in the corpus itself.

Full tag list
  (          opening parenthesis
  )          closing parenthesis
  *          negator
  ,          comma
  --         dash
  .          sentence terminator
  :          colon
  ABL        determiner/pronoun, pre-qualifier
  ABN        determiner/pronoun, pre-quantifier
  ABX        determiner/pronoun, double conjunction or pre-quantifier
  AP         determiner/pronoun, post-determiner
  AP$        determiner/pronoun, post-determiner, genitive
  AP+AP      determiner/pronoun, post-determiner, hyphenated pair
  AT         article
  BE         verb 'to be', infinitive or imperative
  BED        verb 'to be', past tense, 2nd person singular or all persons
             plural
  BED*       verb 'to be', past tense, 2nd person singular or all persons
             plural, negated
  BEDZ       verb 'to be', past tense, 1st and 3rd person singular
  BEDZ*      verb 'to be', past tense, 1st and 3rd person singular, negated
  BEG        verb 'to be', present participle or gerund
  BEM        verb 'to be', present tense, 1st person singular
  BEM*       verb 'to be', present tense, 1st person singular, negated
  BEN        verb 'to be', past participle
  BER        verb 'to be', present tense, 2nd person singular or all persons
             plural
  BER*       verb 'to be', present tense, 2nd person singular or all persons
             plural, negated
  BEZ        verb 'to be', present tense, 3rd person singular
  BEZ*       verb 'to be', present tense, 3rd person singular, negated
  CC         conjunction, coordinating
  CD         numeral, cardinal
  CD$        numeral, cardinal, genitive
  CS         conjunction, subordinating
  DO         verb 'to do', uninflected present tense, infinitive or imperative
  DO*        verb 'to do', uninflected present tense or imperative, negated
  DO+PPSS    verb 'to do', past or present tense + pronoun, personal,
             nominative, not 3rd person singular
  DOD        verb 'to do', past tense
  DOD*       verb 'to do', past tense, negated
  DOZ        verb 'to do', present tense, 3rd person singular
  DOZ*       verb 'to do', present tense, 3rd person singular, negated
  DT         determiner/pronoun, singular
  DT$        determiner/pronoun, singular, genitive
  DT+BEZ     determiner/pronoun + verb 'to be', present tense, 3rd person
             singular
  DT+MD      determiner/pronoun + modal auxillary
  DTI        determiner/pronoun, singular or plural
  DTS        determiner/pronoun, plural
  DTS+BEZ    pronoun, plural + verb 'to be', present tense, 3rd person singular
  DTX        determiner, pronoun or double conjunction
  EX         existential there
  EX+BEZ     existential there + verb 'to be', present tense, 3rd person
             singular
  EX+HVD     existential there + verb 'to have', past tense
  EX+HVZ     existential there + verb 'to have', present tense, 3rd person
             singular
  EX+MD      existential there + modal auxillary
  FW-*       foreign word: negator
  FW-AT      foreign word: article
  FW-AT+NN   foreign word: article + noun, singular, common
  FW-AT+NP   foreign word: article + noun, singular, proper
  FW-BE      foreign word: verb 'to be', infinitive or imperative
  FW-BER     foreign word: verb 'to be', present tense, 2nd person singular or
             all persons plural
  FW-BEZ     foreign word: verb 'to be', present tense, 3rd person singular
  FW-CC      foreign word: conjunction, coordinating
  FW-CD      foreign word: numeral, cardinal
  FW-CS      foreign word: conjunction, subordinating
  FW-DT      foreign word: determiner/pronoun, singular
  FW-DT+BEZ  foreign word: determiner + verb 'to be', present tense, 3rd person
             singular
  FW-DTS     foreign word: determiner/pronoun, plural
  FW-HV      foreign word: verb 'to have', present tense, not 3rd person
             singular
  FW-IN      foreign word: preposition
  FW-IN+AT   foreign word: preposition + article
  FW-IN+NN   foreign word: preposition + noun, singular, common
  FW-IN+NP   foreign word: preposition + noun, singular, proper
  FW-JJ      foreign word: adjective
  FW-JJR     foreign word: adjective, comparative
  FW-JJT     foreign word: adjective, superlative
  FW-NN      foreign word: noun, singular, common
  FW-NN$     foreign word: noun, singular, common, genitive
  FW-NNS     foreign word: noun, plural, common
  FW-NP      foreign word: noun, singular, proper
  FW-NPS     foreign word: noun, plural, proper
  FW-NR      foreign word: noun, singular, adverbial
  FW-OD      foreign word: numeral, ordinal
  FW-PN      foreign word: pronoun, nominal
  FW-PP$     foreign word: determiner, possessive
  FW-PPL     foreign word: pronoun, singular, reflexive
  FW-PPL+VBZ foreign word: pronoun, singular, reflexive + verb, present tense,
             3rd person singular
  FW-PPO     pronoun, personal, accusative
  FW-PPO+IN  foreign word: pronoun, personal, accusative + preposition
  FW-PPS     foreign word: pronoun, personal, nominative, 3rd person singular
  FW-PPSS    foreign word: pronoun, personal, nominative, not 3rd person
             singular
  FW-PPSS+HV foreign word: pronoun, personal, nominative, not 3rd person
             singular + verb 'to have', present tense, not 3rd person singular
  FW-QL      foreign word: qualifier
  FW-RB      foreign word: adverb
  FW-RB+CC   foreign word: adverb + conjunction, coordinating
  FW-TO+VB   foreign word: infinitival to + verb, infinitive
  FW-UH      foreign word: interjection
  FW-VB      foreign word: verb, present tense, not 3rd person singular,
             imperative or infinitive
  FW-VBD     foreign word: verb, past tense
  FW-VBG     foreign word: verb, present participle or gerund
  FW-VBN     foreign word: verb, past participle
  FW-VBZ     foreign word: verb, present tense, 3rd person singular
  FW-WDT     foreign word: WH-determiner
  FW-WPO     foreign word: WH-pronoun, accusative
  FW-WPS     foreign word: WH-pronoun, nominative
  HV         verb 'to have', uninflected present tense, infinitive or
             imperative
  HV*        verb 'to have', uninflected present tense or imperative, negated
  HV+TO      verb 'to have', uninflected present tense + infinitival to
  HVD        verb 'to have', past tense
  HVD*       verb 'to have', past tense, negated
  HVG        verb 'to have', present participle or gerund
  HVN        verb 'to have', past participle
  HVZ        verb 'to have', present tense, 3rd person singular
  HVZ*       verb 'to have', present tense, 3rd person singular, negated
  IN         preposition
  IN+IN      preposition, hyphenated pair
  IN+PPO     preposition + pronoun, personal, accusative
  JJ         adjective
  JJ$        adjective, genitive
  JJ+JJ      adjective, hyphenated pair
  JJR        adjective, comparative
  JJR+CS     adjective + conjunction, coordinating
  JJS        adjective, semantically superlative
  JJT        adjective, superlative
  MD         modal auxillary
  MD*        modal auxillary, negated
  MD+HV      modal auxillary + verb 'to have', uninflected form
  MD+PPSS    modal auxillary + pronoun, personal, nominative, not 3rd person
             singular
  MD+TO      modal auxillary + infinitival to
  NN         noun, singular, common
  NN$        noun, singular, common, genitive
  NN+BEZ     noun, singular, common + verb 'to be', present tense, 3rd person
             singular
  NN+HVD     noun, singular, common + verb 'to have', past tense
  NN+HVZ     noun, singular, common + verb 'to have', present tense, 3rd person
             singular
  NN+IN      noun, singular, common + preposition
  NN+MD      noun, singular, common + modal auxillary
  NN+NN      noun, singular, common, hyphenated pair
  NNS        noun, plural, common
  NNS$       noun, plural, common, genitive
  NNS+MD     noun, plural, common + modal auxillary
  NP         noun, singular, proper
  NP$        noun, singular, proper, genitive
  NP+BEZ     noun, singular, proper + verb 'to be', present tense, 3rd person
             singular
  NP+HVZ     noun, singular, proper + verb 'to have', present tense, 3rd person
             singular
  NP+MD      noun, singular, proper + modal auxillary
  NPS        noun, plural, proper
  NPS$       noun, plural, proper, genitive
  NR         noun, singular, adverbial
  NR$        noun, singular, adverbial, genitive
  NR+MD      noun, singular, adverbial + modal auxillary
  NRS        noun, plural, adverbial
  OD         numeral, ordinal
  PN         pronoun, nominal
  PN$        pronoun, nominal, genitive
  PN+BEZ     pronoun, nominal + verb 'to be', present tense, 3rd person
             singular
  PN+HVD     pronoun, nominal + verb 'to have', past tense
  PN+HVZ     pronoun, nominal + verb 'to have', present tense, 3rd person
             singular
  PN+MD      pronoun, nominal + modal auxillary
  PP$        determiner, possessive
  PP$$       pronoun, possessive
  PPL        pronoun, singular, reflexive
  PPLS       pronoun, plural, reflexive
  PPO        pronoun, personal, accusative
  PPS        pronoun, personal, nominative, 3rd person singular
  PPS+BEZ    pronoun, personal, nominative, 3rd person singular + verb 'to be',
             present tense, 3rd person singular
  PPS+HVD    pronoun, personal, nominative, 3rd person singular + verb 'to
             have', past tense
  PPS+HVZ    pronoun, personal, nominative, 3rd person singular + verb 'to
             have', present tense, 3rd person singular
  PPS+MD     pronoun, personal, nominative, 3rd person singular + modal
             auxillary
  PPSS       pronoun, personal, nominative, not 3rd person singular
  PPSS+BEM   pronoun, personal, nominative, not 3rd person singular + verb 'to
             be', present tense, 1st person singular
  PPSS+BER   pronoun, personal, nominative, not 3rd person singular + verb 'to
             be', present tense, 2nd person singular or all persons plural
  PPSS+BEZ   pronoun, personal, nominative, not 3rd person singular + verb 'to
             be', present tense, 3rd person singular
  PPSS+BEZ*  pronoun, personal, nominative, not 3rd person singular + verb 'to
             be', present tense, 3rd person singular, negated
  PPSS+HV    pronoun, personal, nominative, not 3rd person singular + verb 'to
             have', uninflected present tense
  PPSS+HVD   pronoun, personal, nominative, not 3rd person singular + verb 'to
             have', past tense
  PPSS+MD    pronoun, personal, nominative, not 3rd person singular + modal
             auxillary
  PPSS+VB    pronoun, personal, nominative, not 3rd person singular + verb 'to
             verb', uninflected present tense
  QL         qualifier, pre
  QLP        qualifier, post
  RB         adverb
  RB$        adverb, genitive
  RB+BEZ     adverb + verb 'to be', present tense, 3rd person singular
  RB+CS      adverb + conjunction, coordinating
  RBR        adverb, comparative
  RBR+CS     adverb, comparative + conjunction, coordinating
  RBT        adverb, superlative
  RN         adverb, nominal
  RP         adverb, particle
  RP+IN      adverb, particle + preposition
  TO         infinitival to
  TO+VB      infinitival to + verb, infinitive
  UH         interjection
  VB         verb, base: uninflected present, imperative or infinitive
  VB+AT      verb, base: uninflected present or infinitive + article
  VB+IN      verb, base: uninflected present, imperative or infinitive +
             preposition
  VB+JJ      verb, base: uninflected present, imperative or infinitive +
             adjective
  VB+PPO     verb, uninflected present tense + pronoun, personal, accusative
  VB+RP      verb, imperative + adverbial particle
  VB+TO      verb, base: uninflected present, imperative or infinitive +
             infinitival to
  VB+VB      verb, base: uninflected present, imperative or infinitive;
             hypenated pair
  VBD        verb, past tense
  VBG        verb, present participle or gerund
  VBG+TO     verb, present participle + infinitival to
  VBN        verb, past participle
  VBN+TO     verb, past participle + infinitival to
  VBZ        verb, present tense, 3rd person singular
  WDT        WH-determiner
  WDT+BER    WH-determiner + verb 'to be', present tense, 2nd person singular
             or all persons plural
  WDT+BER+PP WH-determiner + verb 'to be', present, 2nd person singular or all
             persons plural + pronoun, personal, nominative, not 3rd person
             singular
  WDT+BEZ    WH-determiner + verb 'to be', present tense, 3rd person singular
  WDT+DO+PPS WH-determiner + verb 'to do', uninflected present tense + pronoun,
             personal, nominative, not 3rd person singular
  WDT+DOD    WH-determiner + verb 'to do', past tense
  WDT+HVZ    WH-determiner + verb 'to have', present tense, 3rd person singular
  WP$        WH-pronoun, genitive
  WPO        WH-pronoun, accusative
  WPS        WH-pronoun, nominative
  WPS+BEZ    WH-pronoun, nominative + verb 'to be', present, 3rd person
             singular
  WPS+HVD    WH-pronoun, nominative + verb 'to have', past tense
  WPS+HVZ    WH-pronoun, nominative + verb 'to have', present tense, 3rd person
             singular
  WPS+MD     WH-pronoun, nominative + modal auxillary
  WQL        WH-qualifier
  WRB        WH-adverb
  WRB+BER    WH-adverb + verb 'to be', present, 2nd person singular or all
             persons plural
  WRB+BEZ    WH-adverb + verb 'to be', present, 3rd person singular
  WRB+DO     WH-adverb + verb 'to do', present, not 3rd person singular
  WRB+DOD    WH-adverb + verb 'to do', past tense
  WRB+DOD*   WH-adverb + verb 'to do', past tense, negated
  WRB+DOZ    WH-adverb + verb 'to do', present tense, 3rd person singular
  WRB+IN     WH-adverb + preposition
  WRB+MD     WH-adverb + modal auxillary


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
   Logs prevent floating-point underflow when multiplying many small
   probabilities, and abs() turns the negative logs into positive costs.

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
   have no start transitions, so rebuild model_dat.json.
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
    "python3 HMM_POS_Tagger.py --tag model_dat.json file.txt".

Model quality / smaller issues
------------------------------
13. [FIXED] Too many words become UNK.
    The vocabulary now includes words absent from NLTK's English word list
    when they occur at least twice in Brown. Words absent from both lists
    and seen only once in Brown remain mapped to 'UNK'. The same vocabulary
    is used when training, scoring and tagging. Rebuild model_dat.json to
    apply the change.

14. [FIXED] Slow nested loop when building the emission matrix.
    Each observed (word, tag) pair is now processed once and divided by its
    tag count, instead of comparing every tag with every pair. This produces
    the same emission costs with less work.

15. [FIXED] Storing the model as text and parsing it with a regex was fragile.
    The model is now stored as JSON with "m1" and "m2" objects. The reader
    reconstructs the same tuple-key dictionaries used by hmm_sequence() and
    hmm_viterbi().

16. Small cleanups: the regexes are compiled inside loops (lines 122 and
    156), "x in d.keys()" can be "x in d" or collections.Counter, the output
    path is hard-coded to 'model_dat.json', and "matricies" should be
    "matrices".
