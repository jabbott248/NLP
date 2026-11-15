# Created by Jacob Abbott
"""
About:
encoding type program runs on:
    Only tested and run on ascii
data cleaning methods you employed:
    Paragraphs split into sentences with nltk.sent_tokenize, then nltk word tokenizing
information is in Language Models (bigrams, trigrams, etc):
    n-grams of any order from 1 to 5 (--ngram N, default 2: bigrams)
method of smoothing you are using:
    Good-Turing smoothing with Katz backoff (unigram Good-Turing for unseen words)
No other tweaks were made

Same as bigram_classifier.py, except that the language models can use
n-grams of any order from 1 to MAX_N. An n-gram model predicts each word
from the n-1 words before it, and backs off to shorter contexts (down to a
single word's own probability) for word sequences it has never seen.
"""

import argparse
import os
import sys
import re
import nltk
import math
from collections import Counter

# Good-Turing adjusted counts are only used for n-gram counts below this.
# Above it the raw count is used: Good-Turing estimates get unreliable
# for large counts, where N_(c+1) is often 0 or very noisy.
GT_MAX_COUNT = 5

# Token placed before the first word of every sentence (n-1 times for an
# n-gram model), so the model learns which words tend to start a sentence.
# It can't clash with a real token: nltk.word_tokenize() splits "<s>" into
# "<", "s", ">".
START_TOKEN = '<s>'

# Default mode holds out every TEST_EVERY-th sentence of each author for
# testing and trains on the rest (a 90/10 split).
TEST_EVERY = 10

# n-gram orders the command line accepts, and the default
MIN_N = 1
MAX_N = 5
DEFAULT_N = 2

# Language model to pass to methods to calculate probabilities.
# For an n-gram model, "order k" means k-grams: a word and the k-1 words
# before it (its context). Orders 2 to n each have their own counts and
# smoothing tables, keyed by k.
class LanguageModel():
    def __init__(self, author: str, n: int = DEFAULT_N):
        if not MIN_N <= n <= MAX_N:
            raise ValueError(f"n must be between {MIN_N} and {MAX_N}, not {n}")
        self.n = n
        self.unigrams = Counter()   # token -> count
        self.ngrams = {k: Counter() for k in range(2, n + 1)}   # k -> {k-gram tuple: count}
        self.frequencies = {}       # k -> N_c: number of k-grams seen exactly c times
        self.total_words = 0
        self.unknown_unigram = 1    # P(word) for each word never seen in training
        self.seen_word_share = 1    # total P(word) shared by the words seen in training
        self.context_counts = {}    # k -> {context: number of k-grams starting with it}
        self.backoff_weight = {}    # k -> {context: Katz backoff weight alpha}
        self.seen_scale = {}        # k -> {context: scale for seen k-grams (fallback only)}
        self.author = author

    # Counts unigrams and k-grams (k = 2 to n) in sentences, a list of
    # sentences that are each a list of tokens. Every sentence starts with
    # n-1 START_TOKENs, so the first words have a full context, n-grams
    # never run from one sentence into the next, and the model learns which
    # words start a sentence.
    def train(self, sentences: list):
        padding = [START_TOKEN] * (self.n - 1)
        for tokens in sentences:
            self.unigrams.update(tokens)
            padded = padding + tokens
            for k in range(2, self.n + 1):
                # every k-gram that ends on a real token
                self.ngrams[k].update(tuple(padded[i - k + 1:i + 1])
                                      for i in range(self.n - 1, len(padded)))

        # START_TOKEN is not a unigram (it's never predicted as a word), so
        # this is the number of real tokens and unigram probabilities add up
        # to 1
        self.total_words = sum(self.unigrams.values())
        # N_c for each order: how many different k-grams were seen exactly c
        # times. This is computed once counting is finished, so each k-gram
        # is counted under its final count only.
        self.frequencies = {k: Counter(counts.values()) for k, counts in self.ngrams.items()}

    # Builds the model from every sentence in a text file
    def create_lm(self, fname: str):
        self.train([tokens for _, tokens in read_sentences(fname)])

    # Works out the smoothed probabilities. Call this after create_lm(), once
    # every author's model has been built:
    #   vocab_size: number of distinct words across all authors' text, plus
    #               1 for words that none of them used
    # The vocabulary has to be shared so that every model splits its
    # probability for unseen words over the same set of words.
    def smooth(self, vocab_size: int):
        # Good-Turing for unigrams: the share of tokens that were seen only
        # once (N_1 / N) estimates the total probability of all the words
        # this author never used. It is split evenly between those words.
        one_count_unigrams = sum(1 for count in self.unigrams.values() if count == 1)
        unseen_mass = max(one_count_unigrams, 1) / self.total_words
        unseen_words = max(vocab_size - len(self.unigrams), 1)
        self.seen_word_share = 1 - unseen_mass
        self.unknown_unigram = unseen_mass / unseen_words

        # Katz backoff, one order at a time from bigrams up. Good-Turing
        # lowers each seen k-gram's count from c to c*, which frees up some
        # probability after every context. That leftover is shared among the
        # words never seen after the context, in proportion to their
        # probability under the next order down (the context minus its first
        # word), so that P(w | context) adds up to 1 over all words w. Each
        # order backs off to the one below it, so they're built bottom-up.
        for k in range(2, self.n + 1):
            context_counts = Counter()      # Count(context) as a k-gram start
            discounted_counts = Counter()   # sum of c* over words seen after it
            seen_lower_mass = Counter()     # sum of the order k-1 P(w) over those words
            for gram, c in self.ngrams[k].items():
                context, word = gram[:-1], gram[-1]
                context_counts[context] += c
                discounted_counts[context] += self.gt_adjusted_count(k, c)
                seen_lower_mass[context] += gt_get_prob_of_w(self, word, context[1:])
            # Good-Turing's overall share for unseen k-grams (N_1 / N), capped
            # at half. It's only near 1 for tiny training texts where almost
            # every k-gram was seen once.
            unseen_share = min(self.frequencies[k][1] / sum(self.ngrams[k].values()), 0.5)
            self.context_counts[k] = context_counts
            self.backoff_weight[k] = {}
            self.seen_scale[k] = {}
            for context, total in context_counts.items():
                seen_share = discounted_counts[context] / total
                if seen_share < 1:
                    leftover = 1 - seen_share
                else:
                    # Nothing was discounted after this context: every word
                    # after it was seen GT_MAX_COUNT or more times (or its c*
                    # wasn't smaller than c). Reserve unseen_share for unseen
                    # words so they don't get 0, and scale the seen words
                    # down to make room, so the total is still 1.
                    leftover = unseen_share
                    self.seen_scale[k][context] = (1 - leftover) / seen_share
                self.backoff_weight[k][context] = leftover / (1 - seen_lower_mass[context])

    # Good-Turing adjusted count c* = (c + 1) * N_(c+1) / N_c for a k-gram
    # seen c times (c >= 1), using the N_c table for order k
    def gt_adjusted_count(self, k: int, c: int):
        n_c = self.frequencies[k][c]
        n_c1 = self.frequencies[k][c + 1]
        if c >= GT_MAX_COUNT or n_c1 == 0:
            return c        # large or gappy counts: keep the raw count
        return (c + 1) * n_c1 / n_c

    # P(word) on its own. Seen words share seen_word_share in proportion to
    # their counts, and each unseen word gets unknown_unigram, so all words
    # in the vocabulary together add up to 1.
    def unigram_prob(self, word: str):
        if word in self.unigrams:
            return self.seen_word_share * self.unigrams[word] / self.total_words
        return self.unknown_unigram


# Reads a text file and splits it into sentences. Paragraphs are separated
# by blank lines. Inside a paragraph the hard-wrapped lines are joined back
# together, then nltk.sent_tokenize() splits it into sentences.
# Returns a list of (sentence text, list of tokens).
def read_sentences(fname: str):
    with open(fname, 'r') as f:
        text = f.read()
    sentences = []
    for paragraph in re.split(r'\n\s*\n', text):
        paragraph = ' '.join(paragraph.split())   # join the wrapped lines
        for sentence in nltk.sent_tokenize(paragraph):
            tokens = nltk.word_tokenize(sentence)
            if tokens:
                sentences.append((sentence, tokens))
    return sentences

# Author name for a text file: its name without the folder or extension,
# e.g. 'texts/austen.txt' -> 'austen'
def author_name(fname: str):
    return os.path.splitext(os.path.basename(fname))[0]

# Reads the author list: one author text file per line (blank lines are
# ignored). Relative file names are looked up in the author list's own
# folder, so the script works from any directory. Exits with an error if
# the list has fewer than two authors or a file is missing.
def read_authorlist(authorlist: str):
    folder = os.path.dirname(authorlist)
    with open(authorlist, 'r') as f:
        fnames = [os.path.join(folder, line.strip()) for line in f if line.strip()]
    if len(fnames) < 2:
        sys.exit(f"Error: {authorlist} lists {len(fnames)} author file(s); "
                 "it needs at least two to choose between.")
    for fname in fnames:
        if not os.path.isfile(fname):
            sys.exit(f"Error: author file '{fname}' (listed in {authorlist}) "
                     "not found. File names are relative to the author "
                     "list's folder.")
    return fnames


# Returns P(word | context): the probability that word comes right after
# the words in context, using lm's n-gram counts with Good-Turing smoothing
# and Katz backoff. context is a tuple of the words before word, and can be
# up to n-1 long (shorter means a lower order: () is the unigram P(word)).
# For an order-k context (k = len(context) + 1):
#   - seen k-gram:    P = c* / Count(context), where c* is the k-gram's
#                     Good-Turing adjusted count (scaled down after the few
#                     contexts where nothing was discounted; see smooth())
#   - unseen k-gram:  P = alpha(context) * P(word | context[1:]), where alpha
#                     shares out the probability Good-Turing took from the
#                     seen k-grams
#   - context never seen in training: P = P(word | context[1:])
# so an unseen n-gram backs off one word at a time until it reaches a
# context the model has seen, or the unigram P(word).
#   lm:      the author's language model to use
#   word:    the current word
#   context: the words before it (START_TOKENs at the start of a sentence)
def gt_get_prob_of_w(lm: LanguageModel, word: str, context: tuple):
    if not context:
        return lm.unigram_prob(word)
    k = len(context) + 1
    context_count = lm.context_counts[k].get(context, 0)
    if context_count == 0:
        return gt_get_prob_of_w(lm, word, context[1:])
    c = lm.ngrams[k].get(context + (word,), 0)
    if c > 0:
        return lm.gt_adjusted_count(k, c) / context_count * lm.seen_scale[k].get(context, 1)
    return lm.backoff_weight[k][context] * gt_get_prob_of_w(lm, word, context[1:])

# Scores a sentence (a list of tokens) with lm. Returns its cost: the sum
# of -log10 P(word | the n-1 words before it) over every word. Lower cost =
# more likely, so the author whose model gives the lowest cost is the best
# guess. Costs are added because -log(a*b) = -log(a) + -log(b); working in
# logs stops the product of many small probabilities underflowing to 0.
def get_prob_of_sentence(lm: LanguageModel, tokens: list):
    total_cost = 0
    history = [START_TOKEN] * (lm.n - 1) + tokens
    for i in range(lm.n - 1, len(history)):
        context = tuple(history[i - lm.n + 1:i])
        word_prob = gt_get_prob_of_w(lm, history[i], context)
        total_cost += -math.log10(word_prob)

    return total_cost

# Smooths every model using one vocabulary shared by all of them: every word
# in any author's text, plus 1 for words that none of them used.
def smooth_models(models: list):
    vocab = set()
    for lm in models:
        vocab.update(lm.unigrams)
    for lm in models:
        lm.smooth(len(vocab) + 1)


# Returns the model (author) whose cost for the sentence is lowest, i.e.
# the author most likely to have written it
def classify(models: list, tokens: list):
    return min(models, key=lambda lm: get_prob_of_sentence(lm, tokens))


# No test flag
# Splits each author's text into sentences, holds out every TEST_EVERY-th
# sentence for testing and trains n-gram models on the rest, then prints how
# many of each author's test sentences were guessed correctly. The split is
# done in memory, so no files are written.
def run_no_test_flag(authorlist: str, n: int = DEFAULT_N):
    models = []
    test_sets = []
    for fname in read_authorlist(authorlist):
        sentences = [tokens for _, tokens in read_sentences(fname)]
        train = [s for i, s in enumerate(sentences) if i % TEST_EVERY != 0]
        test = [s for i, s in enumerate(sentences) if i % TEST_EVERY == 0]
        new_model = LanguageModel(author=author_name(fname), n=n)
        new_model.train(train)
        models.append(new_model)
        test_sets.append(test)
    smooth_models(models)

    for lm, test in zip(models, test_sets):
        correct_guess_count = sum(1 for tokens in test if classify(models, tokens) is lm)
        print('%s   %i/%i correct'%(lm.author, correct_guess_count, len(test)))
    return 0

# with test flag
# use the entirety of data in each author file to train an n-gram language
# model, then output the most likely author for each sentence in the file
# test_fname
def run_with_test_flag(authorlist: str, test_fname: str, n: int = DEFAULT_N):
    models = []
    for fname in read_authorlist(authorlist):
        new_model = LanguageModel(author=author_name(fname), n=n)
        new_model.create_lm(fname)
        models.append(new_model)
    smooth_models(models)

    width = max(len(lm.author) for lm in models)   # line up the output
    for count, (sentence, tokens) in enumerate(read_sentences(test_fname), start=1):
        author = classify(models, tokens).author
        preview = sentence if len(sentence) <= 50 else sentence[:47] + '...'
        print('%-13s %-*s   %s'%('Sentence %i:'%count, width, author, preview))

def download_nltk_data():
    # 'punkt' is used by NLTK before 3.8.2, 'punkt_tab' by 3.8.2 and later
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                sys.exit(f"Error: could not download NLTK package '{package}'. "
                         "Check your internet connection (on macOS, see the "
                         "certificate step in the top-level README) and try "
                         "again.")

def interface():
    parser = argparse.ArgumentParser(
        prog='ngram_classifier.py',
        description='Guess which author wrote each sentence of a text, using one\n'
                    'n-gram language model per author.',
        epilog='examples:\n'
               '  python3 ngram_classifier.py authorlist1                 '
               'measure accuracy (bigrams)\n'
               '  python3 ngram_classifier.py authorlist1 --ngram 3       '
               'measure accuracy (trigrams)\n'
               '  python3 ngram_classifier.py authorlist1 -test test.txt  '
               'guess the author of each sentence in test.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('authorlist',
                        help='file listing one author text file per line '
                             '(paths relative to the list\'s folder)')
    parser.add_argument('-test', '--test', metavar='TEXTFILE', dest='test_fname',
                        help='train on the full author files and guess the '
                             'author of each sentence in TEXTFILE. Without '
                             'it, every 10th sentence of each author is held '
                             'out and the accuracy is printed.')
    parser.add_argument('--ngram', metavar='N', type=int, default=DEFAULT_N,
                        choices=range(MIN_N, MAX_N + 1),
                        help=f'n-gram order, {MIN_N}-{MAX_N}: 1 = single words, '
                             f'2 = bigrams, 3 = trigrams, ... (default '
                             f'{DEFAULT_N})')
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()

    # Check the files before downloading anything, so mistakes fail fast
    if not os.path.isfile(args.authorlist):
        parser.error(f"author list '{args.authorlist}' not found")
    if args.test_fname is not None and not os.path.isfile(args.test_fname):
        parser.error(f"test file '{args.test_fname}' not found")
    read_authorlist(args.authorlist)   # exits if an author file is missing

    download_nltk_data()
    if args.test_fname is None:
        run_no_test_flag(args.authorlist, args.ngram)
    else:
        run_with_test_flag(args.authorlist, args.test_fname, args.ngram)


if __name__ == "__main__":
    interface()
