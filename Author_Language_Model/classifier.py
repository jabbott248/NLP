# Created by Jacob Abbott
"""
About:
encoding type program runs on:
    Only tested and run on ascii
data cleaning methods you employed:
    Paragraphs split into sentences with nltk.sent_tokenize, then nltk word tokenizing
information is in Language Models (bigrams, trigrams, etc):
    Bigrams only
method of smoothing you are using:
    Good-Turing smoothing with Katz backoff (unigram Good-Turing for unseen words)
No other tweaks were made
"""

import sys
import re
from tokenize import Double
import nltk
import math
from collections import Counter

# Good-Turing adjusted counts are only used for bigram counts below this.
# Above it the raw count is used: Good-Turing estimates get unreliable
# for large counts, where N_(c+1) is often 0 or very noisy.
GT_MAX_COUNT = 5

# Token placed before the first word of every sentence, so the model learns
# which words tend to start a sentence. It can't clash with a real token:
# nltk.word_tokenize() splits "<s>" into "<", "s", ">".
START_TOKEN = '<s>'

# Default mode holds out every TEST_EVERY-th sentence of each author for
# testing and trains on the rest (a 90/10 split).
TEST_EVERY = 10

# Language model to pass to methods to calculate probabilities
class LanguageModel():
    def __init__(self, author: str):
        self.unigrams = {}
        self.bigrams = {}
        self.frequencies = {}       # N_c: number of bigrams seen exactly c times
        self.total_words = 0
        self.unknown_unigram = 1    # P(word) for each word never seen in training
        self.seen_word_share = 1    # total P(word) shared by the words seen in training
        self.context_counts = {}    # prev_word -> number of bigrams starting with it
        self.backoff_weight = {}    # prev_word -> Katz backoff weight alpha
        self.author = author

    # Counts unigrams and bigrams in sentences, a list of sentences that are
    # each a list of tokens. Every sentence starts again from START_TOKEN, so
    # bigrams never run from one sentence into the next, and
    # P(first word | START_TOKEN) is learned from every sentence start.
    def train(self, sentences: list):
        for tokens in sentences:
            prev_token = START_TOKEN
            for token in tokens:
                # update unigram count
                if token in self.unigrams.keys():
                    self.unigrams[token] += 1  # if token already exists, increase its count
                else:
                    self.unigrams[token] = 1   # if token does not exist, add it to the dictionary
                # update bigram count
                if (prev_token, token) in self.bigrams.keys():
                    self.bigrams[(prev_token, token)] += 1
                else:
                    self.bigrams[(prev_token, token)] = 1
                # update the previous token for bigram counts
                prev_token = token

        # START_TOKEN is not a unigram (it's never predicted as a word), so
        # this is the number of real tokens and unigram probabilities add up
        # to 1
        self.total_words = sum(self.unigrams.values())
        # N_c: how many different bigrams were seen exactly c times. This is
        # computed once counting is finished, so each bigram is counted
        # under its final count only.
        self.frequencies = Counter(self.bigrams.values())

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

        # Katz backoff. Good-Turing lowers each seen bigram's count from c to
        # c*, which frees up some probability after every prev_word. That
        # leftover is shared among the words never seen after prev_word, in
        # proportion to their unigram probability, so that P(w | prev_word)
        # adds up to 1 over all words w.
        context_counts = Counter()      # Count(prev_word) as a bigram start
        discounted_counts = Counter()   # sum of c* over words seen after it
        seen_unigram_mass = Counter()   # sum of P(w) over words seen after it
        for (prev_word, word), c in self.bigrams.items():
            context_counts[prev_word] += c
            discounted_counts[prev_word] += self.gt_adjusted_count(c)
            seen_unigram_mass[prev_word] += self.unigram_prob(word)
        unseen_share = self.frequencies[1] / sum(self.bigrams.values())
        for prev_word, total in context_counts.items():
            leftover = 1 - discounted_counts[prev_word] / total
            if leftover <= 0:
                # Every word after prev_word was seen GT_MAX_COUNT or more
                # times, so nothing was discounted. Use Good-Turing's overall
                # share for unseen bigrams (N_1 / N) so they don't get 0.
                leftover = unseen_share
            self.backoff_weight[prev_word] = leftover / (1 - seen_unigram_mass[prev_word])
        self.context_counts = context_counts

    # Good-Turing adjusted count c* = (c + 1) * N_(c+1) / N_c for a bigram
    # seen c times (c >= 1)
    def gt_adjusted_count(self, c: int):
        n_c = self.frequencies[c]
        n_c1 = self.frequencies[c + 1]
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

# Reads the author list: one author text file per line
def read_authorlist(authorlist: str):
    with open(authorlist, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# Calculate probability that wn given wn-1 in a text file
def get_prob_of_w(fname: str, w0: str, w1: str):
    tokens = []
    v = 0          # V: Vocabulay, number of unique words
    w_bigram_count = 0
    w1_count = 0
    # P(wn | wn-1): (Count(wn-1 wn)+1) / (Count(wn-1) + V)
    with open(fname, 'r') as f:
        prev_word = ''                              # pointer to previous word
        for line in f:                              # iterate through each line of the file
            new_tokens = nltk.word_tokenize(line)   
            for token in new_tokens:                # iterate through each word of the line
                if not token in tokens:
                    v = v+1
                if token == w0 and prev_word == w1:
                    w_bigram_count = w_bigram_count+1
                if token == w1:
                    w1_count = w1_count+1           # increase the count of w1
                tokens.append(token)                # attach new token to global token list
                prev_word = token                   # move prev word pointer

    prob = (w_bigram_count + 1) / (w1_count + v)
    return prob


# Returns P(word | prev_word): the probability that word comes right after
# prev_word, using lm's bigram counts with Good-Turing smoothing and Katz
# backoff:
#   - seen bigram:   P = c* / Count(prev_word), where c* is the bigram's
#                    Good-Turing adjusted count
#   - unseen bigram: P = alpha(prev_word) * P(word), where alpha shares out
#                    the probability Good-Turing took from the seen bigrams
#   - prev_word never seen in training: P = P(word)
#   lm:        the author's language model to use
#   word:      the current word
#   prev_word: the word before it (START_TOKEN at the start of a sentence)
def gt_get_prob_of_w(lm: LanguageModel, word: str, prev_word: str):
    prev_count = lm.context_counts.get(prev_word, 0)
    if prev_count == 0:
        return lm.unigram_prob(word)
    c = lm.bigrams.get((prev_word, word), 0)
    if c > 0:
        return lm.gt_adjusted_count(c) / prev_count
    return lm.backoff_weight[prev_word] * lm.unigram_prob(word)

# Scores a sentence (a list of tokens) with lm. Returns its cost: the sum
# of -log10 P(word | prev_word) over every word. Lower cost = more likely,
# so the author whose model gives the lowest cost is the best guess.
# Costs are added because -log(a*b) = -log(a) + -log(b); working in logs
# stops the product of many small probabilities underflowing to 0.
def get_prob_of_sentence(lm: LanguageModel, tokens: list):
    total_cost = 0
    prev_token = START_TOKEN
    for token in tokens:
        word_prob = gt_get_prob_of_w(lm, token, prev_token)
        total_cost += -math.log10(word_prob)
        prev_token = token

    return total_cost

def test_prob_of_sentence():
    test_sentence = 'Oh, the difference of situation and habit! I wish you would try to understand what an amiable young man may be likely to feel in directly opposing those, whom as child and boy he has been looking up to all his life.'
    austen_lm = LanguageModel()
    austen_lm.create_lm('/txt_ascii/austen.txt')
    austen_prob = get_prob_of_sentence(austen_lm, test_sentence)

    dickens_lm = LanguageModel()
    dickens_lm.create_lm('/txt_ascii/dickens.txt')
    dickens_prob = get_prob_of_sentence(dickens_lm, test_sentence)

    tolstoy_lm = LanguageModel()
    tolstoy_lm.create_lm('/txt_ascii/tolstoy.txt')
    tolstoy_prob = get_prob_of_sentence(tolstoy_lm, test_sentence)

    print('austen:  %f'%austen_prob)
    print('dickens: %f'%dickens_prob)
    print('tolstoy: %f'%tolstoy_prob)

def test_good_turing():
    lm = LanguageModel()
    lm.create_lm('/txt_ascii/tolstoy.txt')
    prob = gt_get_prob_of_w(lm, 'may', 'family')
    print('probability: %.6f'%prob)



def test_lm():
    lm = LanguageModel()
    lm.create_lm('/txt_ascii/austen.txt')
    print(lm.frequencies)

    

def test_get_prob_of_w():
    file_name = '/txt_ascii/austen.txt'
    res = get_prob_of_w(file_name, 'be', 'may')
    print(f'{res=}')



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
# sentence for testing and trains on the rest, then prints how many of
# each author's test sentences were guessed correctly. The split is done in
# memory, so no files are written.
def run_no_test_flag(authorlist: str):
    models = []
    test_sets = []
    for fname in read_authorlist(authorlist):
        sentences = [tokens for _, tokens in read_sentences(fname)]
        train = [s for i, s in enumerate(sentences) if i % TEST_EVERY != 0]
        test = [s for i, s in enumerate(sentences) if i % TEST_EVERY == 0]
        new_model = LanguageModel(author=fname)
        new_model.train(train)
        models.append(new_model)
        test_sets.append(test)
    smooth_models(models)

    for lm, test in zip(models, test_sets):
        correct_guess_count = sum(1 for tokens in test if classify(models, tokens) is lm)
        author_name = lm.author.removesuffix('.txt')
        print('%s   %i/%i correct'%(author_name, correct_guess_count, len(test)))
    return 0

# with test flag
# use the entirety of data in each author file to train a language model,
# then output the most likely author for each sentence in the file test_fname
def run_with_test_flag(authorlist: str, test_fname: str):
    models = []
    for fname in read_authorlist(authorlist):
        new_model = LanguageModel(author=fname)
        new_model.create_lm(fname)
        models.append(new_model)
    smooth_models(models)

    for count, (sentence, tokens) in enumerate(read_sentences(test_fname), start=1):
        author = classify(models, tokens).author
        preview = sentence if len(sentence) <= 50 else sentence[:47] + '...'
        print('%-13s %-12s %s'%('Sentence %i:'%count, author, preview))

def interface():
    # check for correct number of command line arguments
    if (len(sys.argv) == 1):
        print('Please supply an authorlist')
    elif (len(sys.argv) == 2):
        # RUN WITHOUT A TEST FLAG
        run_no_test_flag(sys.argv[1])
    elif (len(sys.argv) == 4):  # automatically extract a development from the author data, then train on the remaining training data  
        if sys.argv[2] == '-test':
            run_with_test_flag(sys.argv[1], sys.argv[3])
        else:
            print('No -test flag')
    else: # check for test flag
        print('Wrong number of arguments')


if __name__ == "__main__":
    #nltk.download('words')
    interface()


