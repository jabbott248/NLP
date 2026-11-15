# Created by Jacob Abbott
"""
About:
encoding type program runs on:
    Only tested and run on ascii
data cleaning methods you employed:
    Only using nltk word tokenizing
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

    def create_lm(self, fname: str):
        tokens = []
        with open(fname, 'r') as f:
            prev_token = ''
            self.unigrams[prev_token] = 1
            for line in f:
                new_tokens = nltk.word_tokenize(line)
                for token in new_tokens:
                    # Add to all tokens
                    tokens.append(token)
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

        # includes the '' start token, so unigram probabilities add up to 1
        self.total_words = sum(self.unigrams.values())
        # N_c: how many different bigrams were seen exactly c times. This is
        # computed once counting is finished, so each bigram is counted
        # under its final count only.
        self.frequencies = Counter(self.bigrams.values())

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
#   prev_word: the word before it ('' at the start of a line)
def gt_get_prob_of_w(lm: LanguageModel, word: str, prev_word: str):
    prev_count = lm.context_counts.get(prev_word, 0)
    if prev_count == 0:
        return lm.unigram_prob(word)
    c = lm.bigrams.get((prev_word, word), 0)
    if c > 0:
        return lm.gt_adjusted_count(c) / prev_count
    return lm.backoff_weight[prev_word] * lm.unigram_prob(word)

# Scores a sentence with lm. Returns its cost: the sum of -log10
# P(word | prev_word) over every word. Lower cost = more likely, so the
# author whose model gives the lowest cost is the best guess.
# Costs are added because -log(a*b) = -log(a) + -log(b); working in logs
# stops the product of many small probabilities underflowing to 0.
def get_prob_of_sentence(lm: LanguageModel, sentence: str):
    tokens = nltk.word_tokenize(sentence)
    total_cost = 0
    prev_token = ''
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


# No test flag
# automatically extract a development from the author data, then train on the remaining training data
def run_no_test_flag(authorlist: str):
    # Extract all texts from authorlist file
    txt_fnames = []
    dev_txt_fnames = []
    training_txt_fnames = []
    with open (authorlist, 'r') as f:
        for line in f:
            line = line.strip()
            txt_fnames.append(line)
    
    for fname in txt_fnames:
        # For each text file, create 2 new text files
        dev_txt_name = 'dev_' + fname
        dev_txt_fnames.append(dev_txt_name)
        train_txt_name = 'training_' + fname
        training_txt_fnames.append(train_txt_name)
        with open(fname, 'r') as txt_file:
            with open(dev_txt_name, 'w') as dev_txt:
                with open(train_txt_name, 'w') as training_txt:
                    count = 0
                    for line in txt_file:
                        if count%10 == 0: # if %5, write to training
                            training_txt.write(line)
                        else:            # else, write to development
                            dev_txt.write(line)
                        count+=1


    # Create language models based on development_texts
    models = []
    for text in dev_txt_fnames:
        new_model = LanguageModel(author=text)
        #print('New Model name: %s'%new_model.author)
        new_model.create_lm(text)
        models.append(new_model)
    smooth_models(models)

    # Test language models on training_texts
    for fname in training_txt_fnames:
        correct_guess_count = 0
        total_guesses = 0
        author_name = re.sub('.txt', '', fname)
        author_name = re.sub('training_', '', author_name)

        with open(fname, 'r') as training_txt:
            for line in training_txt:
                # Blank lines have no words to score, so every model would
                # tie at a cost of 0. Skip them instead of counting a guess.
                if not line.strip():
                    continue
                lowest_cost = math.inf   # lowest cost = most likely author
                author = ''
                for lm in models:
                    new_cost = get_prob_of_sentence(lm, line)
                    if new_cost < lowest_cost:
                        lowest_cost = new_cost
                        author = lm.author
                author_guess = re.sub('.txt', '', author)
                author_guess = re.sub('dev_', '', author_guess)
                if author_guess == author_name:
                    correct_guess_count+=1
                total_guesses+=1
            print('%s   %i/%i correct'%(author_name,correct_guess_count,total_guesses))
    return 0

# with test flag
# use the entirety of data in each author file to train a language model, 
# then output classification results for each line in the given file testfile
def run_with_test_flag(authorlist: str, test_fname: str):
    # create a list of file names to create language models from
    txt_fnames = []
    models = []
    with open (authorlist, 'r') as f:
        for line in f:
            line = line.strip()
            txt_fnames.append(line)
    # Create all language models
    for fname in txt_fnames:
        new_model = LanguageModel(author=fname)
        new_model.create_lm(fname)
        models.append(new_model)
    smooth_models(models)

    # Test each language model on the fname
    with open(test_fname, 'r') as f:
        # count is the line number in the file, so the output still lines up
        # with the input when blank lines are skipped
        for count, line in enumerate(f, start=1):
            line = line.strip()
            if not line:    # blank line: nothing to classify
                continue
            lowest_cost = math.inf   # lowest cost = most likely author
            author = ''
            for lm in models:
                new_cost = get_prob_of_sentence(lm, line)
                if new_cost < lowest_cost:
                    lowest_cost = new_cost
                    author = lm.author

            print('Line %i: %s'%(count, author))

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


