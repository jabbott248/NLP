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
    Good-Touring Smoothing
No other tweaks were made
"""

import sys
import re
from tokenize import Double
import nltk
import math

# Language model to pass to methods to calculate probabilities
class LanguageModel():
    def __init__(self, author: str):
        self.unigrams = {}
        self.bigrams = {}
        self.frequencies = {}
        self.unknown_unigram = 1
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
                    # update frequencies
                    if self.bigrams[(prev_token, token)] in self.frequencies.keys():
                        self.frequencies[self.bigrams[(prev_token, token)]] += 1
                    else:
                        self.frequencies[self.bigrams[(prev_token, token)]] = 1
                    # update the previous token for bigram counts
                    prev_token = token

        total_words = len(tokens)
        self.frequencies[1] = 1 # In case no words with frequncy 1 appear in a senetence
        self.frequencies[0] = self.frequencies[1] / total_words
        one_count_unigrams = 0
        for key in self.unigrams.keys():
            if self.unigrams[key] == 1:
                one_count_unigrams = one_count_unigrams + 1
        self.unknown_unigram = one_count_unigrams / total_words


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


def gt_get_prob_of_w(lm: LanguageModel, w0: str, w1: str):
    # Account for key Error in bigrams and frequincies, if no key then sub N(1)/N
    if (w1,w0) in lm.bigrams.keys():                            # if the bigram has a count 1 or greater
        c = lm.bigrams[(w1,w0)]+1
    else:                                                       # if the bigram has a zero count
        c_star = lm.frequencies[1] / len(lm.unigrams.keys())
        if w1 in lm.unigrams.keys():
            return c_star / lm.unigrams[w1]
        else:
            return c_star / lm.unknown_unigram

    # Check bigram frequency exists
    try:
        n_c1 = lm.frequencies[lm.bigrams[(w1,w0)]+1]
        n_c = lm.frequencies[lm.bigrams[(w1,w0)]]
        c_star = c * n_c1 / n_c
    except(KeyError):
        #print('KEY ERROR')
        c_star = c

    if w1 in lm.unigrams.keys():
        prob = c_star / lm.unigrams[w1]
    else:
        prob = c_star / 1

    return prob

# Might have to take absolute values when multipling total prob * log_prob
def get_prob_of_sentence(lm: LanguageModel, sentence: str):
    tokens = nltk.word_tokenize(sentence)
    total_prob = 0
    prev_token = ''
    for token in tokens:
        word_prob = gt_get_prob_of_w(lm, prev_token, token)
        log_prob = abs(math.log10(word_prob))
        if total_prob == 0:     # make sure we are not multiplying by 0
            total_prob = log_prob
        else:
            total_prob = total_prob * log_prob
        prev_token = token

    return total_prob

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

    # Test language models on training_texts
    for fname in training_txt_fnames:
        correct_guess_count = 0
        total_guesses = 0
        author_name = re.sub('.txt', '', fname)
        author_name = re.sub('training_', '', author_name)

        with open(fname, 'r') as training_txt:
            for line in training_txt:
                highest_prob = -1
                author = ''
                for lm in models:
                    #print('lm auth: %s'%lm.author)
                    new_prob = get_prob_of_sentence(lm, line)
                    if new_prob > highest_prob:
                        #print('Changing auth from %s to %s'%(author,lm.author))
                        highest_prob = new_prob
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

    # Test each language model on the fname
    with open(test_fname, 'r') as f:
        count = 1
        for line in f:
            line = line.strip()
            highest_prob = -1
            author = ''
            for lm in models:
                new_prob = get_prob_of_sentence(lm, line)
                if new_prob > highest_prob:
                    highest_prob = new_prob
                    author = lm.author

            print('Line %i: %s'%(count, author))
            count+=1

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


