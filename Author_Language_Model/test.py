# a file to test different python things
import math
import nltk
from typing import List, Tuple

# use the good touring method to calculate probablity
def make_good_touring_counts(fname: str) -> Tuple:
    # Store unigram counts
    unigram_counts = {}
    # Store bigram counts
    bigram_counts = {}
    # Store frequency of bigram frequencies
    frequencies = {}
    tokens = []

    with open(fname, 'r') as f:
        prev_token = ''
        for line in f:
            new_tokens = nltk.word_tokenize(line)
            for token in new_tokens:
                # Add to all tokens
                tokens.append(token)
                # update unigram count
                if token in unigram_counts.keys():  
                    unigram_counts[token] += 1  # if token already exists, increase its count
                else:
                    unigram_counts[token] = 1   # if token does not exist, add it to the dictionary
                # update bigram count
                if (prev_token, token) in bigram_counts.keys():
                    bigram_counts[(prev_token, token)] += 1
                else:
                    bigram_counts[(prev_token, token)] = 1
                # update frequencies
                if bigram_counts[(prev_token, token)] in frequencies.keys():
                    frequencies[bigram_counts[(prev_token, token)]] += 1
                else:
                    frequencies[bigram_counts[(prev_token, token)]] = 1
                # update the previous token for bigram counts
                prev_token = token

    total_words = len(tokens)

    return (total_words, unigram_counts, bigram_counts, frequencies)

def test():
    d = {}
    lst = [1,2,3,4,5,1,2,3]
    for i in lst:
        if i in d.keys():
            d[i] += 1
        else:
            d[i] = 1

    for key in d.keys():
        print(f'key: %d, value: %d'%(key,d[key]))

def test2():
    x = 0.0881480373458931
    log10_x = math.log10(x)
    elog_x = math.log(x)
    print('log10_x: %f'%log10_x)
    print('elog_x: %f'%elog_x)

def test3():
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    print('jake' in english_vocab)

def test4():
    for _ in range(31):
        print("Hello World!")


if __name__ == "__main__":
    test4()