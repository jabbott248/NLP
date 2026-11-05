# Created By Jacob Abbott, 03/15/2022
import json
import math
import re
import sys
from collections import Counter

import nltk

# Cost (-log10 probability) used for any tag transition or (word, tag) pair
# that never appears in the Brown corpus. 10 means a probability of 1e-10,
# rarer than anything the model has seen, so unseen combinations score badly
# instead of crashing the program.
UNSEEN_COST = 10


# Takes the brown corpus, and creates emmission and transition
# Matricies, then writes them to a file named model_dat.json
def hmm_model_build():
    matricies = create_matricies()
    m1 = matricies[0]
    m2 = matricies[1]
    # Write matricies to model_dat.json
    write_matrix_to_file(m1,m2)

def create_vocab(corpus):
    vocab = set(w.lower() for w in nltk.corpus.words.words())
    word_counts = Counter(
        word.lower() for sentence in corpus for word, _ in sentence)
    vocab.update(word for word, count in word_counts.items() if count > 1)
    return vocab

## Helpers for hmm_model_build()
# for each tag: counts number of time each tag occurs
# for each word: counts number of times each tag has a word
# unknown words are treated as an unknown token
# Creates probability matrix that a word has a tag
def create_matricies():
    corpus = nltk.corpus.brown.tagged_sents()
    vocab = create_vocab(corpus)
    tag_counts = {}
    word_counts = {}
    wordTag_counts = {}
    tagTag_counts = {}
    sentence_count = 0
    for sentence in corpus: # For each sentence in the corpus
        sentence_count += 1
        prev_tag = '<ST>'
        for word_tag in sentence:
            word = word_tag[0]
            word = word.lower()
            tag = word_tag[1]
            if not word in vocab: # check that word is in vocab
                word = 'UNK'
            # update tag counts
            if tag in tag_counts.keys():
                tag_counts[tag]+=1
            else:
                tag_counts[tag] = 1
            # update word counts
            if word in word_counts.keys():
                word_counts[word]+=1
            else:
                word_counts[word] = 1
            # update word tag counts
            if (word,tag) in wordTag_counts.keys():
                wordTag_counts[(word,tag)]+=1
            else:
                wordTag_counts[(word,tag)] = 1
            # update tagTag counts
            if (tag,prev_tag) in tagTag_counts.keys():
                tagTag_counts[(tag,prev_tag)]+=1
            else:
                tagTag_counts[(tag,prev_tag)] = 1
            # udpate prev_tag
            prev_tag = tag
    # compute probability matricies
    emission_matrix = {}
    transition_matrix = {}
    # emmision matrix
    for wordTag_key, count in wordTag_counts.items():
        tag = wordTag_key[1]
        prob = count / tag_counts[tag]
        prob = abs(math.log10(prob))
        emission_matrix[(tag,wordTag_key)] = prob
    # transition matrix
    # '<ST>' (start of sentence) is a previous tag once per sentence, so
    # P(tag | <ST>) = (sentences starting with tag) / (number of sentences)
    prev_tag_counts = dict(tag_counts)
    prev_tag_counts['<ST>'] = sentence_count
    for tag_key in prev_tag_counts.keys():
        for tagTag_key in tagTag_counts.keys():
            if tag_key == tagTag_key[1]:
                prob = tagTag_counts[(tagTag_key)] / prev_tag_counts[tag_key]
                prob = abs(math.log10(prob))
                transition_matrix[(tag_key,tagTag_key)] = prob
    return (emission_matrix,transition_matrix)


## Helper for hmm_model_build()
def write_matrix_to_file(m1: dict, m2: dict):
    model = {
        'm1': _format_matrix(m1),
        'm2': _format_matrix(m2),
    }
    with open('model_dat.json', 'w') as f:
        json.dump(model, f)
        f.write('\n')
    return 0

def _format_matrix(matrix):
    """Convert tuple-keyed matrix entries to nested dictionaries for JSON.

    For example, (('NN', ('cat', 'NN')), 0.42) becomes
    {'NN': {'cat': 0.42}}.
    """
    grouped = {}
    for (key, (value, _)), cost in matrix.items():
        grouped.setdefault(key, {})[value] = cost
    return grouped

## Helper for hmm_sequence
# reads a JSON model file containing m1 and m2
# returns two matrix dictionaries
def read_matrix_file(fname: str):
    with open(fname, 'r') as f:
        model = json.load(f)
    m1 = {
        (tag, (word, tag)): cost
        for tag, words in model['m1'].items()
        for word, cost in words.items()
    }
    m2 = {
        (prev_tag, (tag, prev_tag)): cost
        for prev_tag, tags in model['m2'].items()
        for tag, cost in tags.items()
    }
    return m1,m2

# Big function
def hmm_sequence(model_fname: str, txt_fname: str):
    # Use the same vocabulary policy as model training.
    english_vocab = create_vocab(nltk.corpus.brown.tagged_sents())
    # Load matricies from model
    matricies = read_matrix_file(model_fname)
    emission_m = matricies[0]
    transition_m = matricies[1]
    # Get tags from transition matrix
    with open(txt_fname, 'r') as f:
        total_cost = 0 # sum of -log10 probabilities
        prev_tag = '<ST>'
        for line in f:  # file is only one line
            line = line.strip()
            tokens = nltk.word_tokenize(line)   
            for token in tokens:    # for every word/tag pair
                pattern = re.compile(r'(.+)/(.+)')     # Match word/tag
                match = re.match(pattern,token)   
                if match == None:  
                    continue
                word = match.group(1)
                word = word.lower()
                if not word in english_vocab:
                    word = 'UNK'
                tag = match.group(2)
                # Calculate probability of word/tag
                if (tag,(word,tag)) in emission_m:
                    emission_prob = emission_m[(tag,(word,tag))]
                elif (tag,('UNK',tag)) in emission_m:
                    # word never seen with this tag: treat it as an unknown word
                    emission_prob = emission_m[(tag,('UNK',tag))]
                else: # tag never seen with an unknown word either
                    emission_prob = UNSEEN_COST
                # transitions are stored as (prev_tag, (tag, prev_tag));
                # the first word uses the start transition ('<ST>', (tag, '<ST>'))
                transition_prob = transition_m.get((prev_tag,(tag,prev_tag)), UNSEEN_COST)
                # values are -log10 costs, so add them instead of multiplying
                total_cost = total_cost + emission_prob + transition_prob
                # update previous tag
                prev_tag = tag
    # convert the total cost back into a probability
    return 10 ** -total_cost

# Big function
# Tags each line of an untagged text file with the most likely tag sequence.
# Returns a list with one (tokens, tags, probability) tuple per non-empty line,
# where probability is P(words, tags) for the best tag sequence.
def hmm_viterbi(model_fname: str, txt_fname: str):
    # Use the same vocabulary policy as model training.
    english_vocab = create_vocab(nltk.corpus.brown.tagged_sents())
    # Load matricies from model
    emission_m, transition_m = read_matrix_file(model_fname)
    # Index emissions by word so each lookup is fast: word -> {tag: cost}
    emissions = {}
    for (tag, (word, _)), cost in emission_m.items():
        emissions.setdefault(word, {})[tag] = cost
    # Index transitions by tag pair: (prev_tag, tag) -> cost
    # (transitions are stored as (prev_tag, (tag, prev_tag)))
    transitions = {}
    for (prev_tag, (tag, _)), cost in transition_m.items():
        transitions[(prev_tag, tag)] = cost

    results = []
    with open(txt_fname, 'r') as f:
        for line in f:  # each line is one sentence
            tokens = nltk.word_tokenize(line.strip())
            if not tokens:
                continue
            tags, total_cost = viterbi_tag(tokens, english_vocab,
                                           emissions, transitions)
            # convert the total cost back into a probability
            results.append((tokens, tags, 10 ** -total_cost))
    return results

## Helper for hmm_viterbi()
# Finds the lowest-cost (most likely) tag sequence for one sentence.
# Costs are -log10 probabilities, so they are added and the smallest wins.
# Returns (tags, total_cost).
def viterbi_tag(tokens, english_vocab, emissions, transitions):
    def emission_costs(token):
        word = token.lower()
        if word not in english_vocab:
            word = 'UNK'
        # A word that is in the vocab but never appears in Brown is
        # treated as an unknown word
        return emissions.get(word, emissions['UNK'])

    def transition_cost(prev_tag, tag):
        return transitions.get((prev_tag, tag), UNSEEN_COST)

    # best[i][tag] = (lowest cost of any path that ends at word i with tag,
    #                 previous tag on that path)
    best = [{}]
    for tag, e_cost in emission_costs(tokens[0]).items():
        best[0][tag] = (transition_cost('<ST>', tag) + e_cost, None)

    for token in tokens[1:]:
        prev_column = best[-1]
        column = {}
        for tag, e_cost in emission_costs(token).items():
            # cheapest way to reach this tag from any tag at the previous word
            prev_tag, path_cost = min(
                ((p, cost + transition_cost(p, tag))
                 for p, (cost, _) in prev_column.items()),
                key=lambda pair: pair[1])
            column[tag] = (path_cost + e_cost, prev_tag)
        best.append(column)

    # Trace back from the cheapest tag at the last word
    tag = min(best[-1], key=lambda t: best[-1][t][0])
    total_cost = best[-1][tag][0]
    tags = [tag]
    for i in range(len(tokens) - 1, 0, -1):
        tag = best[i][tag][1]
        tags.append(tag)
    tags.reverse()
    return tags, total_cost

def download_nltk_data():
    # 'punkt' is used by NLTK before 3.8.2, 'punkt_tab' by 3.8.2 and later
    resources = [
        ('corpora/brown', 'brown'),
        ('corpora/words', 'words'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            if not nltk.download(package):
                sys.exit(f"Error: could not download NLTK package '{package}'. "
                         "Check your internet connection (on macOS, see the "
                         "certificate step in the README) and try again.")

def main():
    # Download required NLTK data if not present
    download_nltk_data()

    # Check command-line arguments
    if len(sys.argv) == 1:
        # No arguments: build the model
        print("Building HMM model from Brown corpus...")
        hmm_model_build()
        print("Model saved to model_dat.json")
    elif len(sys.argv) == 3:
        # Two arguments: run hmm_sequence
        model_fname = sys.argv[1]
        txt_fname = sys.argv[2]
        print(f"Running HMM sequence on {txt_fname} using model {model_fname}")
        result = hmm_sequence(model_fname, txt_fname)
        print(f"Total probability: {result}")
    elif len(sys.argv) == 4 and sys.argv[1] == '--tag':
        # --tag <model> <textfile>: tag untagged text with hmm_viterbi
        model_fname = sys.argv[2]
        txt_fname = sys.argv[3]
        print(f"Running HMM Viterbi on {txt_fname} using model {model_fname}")
        for tokens, tags, probability in hmm_viterbi(model_fname, txt_fname):
            print(' '.join(f'{token}/{tag}' for token, tag in zip(tokens, tags)))
            print(f"Probability of best tagging: {probability}")
    else:
        print("Usage:")
        print("  python HMM_POS_Tagger.py                          (builds model_dat.json)")
        print("  python HMM_POS_Tagger.py <model> <textfile>       (runs hmm_sequence)")
        print("  python HMM_POS_Tagger.py --tag <model> <textfile> (runs hmm_viterbi)")
        
   # hmm_model_build()


if __name__ == '__main__':
   main()