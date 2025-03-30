# Created By Jacob Abbott, 03/15/2022
import re
import nltk
import math



# Takes the brown corpus, and creates emmission and transition
# Matricies, then writes them to a file named model.dat
def hmm_model_build():
    matricies = create_matricies()
    m1 = matricies[0]
    m2 = matricies[1]
    # Write matricies to model.dat
    write_matrix_to_file(m1,m2)

## Helpers for hmm_model_build()
# for each tag: counts number of time each tag occurs
# for each word: counts number of times each tag has a word
# unknown words are treated as an unknown token
# Creates probability matrix that a word has a tag
def create_matricies():
    corpus = nltk.corpus.brown.tagged_sents()
    vocab = set(w.lower() for w in nltk.corpus.words.words())
    tag_counts = {}
    word_counts = {}
    wordTag_counts = {}
    tagTag_counts = {}
    for sentence in corpus: # For each sentence in the corpus
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
    for tag_key in tag_counts.keys():
        for wordTag_key in wordTag_counts.keys(): 
            if tag_key == wordTag_key[1]:
                prob = wordTag_counts[wordTag_key] / tag_counts[tag_key]
                prob = abs(math.log10(prob))
                emission_matrix[(tag_key,wordTag_key)] = prob
    # transition matrix
    for tag_key in tag_counts.keys():
        for tagTag_key in tagTag_counts.keys():
            if tag_key == tagTag_key[1]:
                prob = tagTag_counts[(tagTag_key)] / tag_counts[tag_key]
                prob = abs(math.log10(prob))
                transition_matrix[(tag_key,tagTag_key)] = prob
    return (emission_matrix,transition_matrix)


## Helper for hmm_model_build()
def write_matrix_to_file(m1: dict, m2: dict):
    with open('model.dat', 'w') as f:
        f.write('m1\n')
        for key in m1.keys():
            value = m1[key]
            f.write(f'{key},{value}\n')
        f.write('m2\n')
        for key in m2.keys():
            value = m2[key]
            f.write(f'{key},{value}\n')
    return 0

## Helper for hmm_sequence
# reads a matrix.dat file, looks for m1 and m2
# returns two matrix dictionaries
def read_matrix_file(fname: str):
    m1 = {} # emission
    m2 = {} # transition
    curr_matrix = -1
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip() # get rid of new line characters
            # check first line is 'm1'
            if line == 'm1':
                curr_matrix = 1
            elif line == 'm2':
                curr_matrix = 2
            # update appropriate matrix
            else:
                # ('AT', ('UNK', 'AT')),0.08493349258363193
                pattern = re.compile(r"\(['\"](.+)['\"], \(['\"](.+)['\"], ['\"](.+)['\"]\)\),(.+)")
                match = re.match(pattern, line)
                if match == None:
                    continue
                tag = match.group(1)
                bigram1 = match.group(2)
                bigram2 = match.group(3)
                prob = match.group(4)
                prob = float(prob)
                if prob == 0.0: # Don't add any zero values 
                    prob = 0.0003 
                if curr_matrix == 1: # Matches emission pattern
                    m1[(tag,(bigram1,bigram2))] = prob
                elif curr_matrix == 2: # Matches Transition pattern
                    m2[(tag,(bigram1,bigram2))] = prob
                
    return m1,m2

# Big function
def hmm_sequence(model_fname: str, txt_fname: str):
    # determine english vocab
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    # Load matricies from model
    matricies = read_matrix_file(model_fname)
    emission_m = matricies[0]
    transition_m = matricies[1]
    # Get tags from transition matrix
    with open(txt_fname, 'r') as f:
        total_prob = 1
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
                if (tag,(word,tag)) in emission_m.keys():
                    emission_prob = emission_m[(tag,(word,tag))]
                else: # if word tag bigram does not exist, use the probability that a tag follows an unknown word
                    emission_prob = emission_m[(tag,('UNK',tag))]
                # skip starting probability
                if prev_tag == '<ST>':
                    transition_prob = 1
                else:
                    transition_prob = transition_m[(tag,(prev_tag,tag))]
                prob = emission_prob * transition_prob
                # Update total probability
                total_prob = total_prob * prob
                # update previous tag
                prev_tag = tag
    return total_prob

# Big function
def hmm_viterbi(model_fname: str, txt_fname: str):
    # determine english vocab
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    # Load matricies from model
    matricies = read_matrix_file(model_fname)
    emission_m = matricies[0]
    transition_m = matricies[1]
    # Get tags from transition matrix
    words = []
    with open(txt_fname, 'r') as f:
        for line in f: # Only one line
            line = line.strip()
            tokens = nltk.word_tokenize(line)
    
    # Create list of words
    for word in tokens:
        word = word.lower()
        if not word in english_vocab:
            word = 'UNK'
        words.append(word)

    # iterate over all words
    POS_sequence = []
    prev_tag = '<ST>'
    # For each word: find best possible tag
    for word in words: 
        emission_prob = -1
        transition_prob = -1
        cur_tag = ''
        # For each tag:
        for e_key in emission_m.keys():
            cur_word = e_key[1][0]
            cur_tag = e_key[0]
            best_tag = ''
            best_prob = -1
            cur_e_prob = emission_m[e_key]
            cur_t_prob = -1
            if cur_word == word:    
                # search transition probabilities with tag
                for t_key in transition_m.keys(): # for every tag that the word has
                    if t_key[0] == cur_tag:
                        cur_t_prob = transition_m[t_key]
                        # Check if emision prob * transition prob is greater than cur best guess
                        if (cur_t_prob * cur_e_prob) > best_prob:
                            best_prob = cur_t_prob * cur_e_prob
                            best_tag = cur_tag
                        
            prev_tag = best_tag
        
        POS_sequence.append(best_tag)
                            

    print(f'{POS_sequence=}')

def main():
    hmm_model_build()


if __name__ == '__main__':
   main()