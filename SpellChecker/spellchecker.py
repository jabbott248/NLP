# Goal: Create a basic spellchecker
from typing import List, Tuple
import sys
import re

# Trie Node system, used to store our dictionary
class TrieNode(object):
    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.word_finished = False  # Determines if it is the last character of the word
        self.counter = 1            # How many times this character appeared in the insertion process

# inserts a word into the trie structure
def insert(root, word: str):
    # defines a pointer to the root of the trie
    cur_node = root 
    # for every character in the given word:
    for char in word:
        found_in_child = False
        # try to find same char in the children of the current node
        for child in cur_node.children:
            if child.char == char: # match
                child.counter += 1      # increase the amount of times this char appeard in the insertion process
                cur_node = child        # point the current node to this child
                found_in_child = True   # set found_in_child to true
                break                   # can stop searching nodes children for matches as one has already been found

        if not found_in_child:  # If a match has not been found
            new_node = TrieNode(char)           # create a new node representing the current char
            cur_node.children.append(new_node)  # appends new node to current nodes children
            cur_node = new_node                 # point node to the newly created node

    cur_node.word_finished = True

# returns the amount of times a prefix occurs in a trie, and a bool if it is a full word
def find_prefix(root, prefix: str) -> Tuple[bool, int]:
    cur_node = root  # pointer to the root of the trie
    if not root.children:   # If the root has no children, trie is empty so the prefix does not exist
        return False, 0

    for char in prefix:
        char_not_found = True
        for child in cur_node.children:
            if child.char == char: # match of current char
                char_not_found = False 
                cur_node = child # move the pointer to matched child
                break

        if char_not_found:  # not all chars were found, prefix occurs 0 times
            return False, 0

    # since we passed previous return statement, we have found a full match
    # check if this is the end of a word:
    return cur_node.word_finished, cur_node.counter

# checks if a word exists in the dictionary
def word_exists(root, word: str) -> bool:
    ret_tuple = find_prefix(root, word)
    return ret_tuple[0]

# creates a pointer to the root of a trie created by all words in a dictionary
def trie_dict(fname):
    root = TrieNode('*')
    with open(fname) as f:
        for line in f:
            words = line.split()
            for word in words:
                insert(root, word)
    return root

def add_punction_to_trie(root):
    insert(root, ',')
    insert(root, '.')
    insert(root, '\\')
    insert(root, '\'')
    insert(root, '"')
    insert(root, '/')
    insert(root, '?')
    insert(root, '*')
    insert(root, '!')
    insert(root, '@')
    insert(root, '#')
    insert(root, '$')
    insert(root, '%')
    insert(root, '^')
    insert(root, '&')
    insert(root, '(')
    insert(root, ')')
    insert(root, '_')
    insert(root, '-')
    insert(root, '+')
    insert(root, '=')
    insert(root, '~')
    insert(root, '`')
    insert(root, ';')
    insert(root, '—')

# notes
def get_similar_words(root, word: str) -> list[str]:
    # follow tree as far as possible with prefixes that do exist
    cur_node = root  # pointer to the root of the trie
    prefix = ''
    if not root.children:   # If the root has no children, trie is empty so the prefix does not exist
        return []           # return the empty array
    
    for char in word:                       # iterate through each charcter of the input word
        char_not_found = True
        for child in cur_node.children:     # iterate through each child of current char's child
            if child.char == char:
                char_not_found = False
                prefix = prefix + char  # add current char to the prefix string
                cur_node = child        # move the pointer to matched child
                break                   # continue to the next node

        if char_not_found:      # begin adding all words with this prefix to the return array
            return get_all_children(cur_node, prefix)
                   # entire word matches

    return []

# helper function to return all children of a node in a trie structure
def get_all_children(root, prefix: str) -> list[str]:
    ret_lst = []
    if not root.children:   # If the root has no children, trie is empty so the prefix does not exist
        return []
    
    for child in root.children:
        if child.word_finished:
            ret_lst.append(prefix + child.char)
        else:
            new_prefix = prefix + child.char
            new_children = get_all_children(child,new_prefix)
            for new_child in new_children:
                ret_lst.append(new_child)

    return ret_lst

# takes a word and given a list of words return the closest words
def suggest(words: list[str], source: str) -> list[str]:
    ret_lst = []
    for word in words:
        dist = edit_dist(source, word)
        if len(ret_lst) < 3:
            ret_lst.append(word)
        else:
            if dist > edit_dist(source, ret_lst[2]):        # dist is not a minumum distance
                continue
            elif dist >= edit_dist(source, ret_lst[2]):     # dist is a minimum distance
                # add new distance in
                ret_lst[2] = word
                # sort ret_list based on edit distance
                ret_lst.sort(key=lambda str: edit_dist(source,str))

    return ret_lst

# TOKENIZATION
# param: fname, file name to be read
# returns: a list of strings containing individual tokens
def tokenize(fname):
    with open(fname) as fp:
        tokens = []
        for line in fp:
            # get rid of unambiguous punctuation
            line = re.sub(r'([\\?!()\";/\\|`])', r' \1 ', line)
            # seperate out commas unless they're in numbers
            line = re.sub(r'([^0-9]),', r'\1 , ', line)
            line = re.sub(r',([^0-9])', r' , \1', line)
            # distinguish apostrophes from single quotes (single qoutes not preceded by a letter)
           # line = re.sub(r'([^\'])', r'\1 ', line)
           # line = re.sub(r"([^A-Za-z0-9])'", r"\1 '", line) # non letter followed by an apostrophe
            # pull off word-final clitics.
            # deal with periods at the end of words
            line = re.sub(r'(.)\.', r'\1 . ', line)
            tokens.extend(line.split())
    return tokens
    
# EDIT DISTANCE, from book figure 3.25
cost_insertion = 1
cost_deletion = 1
cost_substitution = 2
# param: t(arget), intended word
# param: s(ource), actual word
# returns: minimum edit distance from target to source
def edit_dist(s,t):
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]

    # intialize source compared to the empty string
    for i in range(1, rows):
        dist[i][0] = i

    # intialize target compared to the empty string
    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                sub_cost = 0
            else:
                sub_cost = cost_substitution
            dist[row][col] = min(dist[row-1][col] + cost_deletion,      # deletion
                                 dist[row][col-1] + cost_insertion,     # insertion
                                 dist[row-1][col-1] + sub_cost)         # substitution
    # helper code to print out matrices
    # for r in range(rows):
    #   print(dist[r])
    return dist[row][col]

# INTERFACE
def interface():
    # check for correct number of command line arguments
    if len(sys.argv) != 2:
        print("Incorrect number of arguments")
        return
    # root of dictionary
    root = trie_dict('/files/words')
    # add punctuation to dictionary
    add_punction_to_trie(root)

    fname = sys.argv[1]
    new_fname = 'corrected_' + fname
    tokens = tokenize(fname)
    with open(new_fname, 'w') as f:
        for word in tokens:
            if word_exists(root, word):
                f.write(word + ' ')
            else:
                similar_words = get_similar_words(root, word)
                suggestions = suggest(similar_words, word) 
                if len(suggestions) > 0: # if there is a spell correction:
                    print('Changing %s to %s'%(word,suggestions[0]))    # prints word and line
                    f.write(suggestions[0] + ' ')                             # write correct spelling
                else:
                    print('No suggestions for: %s'%(word))              # print no correction found
                    f.write(word + ' ')                                       # write same word

    # put the file back together using regex

    # load a file whose name is given as a command-line argument.
    # look for words that are incorrectly spelled. prints out the line that the word appeared in
    # save its corrected version to a new file with the prefix `corrected_`.
def test_trie():
    root = trie_dict('/files/words')
    print(len(root.children))
    print(str(word_exists(root, 'abondon')))

def test_edit_distance(x,y):
    dist = edit_dist(x,y)
    print(dist)


def test_find_similar_words():
    root = trie_dict('words')
    example_word = 'abondon'
    if word_exists(root, example_word):
        print('word exists')
        return []
    else:
        res = get_similar_words(root, example_word)
        for word in res:
            print(word)
        return res

def test_tokenize():
    ret = tokenize('test_tokenize.txt')
    for word in ret:
        print(word)

if __name__ == "__main__":
    interface()
