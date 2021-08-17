import nltk
import numpy as np
# nltk.download('punkt')        # You only need to run this line once.
from nltk.stem.porter import PorterStemmer

def tokenize(sentance):
    # split sentence into array of words/tokens
    return nltk.word_tokenize(sentance)

stemmer = PorterStemmer()
def stem(word):
    # stemming = find the root form of the word
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sents, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    eg:
    sentence = ["hello", "how", "are", "you"]
    all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sents]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1
    
    return bag


#Test your fxns:

#sentance = "How did Pain kill Jiraya?"
#print(sentance)
#tokens = tokenize(sentance)
#print(tokens)

#words = ['Organize', 'Organizing', 'Organizers']
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words)

#sentence = ["hello", "how", "are", "you"]
#all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
#bog = bag_of_words(sentence, all_words)
#print(bog)