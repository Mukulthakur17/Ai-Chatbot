import nltk
# nltk.download('punkt')        # You only need to run this line once.
from nltk.stem.porter import PorterStemmer

def tokenize(sentance):
    return nltk.word_tokenize(sentance)

stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sents, all_words):
    pass


#Test your fxns:

#sentance = "How did Pain kill Jiraya?"
#print(sentance)
#tokens = tokenize(sentance)
#print(tokens)

#words = ['Organize', 'Organizing', 'Organizers']
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words)