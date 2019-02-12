# transform text into feature vector
# TF.IDF emphasizes the most important words for a given document

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# remove stopwords
# change alpha parameter of NB algo
# ignore words that appear fewer than 5 times
# use NLTK tokenizer to split words,
#   bring words back to base form using stemmer and
#   filter punctuation

from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

nb = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                   stop_words=stopwords.words('english')+ list(string.punctuation),
                                   min_df=3)),
    ('classifier', MultinomialNB(alpha=1)),])