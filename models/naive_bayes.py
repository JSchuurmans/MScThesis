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

# # TODO write to class
# class NB(object):
#     def __init__(self):
#         super(NB).__init__(self)

#     def nb(self):
#         Pipeline([
#             ('vectorizer', TfidfVectorizer(tokenizer=self.stemming_tokenizer,
#                                    stop_words=stopwords.words('english')+ list(string.punctuation),
#                                    min_df=3)),
#             ('classifier', MultinomialNB(alpha=1)),])

#     def stemming_tokenizer(self, text):
#         stemmer = PorterStemmer()
#         return [stemmer.stem(w) for w in word_tokenize(text)]

#############################################################

### baseModel layout#####################################3
# def load_data():
#     df_test = pd.read_pickle()
#     df_train = pd.read_pickle()

#     X_test = df_test['utterance']
#     y_test = df_test['intent']

#     X = df_train['utterance']
#     y = df_train['intent']

# def load_model():
#     from models.naive_bayes import nb
#     model = nb

# def train():
#     model = nb.fit(X,y)
#     y_hat = model.predict(X_test)

#     prec_macro = precision_score(y_test, y_hat, average='macro')
#     rec_macro = recall_score(y_test,y_hat, average='macro')
#     f1_macro = f1_score(y_test,y_hat, average='macro')

#     print('recall macro: ' + str(rec_macro))
#     print('precision macro: ' + str(prec_macro))
#     print('F1 macro: ' + str(f1_macro))

#     # training score
#     y_hat_train = model.predict(X) 
#     prec_macro_train = precision_score(y, y_hat_train, average='macro')
#     rec_macro_train = recall_score(y,y_hat_train, average='macro')
#     f1_macro_train = f1_score(y,y_hat_train, average='macro')

#     print('recall macro train: ' + str(rec_macro_train))
#     print('precision macro train: ' + str(prec_macro_train))
#     print('F1 macro train: ' + str(f1_macro_train))