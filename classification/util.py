from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd


def load_dataset(dataset_name='20ng'):
    if dataset_name == '20ng':
        return load_20ng()
    elif dataset_name == 'agnews':
        return load_agnews()
    else:
        return None

def load_agnews():
    ag_train = pd.read_csv('/exp/datasets/agnews/train.csv', header=None)
    ag_test = pd.read_csv('/exp/datasets/agnews/test.csv', header=None)
    ag_train.columns = ['Topic', 'Title', 'Article']
    ag_test.columns = ['Topic', 'Title', 'Article']
    ag_news = pd.concat([ag_train, ag_test[1:]])
    ag_news['text'] = ag_news.Title + " " + ag_news.Article
    X = ag_news['text'][1:].to_list()
    y = [int(s) for s in ag_news['Topic'][1:].to_list()]
    return (X, y, None)

def load_20ng(categories=None, subset=False):
    remove = ('headers', 'footers', 'quotes')
    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")
    result = []
    if subset:
        data_train = fetch_20newsgroups(subset='train', categories=categories,
                                        shuffle=True, random_state=42,
                                        remove=remove)
        data_test = fetch_20newsgroups(subset='test', categories=categories,
                                       shuffle=True, random_state=42,
                                       remove=remove)
        result = (data_train.data, data_test.data, data_train.target,
                  data_test.target, data_train.target_names)
    else:
        data_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42,
                                        remove=remove)
        result = (data_train.data, data_train.target, data_train.target_names)
    print('data loaded')
    return result


def text_preproc(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords = nltk.corpus.stopwords.words('english') + ['would']
    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = pattern.sub('', docs[idx])  # remove stopwords
        # remove non-alphabet characters
        docs[idx] = re.sub(r'[^a-z]', ' ', docs[idx])
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove words that are only two character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    # Lemmatize all words in documents.        
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # create strings
    docs = [' '.join(doc) for doc in docs]
    return docs
