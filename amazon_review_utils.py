from collections import Counter
from functools import reduce
import numpy as np
import time
import gzip
import json
from nltk.stem import WordNetLemmatizer
import string



L = WordNetLemmatizer()

def make_dataset(path):
    '''
    Read datafile and preprocess.
    :param path: path to .json.gz file.
    :return: preprocessed dataset.
    '''
    if path[-8:] != '.json.gz':
        print('ERROR: datafile isn\'t .json.gz')
        exit()
    # read json.gz and preprocess
    return preprocess(read_data(path))


def read_data(path):
    '''
    Extract rating and comment string from path file.
    :param path: string path to data file.
    :return: dict {'ratings': [list of ratings] and 'comments': [list of ratings]}
    '''
    D = {'ratings':[], 'comments':[]}
    with gzip.open(path,'rt') as f:
        for l in f:
            rev = json.loads(l)
            D['ratings'].append(rev['overall'])
            D['comments'].append(rev['reviewText'])
    return D


def preprocess(D):
    '''
    Convert stars to pos/neg and preprocess comment string.
    :param D: Dataset
    :return: preprocessed D
    '''
    # convert stars to posneg
    D['ratings'] = [star2posneg(s) for s in D['ratings']]
    # preprocess comments
    D['comments'] = [comment2dict(process_comment(c)) for c in D['comments']]
    return D


def star2posneg(star,thresh=3):
    '''
    Converts star to positive/negative rating.
    e.g., if thresh=3 then
    5,4,3,2,1 -> 'pos','pos','pos','neg','neg'
    '''
    return 'pos' if star>=thresh else 'neg'


def process_comment(comment):
    '''
    Remove whitespace, remove punctuation, stem, lemmatize from comment string.
    :param comment: string
    :return: list of strings (words) stemmed and lemmatized and lowercase.
    '''
    words = []
    for w in comment.split():
        # remove punctuation, make lowercase, lemmatize
        w = L.lemmatize(w.translate(str.maketrans('','',string.punctuation)).lower(), pos='v')
        words.append(w)
    return words


def comment2dict(comment):
    '''
    Convert a list of strings to a dictionary of counts.
    :param comment: list of strings (words)
    :return: dictionary of word counts
    '''
    wdict = {}
    for w in comment:
        if w in wdict:
            wdict[w] += 1
        else:
            wdict[w] = 1
    return wdict



class NaiveBayes:

    def __init__(self):
        self.is_fit = False
        self.priors = {'pos': 0.0, 'neg': 0.0}        # priors for positive and negative reviews
        self.lpriors = {'pos': 0.0, 'neg': 0.0}        # log of priors for positive and negative reviews
        self.llikelihoods = {'pos':{},'neg':{}}
        self.dllikelihoods = {}           # dict difference in log likelihood between classes
        self.shared_words = {}            # set of word in both pos and neg reviews
        self.pos_only_words = {}
        self.neg_only_words = {}


    def fit(self,X,y,fmin=0.0,verbose=False):
        '''
        Fit a model to a set of training samples X having labels y.
        :param X: List of word count dicts.  One dict per review.
        :param y: List of ratings.
        :param fmin: remove word from dict if has <= fmin frequency
        :param verbose:
        :return: None
        '''
        if verbose: print('Fitting model...')
        if verbose: st = time.time()
        n_revs = len(y)                    # total number of reviews

        # compute priors
        if verbose: print('\tcomputing priors...')
        self.priors = Counter(y)           # count number of positive and negative training samples
        self.priors = {key: self.priors[key]/n_revs for key in self.priors}  # normalize
        self.lpriors = {key: np.log(self.priors[key]) for key in self.priors}

        # seperate positive and negative reviews
        Xpos = [X[i] for i in range(len(X)) if y[i]=='pos']
        Xneg = [X[i] for i in range(len(X)) if y[i]=='neg']

        # combine review dicts
        if verbose: print('\taccumulating word counts...')
        neg_dict = reduce(lambda x1,x2: Counter(x1)+Counter(x2), Xneg)  # slow here
        pos_dict = reduce(lambda x1,x2: Counter(x1)+Counter(x2), Xpos)  # slow here

        # remove infrequent words
        if fmin is not None:
            if verbose: print('\tremoving infrequent words...')
            n_pos_words = sum(pos_dict.values())
            n_neg_words = sum(neg_dict.values())
            thresh_pos = round(fmin*n_pos_words)
            thresh_neg = round(fmin*n_neg_words)
            pos_dict = {w: pos_dict[w] for w in pos_dict if pos_dict[w]>=thresh_pos}
            neg_dict = {w: neg_dict[w] for w in neg_dict if neg_dict[w]>=thresh_neg}
            if verbose: print('\t\tremoved pos words having <%i counts' % thresh_pos)
            if verbose: print('\t\tremoved neg words having <%i counts' % thresh_neg)

        # compute log likelihoods
        if verbose: print('\tcomputing log-likelihoods...')
        n_pos_words = sum(pos_dict.values())
        n_neg_words = sum(neg_dict.values())
        self.llikelihoods['pos'] = {w: np.log(pos_dict[w]/n_pos_words) for w in pos_dict}
        self.llikelihoods['neg'] = {w: np.log(neg_dict[w]/n_neg_words) for w in neg_dict}

        # determine shared words and uniqiue words
        pos_words = set(self.llikelihoods['pos'].keys())
        neg_words = set(self.llikelihoods['neg'].keys())
        self.shared_words = pos_words & neg_words
        self.pos_only_words = pos_words - neg_words
        self.neg_only_words = neg_words - pos_words

        # compute difference in log likelihood between pos and neg words *only if word is shared*
        self.dllikelihoods = {w: self.llikelihoods['pos'][w]-self.llikelihoods['neg'][w] for w in self.shared_words}
        self.is_fit = True
        if verbose: print('----DONE FITTING----')
        if verbose: print('Fit time = %.2fs' % (time.time()-st))
        if verbose: print('--------------------')


    def predict(self,X):
        '''
        Predict the sentiment of each dict in X
        :param X: List of word-count dicts
        :return: List of predicted sentiment (pos/neg) for each dict
        '''
        return [self.get_class(x) for x in X]


    def get_class(self,x):
        '''
        Get class of a word dict using a trained model
        :param x: word dict
        :return: predicted class
        '''
        if self.is_fit:
            acc_pos = 0
            acc_neg = 0
            for word in x:
                if word in self.shared_words:
                    acc_pos += x[word]*self.llikelihoods['pos'][word]
                    acc_neg += x[word]*self.llikelihoods['neg'][word]
                #elif word in self.pos_only_words:
                #    return 'pos'
                #elif word in self.neg_only_words:
                #    return 'neg'

            if acc_pos + self.lpriors['pos'] > acc_neg + self.lpriors['neg']:
                return 'pos'
            else:
                return 'neg'
        else:
            print('ERROR: can\'t get class: model hasn\'t been fit')
