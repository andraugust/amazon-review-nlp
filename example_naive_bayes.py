import amazon_review_utils as azu
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pprint import pprint as pp
import numpy as np
import pickle
import os.path


cache_path = './data_cache.p'
data_path = './reviews_Amazon_Instant_Video_5.json.gz'
downsample_frac = 0.5


# import data
# check if preprocessed data is cached
if os.path.isfile(cache_path):
    print('Loading cached dataset...')
    D = pickle.load(open(cache_path,'rb'))
else:
    print('Making dataset...')
    D = azu.make_dataset(data_path)
    pickle.dump(D,open(cache_path,'wb'))

# define datasets
nrevs = len(D['ratings'])
print('Original dataset size: %i reviews' % nrevs)
print('Downsampling...')
keeps = np.random.choice(nrevs,int(nrevs*downsample_frac),replace=False)
print('Modeling %i reviews...' % len(keeps))
D = {k: np.array(D[k])[keeps] for k in D}

# define training and testing sets
Xall = D['comments']
yall = D['ratings']
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.33)


# fit and test
M = azu.NaiveBayes()
M.fit(Xtr,ytr,fmin=0.0005,verbose=True)
yM = M.predict(Xte)


# results
accuracy = accuracy_score(yte,yM)
print('accuracy = %f' % accuracy)
C = confusion_matrix(yte, yM, labels=['pos','neg'])        # C_ij = number of times labels[i] was predicted to be labels[j]
p0 = C[0,0]/np.sum(C[0,:])
p1 = C[1,1]/np.sum(C[1,:])
cn_accuracy = (p0+p1)/2
print('class-normalized accuracy = %f' % cn_accuracy)
print('confusion matrix = ')
pp(C)

# sentiment scores
sentiment_scores = M.dllikelihoods
for w in sorted(sentiment_scores, key=sentiment_scores.get, reverse=True):
    print(w, sentiment_scores[w])
