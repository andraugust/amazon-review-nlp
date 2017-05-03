# amazon_review-utils
Utilities for preprocessing and analyzing [Amazon review data](http://jmcauley.ucsd.edu/data/amazon/).  Includes a Naive Bayes sentiment model.

Dependencies: [nltk](http://www.nltk.org/), [WordNetLemmatizer](http://www.nltk.org/_modules/nltk/stem/wordnet.html), [numpy](http://www.numpy.org/).

To download the corpus used by WordNetLemmatizer, do this in python:

```python
import nltk
nltk.download("wordnet", "/path/to/save/nltk_data")
```


# Usage example
Make a stemmed and lemmatized bag-of-words dataset:

```python
import amazon_review_utils as azu

D = azu.make_dataset('reviews_Amazon_Instant_Video_5.json.gz')

# D = 
```

Train a naive Bayes sentiment model.
```python
import amazon_review_utils as azu
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pprint import pprint as pp
import numpy as np


data_path = './reviews_Amazon_Instant_Video_5.json.gz'
D = azu.make_dataset(data_path)
# define training and testing sets
Xall = D['comments']
yall = D['ratings']
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.33)

## fit and test
M = azu.NaiveBayes()
M.fit(Xtr,ytr,verbose=True)
yM = M.predict(Xte)
```

Print model results
```
## print results
# performance
accuracy = accuracy_score(yte,yM)
print('accuracy = %f' % accuracy)
C = confusion_matrix(yte, yM, labels=['pos','neg'])        # C_ij = number of times labels[i] was predicted to be labels[j]
p0 = C[0,0]/np.sum(C[0,:])
p1 = C[1,1]/np.sum(C[1,:])
cn_accuracy = (p0+p1)/2
print('class-normalized accuracy = %f' % cn_accuracy)
print('confusion matrix = ')
pp(C)
```




```
